"""Facts about this machine and the model that decide which configs can run.

Nothing here touches CUDA: a CUDA context in the parent process would hold GPU
memory for the whole run and shrink what every trial gets.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class Gpu:
    name: str
    compute_cap: float
    total_mib: int
    free_mib: int


def query_gpu() -> Gpu:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    name, cap, total, free = (s.strip() for s in out.strip().split(","))
    return Gpu(name, float(cap), int(total), int(free))


def memory_utilization(gpu: Gpu, reserve_mib: int = 512, cap: float = 0.9) -> float:
    """Largest gpu_memory_utilization vLLM accepts now: it refuses to start unless free >= total * utilization."""
    return min(cap, math.floor((gpu.free_mib - reserve_mib) / gpu.total_mib * 100) / 100)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _searchable(d: str) -> bool:
    return os.path.isdir(d) and os.access(d, os.X_OK)


def host_facts() -> dict:
    proc_version = Path("/proc/version")
    return {
        "vllm": package_version("vllm"),
        "nvcc": shutil.which("nvcc") is not None,
        "wsl": proc_version.exists() and "microsoft" in proc_version.read_text().lower(),
        "bitsandbytes": importlib.util.find_spec("bitsandbytes") is not None,
        "unsearchable_path": sum(not _searchable(d) for d in os.environ.get("PATH", "").split(os.pathsep) if d),
    }


def serve_env(nvcc: bool) -> dict[str, str]:
    """Variables vLLM needs on this host beyond the user's environment."""
    return {} if nvcc else {"VLLM_USE_FLASHINFER_SAMPLER": "0"}  # its top-k/top-p sampler JIT-compiles with nvcc


def child_env(nvcc: bool) -> dict[str, str]:
    """Environment for trial processes and `dyno serve`."""
    env = dict(os.environ)
    # WSL appends Windows dirs to PATH. Exec into one Linux cannot search fails with EACCES,
    # and torch inductor's `nvcc --version` probe only handles ENOENT.
    env["PATH"] = os.pathsep.join(d for d in env.get("PATH", "").split(os.pathsep) if _searchable(d))
    for key, value in serve_env(nvcc).items():
        env.setdefault(key, value)
    return env


@dataclass(frozen=True)
class AttentionShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_hf_config(cls, cfg: dict) -> AttentionShape:
        cfg = cfg.get("text_config", cfg)  # multimodal configs nest the language model
        heads = cfg["num_attention_heads"]
        return cls(
            num_layers=cfg["num_hidden_layers"],
            num_kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
        )


def kv_bytes_per_token(shape: AttentionShape, dtype_bytes: float = 2) -> int:
    """Every layer stores one K and one V vector per KV head for each token."""
    return int(2 * shape.num_layers * shape.num_kv_heads * shape.head_dim * dtype_bytes)


def model_files(repo: str) -> tuple[dict, int, str | None]:
    """config.json, the total size of the safetensors weights and the current commit, without downloading weights."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo, files_metadata=True)
    cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=info.sha)).read_text())
    weight_bytes = sum(s.size or 0 for s in info.siblings or [] if s.rfilename.endswith(".safetensors"))
    return cfg, weight_bytes, info.sha
