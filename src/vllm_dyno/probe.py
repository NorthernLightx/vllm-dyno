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


def memory_utilization(gpu: Gpu, *, wsl: bool = False, cap: float = 0.9) -> float:
    """Largest gpu_memory_utilization vLLM accepts now: it refuses to start unless free >= total * utilization.

    Under WSL the Windows desktop shares the GPU, its use grows as windows open, and WDDM moves
    allocations to system memory rather than refuse them, so more is left for the desktop to grow into.
    """
    reserve_mib = 1536 if wsl else 512
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


def serve_env(host: dict, util: float) -> dict[str, str]:
    """Variables vLLM needs on this host, given its host_facts() and gpu_memory_utilization, beyond the user's environment."""
    env = {}
    if not host.get("nvcc"):
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # its top-k/top-p sampler JIT-compiles with nvcc
    if host.get("wsl"):
        # Under WSL cudaMalloc spills into system memory instead of failing, and only a failure
        # makes torch's caching allocator release cached blocks. Blocks too small for what is asked
        # next, such as fp32 logits copies at a growing batch's size, then pile up past GPU memory.
        # A cap at vLLM's own budget makes torch release them there. Expandable segments are no
        # way out: at the end of GPU memory they fail with "CUDA driver error: device not ready",
        # which torch does not take for running out of memory, so it releases nothing either.
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"per_process_memory_fraction:{util}"
    return env


def child_env(host: dict, util: float) -> dict[str, str]:
    """Environment for trial processes and `vllm serve` on a host from host_facts(), at gpu_memory_utilization util."""
    env = dict(os.environ)
    # WSL appends Windows dirs to PATH. Exec into one Linux cannot search fails with EACCES,
    # and torch inductor's `nvcc --version` probe only handles ENOENT.
    env["PATH"] = os.pathsep.join(d for d in env.get("PATH", "").split(os.pathsep) if _searchable(d))
    for key, value in serve_env(host, util).items():
        env.setdefault(key, value)
    return env


def measured_env(env: dict[str, str], host: dict, util: float) -> dict[str, str | None]:
    """The values in env of the variables serve_env() sets, which measurements depend on."""
    return {key: env.get(key) for key in serve_env(host, util)}


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
