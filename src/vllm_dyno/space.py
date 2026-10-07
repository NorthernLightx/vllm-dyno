"""The search space: which settings a trial can change, and which combinations cannot run here."""

from __future__ import annotations

import dataclasses
import json
import logging
import shlex
from dataclasses import dataclass

from .probe import GIB, MIB, AttentionShape, Gpu, kv_bytes_per_token, model_files

log = logging.getLogger(__name__)

KNOBS = ("weights", "kv_cache_dtype", "attention_backend", "speculative")
DEFAULTS = {"weights": "base", "kv_cache_dtype": "auto", "attention_backend": "auto", "speculative": "none"}
VALUES = {
    "kv_cache_dtype": ["auto", "fp8", "int8_per_token_head", "turboquant_k8v4"],
    "attention_backend": ["auto", "TRITON_ATTN", "FLASHINFER"],
    "speculative": ["none", "ngram"],
}
NGRAM = {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 4}
OVERHEAD_BYTES = GIB  # activations and CUDA graphs; a rough figure, used only to rule out configs that cannot fit

SKIP_TAGS = {"gguf", "onnx", "mlc", "openvino", "mlx", "exl2", "coreml"}
QUANT_METHODS = {"awq", "gptq", "compressed-tensors", "fp8", "bitsandbytes"}


@dataclass(frozen=True)
class Candidate:
    weights: str = "base"  # "base", "fp8" (quantized while loading) or a quantized checkpoint repo
    kv_cache_dtype: str = "auto"
    attention_backend: str = "auto"
    speculative: str = "none"

    def changes(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in KNOBS if getattr(self, k) != DEFAULTS[k]}

    def label(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.changes().items()) or "baseline"

    def replace(self, **changes: str) -> Candidate:
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    method: str
    bits: int | None
    weight_bytes: int
    revision: str | None = None  # commit measured; serving the same commit rules out a repo changed since


@dataclass(frozen=True)
class Context:
    """Everything that decides feasibility and engine arguments. Stored in run.json so a resumed run matches."""

    model: str
    gpu: Gpu
    host: dict
    shape: AttentionShape
    base_bytes: int
    checkpoints: tuple[Checkpoint, ...]
    util: float
    max_model_len: int
    revision: str | None = None  # commit of the base model

    def checkpoint(self, repo: str) -> Checkpoint | None:
        return next((c for c in self.checkpoints if c.repo == repo), None)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Context:
        return cls(
            model=d["model"],
            gpu=Gpu(**d["gpu"]),
            host=d["host"],
            shape=AttentionShape(**d["shape"]),
            base_bytes=d["base_bytes"],
            checkpoints=tuple(Checkpoint(**c) for c in d["checkpoints"]),
            util=d["util"],
            max_model_len=d["max_model_len"],
            revision=d.get("revision"),
        )


def weight_values(ctx: Context) -> list[str]:
    return ["base", "fp8", *(c.repo for c in ctx.checkpoints)]


def engine_args(c: Candidate, ctx: Context) -> dict:
    ckpt = ctx.checkpoint(c.weights)
    args: dict = {"model": ckpt.repo if ckpt else ctx.model}
    revision = ckpt.revision if ckpt else ctx.revision
    if revision:
        args["revision"] = revision
    args |= {"gpu_memory_utilization": ctx.util, "max_model_len": ctx.max_model_len}
    if c.weights == "fp8":
        args["quantization"] = "fp8"
    if c.kv_cache_dtype != "auto":
        args["kv_cache_dtype"] = c.kv_cache_dtype
    if c.attention_backend != "auto":
        args["attention_backend"] = c.attention_backend
    if c.speculative == "ngram":
        args["speculative_config"] = NGRAM
    return args


def serve_command(c: Candidate, ctx: Context) -> str:
    args = engine_args(c, ctx)
    parts = ["vllm", "serve", args.pop("model")]
    for key, value in args.items():
        parts += [
            f"--{key.replace('_', '-')}",
            json.dumps(value, separators=(",", ":")) if isinstance(value, dict) else str(value),
        ]
    return shlex.join(parts)


def _weight_bytes(c: Candidate, ctx: Context) -> int:
    if c.weights == "base":
        return ctx.base_bytes
    if c.weights == "fp8":
        return ctx.base_bytes // 2  # one byte per bf16 weight
    ckpt = ctx.checkpoint(c.weights)
    return ckpt.weight_bytes if ckpt else ctx.base_bytes


def infeasible_reason(c: Candidate, ctx: Context) -> str | None:
    """Why this candidate cannot run on this machine, or None if it might.

    Rules cover failures known before loading anything. Everything else is found by trying.
    """
    cc, nvcc = ctx.gpu.compute_cap, ctx.host.get("nvcc", False)
    ckpt = ctx.checkpoint(c.weights)
    if c.kv_cache_dtype.startswith("fp8") and cc < 8.9 and not nvcc:
        return "fp8 KV cache below compute capability 8.9 needs FlashInfer, which compiles with nvcc (not on PATH)"
    if c.attention_backend.startswith("FLASHINFER") and not nvcc:
        return "FlashInfer compiles its kernels with nvcc, which is not on PATH"
    if c.weights == "fp8" and cc < 8.0:
        return "fp8 weight-only kernels need compute capability 8.0+"
    if ckpt and ckpt.method == "gptq" and ckpt.bits not in (4, 8):
        return f"vLLM runs GPTQ at 4 or 8 bits; this checkpoint is {ckpt.bits}-bit"
    if ckpt and ckpt.method == "bitsandbytes" and not ctx.host.get("bitsandbytes"):
        return "the bitsandbytes package is not installed"
    kv_dtype_bytes = 2 if c.kv_cache_dtype in ("auto", "bfloat16", "float16") else 1
    need = _weight_bytes(c, ctx) + kv_bytes_per_token(ctx.shape, kv_dtype_bytes) * ctx.max_model_len + OVERHEAD_BYTES
    budget = ctx.util * ctx.gpu.total_mib * MIB
    if need > budget:
        return f"needs about {need / GIB:.1f} GiB for one {ctx.max_model_len}-token sequence; budget is {budget / GIB:.1f} GiB"
    return None


def checkpoint_from(repo: str, methods: set[str] = QUANT_METHODS) -> Checkpoint | None:
    """The repo as a Checkpoint if its config declares one of `methods`, else None."""
    cfg, size, revision = model_files(repo)
    q = cfg.get("quantization_config") or {}
    method = q.get("quant_method")
    if method not in methods:
        return None
    bits = q.get("bits") or q.get("w_bit") or (4 if q.get("load_in_4bit") else 8 if q.get("load_in_8bit") else None)
    return Checkpoint(repo, method, bits, size, revision)


def owner(repo: str) -> str:
    return repo.split("/", 1)[0]


def discover_checkpoints(
    model: str, limit: int, methods: set[str] = QUANT_METHODS, any_owner: bool = False, scan: int = 50
) -> list[Checkpoint]:
    """Most downloaded quantized derivatives of `model` on the Hub, one per (method, bits), in `methods`.

    Unless `any_owner`, only repos published by the base model's owner count: a recommendation
    puts these weights into production, and anyone can upload a derivative.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    found: dict[tuple[str, int | None], Checkpoint] = {}
    for m in HfApi().list_models(filter=f"base_model:quantized:{model}", sort="downloads", limit=scan):
        tags = set(m.tags or [])
        if tags & SKIP_TAGS or "safetensors" not in tags:
            continue
        if not any_owner and owner(m.id) != owner(model):
            continue
        try:
            ckpt = checkpoint_from(m.id, methods)
        except (HfHubHTTPError, EntryNotFoundError, OSError, ValueError) as e:  # gated, deleted or malformed repos
            log.info("skip     %s: %s", m.id, type(e).__name__)
            continue
        if ckpt:
            found.setdefault((ckpt.method, ckpt.bits), ckpt)
        if len(found) == limit:
            break
    return list(found.values())


def screen_candidates(ctx: Context) -> list[Candidate]:
    """The baseline, then every non-default value of one setting at a time."""
    values = {"weights": weight_values(ctx), **VALUES}
    base = Candidate()
    return [base] + [base.replace(**{k: v}) for k in KNOBS for v in values[k] if v != DEFAULTS[k]]
