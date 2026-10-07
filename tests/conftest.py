import pytest

from vllm_dyno.probe import AttentionShape, Gpu
from vllm_dyno.space import Checkpoint, Context

GIB = 1024**3


@pytest.fixture
def ctx() -> Context:
    """An RTX 3070 under WSL without nvcc, serving a Qwen2.5-1.5B-shaped model."""
    return Context(
        model="org/base",
        gpu=Gpu("NVIDIA GeForce RTX 3070", 8.6, 8192, 5676),
        host={"vllm": "0.31.0", "nvcc": False, "wsl": True, "bitsandbytes": False},
        shape=AttentionShape(num_layers=28, num_kv_heads=2, head_dim=128),
        base_bytes=int(2.9 * GIB),
        checkpoints=(
            Checkpoint("org/base-AWQ", "awq", 4, int(1.1 * GIB)),
            Checkpoint("org/base-GPTQ-3bit", "gptq", 3, int(0.9 * GIB)),
        ),
        util=0.63,
        max_model_len=2048,
    )
