import os

from vllm_dyno.probe import AttentionShape, Gpu, child_env, kv_bytes_per_token, memory_utilization


def test_memory_utilization_leaves_free_memory_check_satisfied():
    gpu = Gpu("x", 8.6, total_mib=8192, free_mib=5676)
    util = memory_utilization(gpu)
    assert util == 0.63
    assert util * gpu.total_mib <= gpu.free_mib


def test_memory_utilization_is_capped():
    assert memory_utilization(Gpu("x", 9.0, total_mib=80000, free_mib=79500)) == 0.9


def test_child_env_drops_unsearchable_path_entries_and_disables_flashinfer_sampler(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path), "/no/such/dir"]))
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    env = child_env(nvcc=False)
    assert env["PATH"] == str(tmp_path)
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "VLLM_USE_FLASHINFER_SAMPLER" not in child_env(nvcc=True)


def test_kv_bytes_per_token_for_qwen2_5_1_5b():
    shape = AttentionShape.from_hf_config(
        {"num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "hidden_size": 1536}
    )
    assert shape == AttentionShape(28, 2, 128)
    assert kv_bytes_per_token(shape) == 28 * 1024
    assert kv_bytes_per_token(shape, 1) == 14 * 1024
