"""End-to-end trials on a real GPU: `uv run pytest -m gpu`."""

import json

import pytest

from vllm_dyno import data
from vllm_dyno.probe import AttentionShape, child_env, host_facts, memory_utilization, model_files, query_gpu
from vllm_dyno.runner import Run, run_trial
from vllm_dyno.search import POOL_SIZE, SCREEN
from vllm_dyno.space import Candidate, Context

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

pytestmark = [pytest.mark.gpu, pytest.mark.filterwarnings("ignore")]


def test_baseline_writes_a_reference_and_a_rerun_matches_it(tmp_path):
    gpu, host = query_gpu(), host_facts()
    cfg, weight_bytes, revision = model_files(MODEL)
    shape = AttentionShape.from_hf_config(cfg)
    ctx = Context(MODEL, gpu, host, shape, weight_bytes, (), memory_utilization(gpu), 2048, revision)
    run = Run(tmp_path)
    run.write_meta({"free_mib_at_start": gpu.free_mib})
    built = data.build(MODEL, revision, None, SCREEN["eval_tokens"], POOL_SIZE)
    (tmp_path / "data.json").write_text(json.dumps(built))
    env = child_env(host["nvcc"])

    base = run_trial(run, ctx, "screen", Candidate(), SCREEN, write_reference=True, env=env, timeout_s=900)
    assert base["status"] == "ok", base.get("error")
    assert (tmp_path / "reference.npz").exists()
    assert base["kv_tokens"] > 0 and base["tput_tok_s"] > 0 and base["tpot_ms"] > 0 and base["ttft_ms"] > 0
    assert 1 < base["ppl"] < 100

    rerun = run_trial(run, ctx, "rerun", Candidate(), SCREEN, write_reference=False, env=env, timeout_s=900)
    assert rerun["status"] == "ok", rerun.get("error")
    assert rerun["kl"] < 0.01 and rerun["top1_agree"] > 0.97
    assert rerun["ppl"] == pytest.approx(base["ppl"], rel=0.01)
