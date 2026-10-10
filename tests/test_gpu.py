"""End-to-end trials on a real GPU: `uv run pytest -m gpu`."""

import asyncio
import json
import shlex

import pytest

from vllm_dyno import data
from vllm_dyno.load import KV_LINE, OUTPUT_LEN, Slo, batch_run, chat_probe, prompt_pool, send, serving
from vllm_dyno.probe import AttentionShape, child_env, host_facts, memory_utilization, model_files, query_gpu
from vllm_dyno.runner import Run, run_trial
from vllm_dyno.search import POOL_SIZE, SCREEN
from vllm_dyno.space import Candidate, Context, engine_args, serve_command

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

pytestmark = [pytest.mark.gpu, pytest.mark.filterwarnings("ignore")]


def context() -> Context:
    gpu, host = query_gpu(), host_facts()
    cfg, weight_bytes, revision = model_files(MODEL)
    shape = AttentionShape.from_hf_config(cfg)
    return Context(MODEL, gpu, host, shape, weight_bytes, (), memory_utilization(gpu), 2048, revision)


def test_baseline_writes_a_reference_and_a_rerun_matches_it(tmp_path):
    ctx = context()
    run = Run(tmp_path)
    run.write_meta({"free_mib_at_start": ctx.gpu.free_mib})
    built = data.build(MODEL, ctx.revision, None, SCREEN["eval_tokens"], POOL_SIZE)
    (tmp_path / "data.json").write_text(json.dumps(built))
    env = child_env(ctx.host, server=False)

    base = run_trial(run, ctx, "screen", Candidate(), SCREEN, write_reference=True, env=env, timeout_s=900)
    assert base["status"] == "ok", base.get("error")
    assert (tmp_path / "reference.npz").exists()
    assert base["kv_tokens"] > 0 and base["tput_tok_s"] > 0 and base["tpot_ms"] > 0 and base["ttft_ms"] > 0
    assert 1 < base["ppl"] < 100

    rerun = run_trial(run, ctx, "rerun", Candidate(), SCREEN, write_reference=False, env=env, timeout_s=900)
    assert rerun["status"] == "ok", rerun.get("error")
    assert rerun["kl"] < 0.01 and rerun["top1_agree"] > 0.97
    assert rerun["ppl"] == pytest.approx(base["ppl"], rel=0.01)


def test_vllm_serve_answers_requests_a_chat_probe_and_a_batch(tmp_path):
    ctx = context()
    pool = prompt_pool(data.build(MODEL, ctx.revision, None, SCREEN["eval_tokens"], POOL_SIZE))
    model = engine_args(Candidate(), ctx)["model"]
    log_path = tmp_path / "server.log"
    with serving(shlex.split(serve_command(Candidate(), ctx)), log_path, child_env(ctx.host, server=True), 600) as url:
        assert KV_LINE.search(log_path.read_text())
        sent = asyncio.run(send(url, model, pool[:2], [0.0, 0.0], OUTPUT_LEN))
        probe = chat_probe(url, model, pool, 2.0, Slo(), 5)
        batch = batch_run(url, model, pool, 32)
    # ignore_eos holds every answer to the requested length, and usage reports it
    assert [(t.tokens, t.error) for t in sent.timings] == [(OUTPUT_LEN, None)] * 2
    assert probe["errors"] == 0 and probe["finished"] == probe["requests"] == 20
    assert 0 < probe["ttft_p50_ms"] < 1000 and 0 < probe["tpot_p50_ms"] < 50
    assert probe["kv_peak"] > 0
    assert batch["errors"] == 0 and batch["finished"] == 32 and batch["tput_tok_s"] > 0
