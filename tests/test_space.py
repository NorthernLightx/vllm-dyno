import dataclasses
import shlex

from vllm_dyno.space import Candidate, Context, engine_args, infeasible_reason, screen_candidates, serve_command


def reason(cand: Candidate, ctx: Context) -> str:
    return infeasible_reason(cand, ctx) or ""


def test_screen_starts_with_baseline_and_changes_one_setting_at_a_time(ctx):
    cands = screen_candidates(ctx)
    assert cands[0] == Candidate()
    assert all(len(c.changes()) == 1 for c in cands[1:])
    # weights: fp8 + 2 checkpoints, KV: 3, attention: 2, speculative: 1
    assert len(cands) == 1 + 3 + 3 + 2 + 1


def test_rules_explain_known_failures(ctx):
    assert "nvcc" in reason(Candidate(kv_cache_dtype="fp8"), ctx)
    assert "nvcc" in reason(Candidate(attention_backend="FLASHINFER"), ctx)
    assert "4 or 8 bits" in reason(Candidate(weights="org/base-GPTQ-3bit"), ctx)
    assert infeasible_reason(Candidate(weights="org/base-AWQ"), ctx) is None
    assert infeasible_reason(Candidate(), ctx) is None


def test_fp8_kv_is_allowed_when_nvcc_exists(ctx):
    with_nvcc = dataclasses.replace(ctx, host={**ctx.host, "nvcc": True})
    assert infeasible_reason(Candidate(kv_cache_dtype="fp8"), with_nvcc) is None


def test_memory_rule_rejects_weights_larger_than_the_budget(ctx):
    big = dataclasses.replace(ctx, base_bytes=10 * 1024**3)
    assert "needs about" in reason(Candidate(), big)
    assert "needs about" not in reason(Candidate(weights="org/base-AWQ"), big)


def test_engine_args_only_carry_changed_settings(ctx):
    assert engine_args(Candidate(), ctx) == {"model": "org/base", "gpu_memory_utilization": 0.63, "max_model_len": 2048}
    fp8 = engine_args(Candidate(weights="fp8", attention_backend="TRITON_ATTN"), ctx)
    assert fp8["model"] == "org/base" and fp8["quantization"] == "fp8" and fp8["attention_backend"] == "TRITON_ATTN"
    assert engine_args(Candidate(weights="org/base-AWQ"), ctx)["model"] == "org/base-AWQ"
    assert engine_args(Candidate(speculative="ngram"), ctx)["speculative_config"]["method"] == "ngram"


def test_serve_command_round_trips_through_a_shell(ctx):
    cmd = serve_command(Candidate(weights="org/base-AWQ", speculative="ngram"), ctx)
    argv = shlex.split(cmd)
    assert argv[:3] == ["vllm", "serve", "org/base-AWQ"]
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.63"
    assert argv[argv.index("--speculative-config") + 1].startswith('{"method":"ngram"')


def test_context_survives_json(ctx):
    assert Context.from_json(ctx.to_json()) == ctx
