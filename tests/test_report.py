from dataclasses import asdict

from vllm_dyno.report import pareto_keys, recommendation, render
from vllm_dyno.search import Goal
from vllm_dyno.space import Candidate


def rec(stage: str, tput: float, kl: float, status: str = "ok", **changes: str) -> dict:
    cand = Candidate(**changes)
    r = {"key": f"{stage}|{cand.label()}", "stage": stage, "label": cand.label(), "status": status, **asdict(cand)}
    if status == "ok":
        r |= {
            "tput_tok_s": tput,
            "tput_cv": 0.01,
            "tpot_ms": 10_000 / tput,
            "ttft_ms": 30.0,
            "kv_tokens": 40_000,
            "kl": kl,
            "top1_agree": 1 - kl,
            "ppl": 11.5 + kl,
            "load_s": 40.0,
        }
    else:
        r["error"] = "no kernel"
    return r


RECORDS = [
    rec("screen", 1000, 0.0),
    rec("screen", 1200, 0.01, weights="fp8"),
    rec("screen", 0, 0, status="skipped", kv_cache_dtype="fp8"),
    rec("confirm", 1010, 0.001),
    rec("confirm", 1190, 0.012, weights="fp8"),
]


def meta(ctx) -> dict:
    return {"created": "2026-10-07 16:00", "context": ctx.to_json(), "goal": asdict(Goal()), "gpu_price": 0.25}


def test_recommendation_is_the_best_confirmed_config_within_the_goal():
    assert (recommendation(RECORDS, Goal()) or {}).get("label") == "weights=fp8"
    assert (recommendation(RECORDS, Goal(max_kl=0.005)) or {}).get("label") == "baseline"


def test_pareto_front_drops_dominated_trials():
    # confirm|weights=fp8 is slower, slower per token and further from the reference than screen|weights=fp8
    assert pareto_keys(RECORDS) == {"screen|baseline", "screen|weights=fp8", "confirm|baseline"}


def test_report_names_the_winner_its_command_and_the_skips(ctx, tmp_path):
    text = render(meta(ctx), RECORDS, tmp_path)
    assert "`weights=fp8`: 1,190 tok/s, 1.18x the baseline" in text
    assert f"dyno serve {tmp_path}" in text
    # no nvcc on the test host: the direct command carries the sampler switch
    assert (
        "VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve org/base --gpu-memory-utilization 0.63 --max-model-len 2048 "
        "--quantization fp8" in text
    )
    assert "`kv_cache_dtype=fp8`, screen: skipped, no kernel" in text


def test_configs_within_noise_of_the_winner_are_named_with_their_quality(ctx, tmp_path):
    # 1190 / 1150 is a 3.5% gap, under the 5% smallest counted gain
    close = rec("confirm", 1150, 0.001, weights="org/base-AWQ")
    text = render(meta(ctx), [*RECORDS, close], tmp_path)
    assert "trail `weights=fp8` by less than the noise" in text
    assert "- `weights=org/base-AWQ`: 3.5% lower throughput, noise 5.0%. KL 0.0010 (winner 0.0120)" in text


def test_a_noisy_config_widens_the_tie(ctx, tmp_path):
    # a 6.2% gap is above the 5% smallest counted gain but under this config's 7% CV
    noisy = rec("confirm", 1120, 0.001, weights="org/base-AWQ") | {"tput_cv": 0.07}
    assert "6.2% lower throughput, noise 7.0%" in render(meta(ctx), [*RECORDS, noisy], tmp_path)
    assert "by less than the noise" not in render(meta(ctx), [*RECORDS, noisy | {"tput_cv": 0.05}], tmp_path)


def test_no_tie_section_when_the_winner_clears_the_noise(ctx, tmp_path):
    # fp8 leads the baseline by 17.8%
    assert "by less than the noise" not in render(meta(ctx), RECORDS, tmp_path)


def test_latency_goal_ties_on_tpot(ctx, tmp_path):
    # TPOT 10000/1150 against the winner's 10000/1190 is 3.5% higher
    close = rec("confirm", 1150, 0.001, weights="org/base-AWQ")
    latency = meta(ctx) | {"goal": asdict(Goal(target="latency"))}
    assert "- `weights=org/base-AWQ`: 3.5% higher TPOT, noise 5.0%." in render(latency, [*RECORDS, close], tmp_path)


def test_report_shows_capacity_under_load(ctx, tmp_path):
    slo = {"ttft_ms": 1000, "tpot_ms": 50, "share": 0.9}
    ok = {
        "label": "baseline",
        "status": "ok",
        "slo": slo,
        "capacity_rps": 11.31,
        "capped": False,
        "chat": {"ttft_p90_ms": 210.0, "tpot_p90_ms": 41.2},
        "batch": {"tput_tok_s": 2500.0, "tpot_p50_ms": 30.5},
        "kv_tokens": 44624,
    }
    failed = {"label": "weights=fp8", "status": "oom", "slo": slo}
    text = render(meta(ctx), RECORDS, tmp_path, [ok, failed])
    assert "capacity is the highest rate at which 90% of requests had TTFT ≤ 1000 ms and TPOT ≤ 50 ms" in text
    assert "| `baseline` | 11.31 | 210 | 41.2 | 2,500 | 30.5 | 44,624 |" in text
    assert "| `weights=fp8` | oom |" in text
