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
