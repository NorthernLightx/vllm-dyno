from dataclasses import asdict

import pytest

from vllm_dyno.report import load_recommendation, pareto_keys, recommendation, render
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
    # WSL without nvcc on the test host: the direct command carries the sampler switch and the allocator setting
    assert (
        "VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.63 vllm serve org/base "
        "--gpu-memory-utilization 0.63 --max-model-len 2048 --quantization fp8" in text
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


def load_rec(label: str, session: str, capacity: float, batch: float, backend: str = "FLASH_ATTN") -> dict:
    cand = Candidate() if label == "baseline" else Candidate(**dict(kv.split("=") for kv in label.split()))
    return {
        "label": label,
        **asdict(cand),
        "status": "ok",
        "slo": {"ttft_ms": 1000, "tpot_ms": 50, "share": 0.9},
        "session": session,
        "backend": backend,
        "capacity_rps": capacity,
        "capped": False,
        "chat": {"ttft_p90_ms": 210.0, "tpot_p90_ms": 41.2},
        "batch": {"tput_tok_s": batch, "tpot_p50_ms": 30.5},
        "kv_tokens": 44624,
    }


def test_report_shows_capacity_under_load_against_the_baseline_of_the_same_session(ctx, tmp_path):
    failed = {"label": "kv_cache_dtype=fp8", "status": "oom", "slo": load_rec("baseline", "", 1, 1)["slo"]}
    load = [
        load_rec("baseline", "night", 10.0, 2000.0),
        load_rec("weights=fp8", "night", 11.31, 2500.0),
        load_rec("baseline", "morning", 12.0, 2400.0),
        load_rec("attention_backend=TRITON_ATTN", "morning", 15.0, 2400.0, backend="TRITON_ATTN"),
        failed | {"session": "morning"},
    ]
    text = render(meta(ctx), RECORDS, tmp_path, load)
    assert "capacity is the highest rate at which 90% of requests had TTFT ≤ 1000 ms and TPOT ≤ 50 ms" in text
    assert "| `weights=fp8` | FLASH_ATTN | 11.31 | 1.13x | 210 | 41.2 | 2,500 | 1.25x | 44,624 | 0.0120 |" in text
    assert "| `attention_backend=TRITON_ATTN` | TRITON_ATTN | 15.00 | 1.25x | 210 | 41.2 | 2,400 | 1.00x |" in text
    assert "| `kv_cache_dtype=fp8` |  | oom |" in text


GPTQ = "org/base-AWQ"  # a checkpoint the ctx fixture knows
# Qwen2.5-1.5B on an RTX 3070: chat req/s and batch tok/s of one session, and offline KL
LOADED = {
    "baseline": (19.03, 2438, 0.0),
    f"weights={GPTQ}": (18.22, 2806, 0.0006),
    f"weights={GPTQ} attention_backend=TRITON_ATTN": (22.63, 2937, 0.0006),
    "weights=fp8 attention_backend=TRITON_ATTN": (22.63, 3035, 0.0075),
    "weights=fp8 kv_cache_dtype=int8_per_token_head": (22.63, 3082, 0.0234),
    "attention_backend=TRITON_ATTN": (21.67, 2399, 0.0006),
    "weights=org/base-GPTQ-3bit attention_backend=TRITON_ATTN": (30.0, 3500, 0.07),  # fast, over the KL limit
}


def loaded_run(ctx) -> tuple[list[dict], list[dict]]:
    load, records = [], []
    for label, (chat, batch, kl) in LOADED.items():
        load.append(load_rec(label, "s", chat, batch))
        cand = load_rec(label, "s", 0, 0)
        changes = {
            k: cand[k] for k in ("weights", "kv_cache_dtype", "attention_backend") if cand[k] != asdict(Candidate())[k]
        }
        records.append(rec("combine" if len(changes) > 1 else "screen", 1000, kl, **changes))
    return load, records


@pytest.mark.parametrize("target", ["chat", "batch"])
def test_load_recommendation_breaks_ties_on_the_other_profile_then_quality(ctx, target):
    load, records = loaded_run(ctx)
    picked = load_recommendation(load, records, Goal(), target)
    assert picked is not None
    best, ties = picked
    assert best["label"] == f"weights={GPTQ} attention_backend=TRITON_ATTN"
    tied = {r["label"] for r in ties}
    assert "weights=fp8 kv_cache_dtype=int8_per_token_head" in tied
    assert "weights=org/base-GPTQ-3bit attention_backend=TRITON_ATTN" not in tied
    assert ("attention_backend=TRITON_ATTN" in tied) == (target == "chat")


def test_report_recommends_the_load_pick_and_keeps_the_offline_one_below(ctx, tmp_path):
    load, records = loaded_run(ctx)
    text = render(meta(ctx), [*RECORDS, *records], tmp_path, load, "chat")
    pick = text[text.index("## Recommendation") : text.index("### Without load")]
    assert pick.startswith(
        f"## Recommendation\n\n`weights={GPTQ} attention_backend=TRITON_ATTN` for chat traffic: "
        "chat 22.63 req/s (1.19x the baseline), batch 2,937 tok/s (1.20x)"
    )
    assert (
        f"vllm serve {GPTQ} --gpu-memory-utilization 0.63 --max-model-len 2048 --attention-backend TRITON_ATTN" in pick
    )
    assert "- `weights=fp8 attention_backend=TRITON_ATTN`: chat 22.63 req/s (1.19x the baseline)" in pick
    assert "`weights=fp8`: 1,190 tok/s" in text[text.index("### Without load") :]


def test_a_degraded_config_is_listed_with_its_error_and_not_recommended(ctx, tmp_path):
    load, records = loaded_run(ctx)
    error = "TPOT p50 at 2 req/s rose from 8.9 to 597.0 ms during the measurement"
    slow = load_rec("kv_cache_dtype=int8_per_token_head", "s", 40.0, 5000.0)  # ahead of every config on both profiles
    load.append(slow | {"status": "degraded", "error": error, "dir": "load/007-kv-cache-dtype-int8-per-token-head"})
    records.append(rec("screen", 1000, 0.013, kv_cache_dtype="int8_per_token_head"))
    text = render(meta(ctx), [*RECORDS, *records], tmp_path, load, "chat")
    assert f"## Recommendation\n\n`weights={GPTQ} attention_backend=TRITON_ATTN` for chat traffic" in text
    assert "| `kv_cache_dtype=int8_per_token_head` |  | degraded |" in text
    listed = f"- `kv_cache_dtype=int8_per_token_head`, load: degraded, {error} (`{load[-1]['dir']}/server.log`)"
    assert listed in text[text.index("## Skipped and failed") :]


@pytest.mark.parametrize(
    ("sampling", "note"),
    [
        (
            {"sampling": "{'temperature': 0.7}"},
            "Requests left sampling to the server, which applied the model's generation_config: `{'temperature': 0.7}`.",
        ),
        ({"sampling": None}, "Requests left sampling to the server, which used vLLM's defaults."),
        ({}, "Requests decoded greedily without penalties."),
    ],
)
def test_the_load_section_names_the_sampling_requests_used(ctx, tmp_path, sampling, note):
    load, records = loaded_run(ctx)
    text = render(meta(ctx), [*RECORDS, *records], tmp_path, [r | sampling for r in load], "chat")
    assert note in text


def test_report_lists_the_planner_picks_with_expected_and_measured_gains(ctx, tmp_path):
    picks = [
        {
            "model": "org/model",
            "config": "weights=fp8",
            "hypothesis": "fp8 | more KV",
            "expected_chat": 1.1,
            "expected_batch": 1.2,
            "result": "chat 11.31 req/s (1.13x), batch 2500 tok/s (1.25x), KL 0.0120",
            "tokens": {"prompt_tokens": 2100, "completion_tokens": 150},
        },
        {
            "model": "org/model",
            "config": None,
            "hypothesis": "nothing left",
            "expected_chat": None,
            "result": "stop",
            "tokens": {"prompt_tokens": 2400, "completion_tokens": 90},
        },
    ]
    text = render(meta(ctx), RECORDS, tmp_path, picks=picks)
    assert "`org/model` picked the configs to load after the baseline" in text
    assert (
        "| 1 | `weights=fp8` | 1.10x, 1.20x | chat 11.31 req/s (1.13x), batch 2500 tok/s (1.25x), KL 0.0120 "
        r"| fp8 \| more KV |"
    ) in text
    assert "| 2 | stop |  | stop | nothing left |" in text
    assert "The endpoint counted 4,500 prompt tokens and 240 completion tokens." in text
