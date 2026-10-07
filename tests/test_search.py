from vllm_dyno.search import Goal, combinations, finalists, promising
from vllm_dyno.space import Candidate


def rec(stage: str, tput: float, kl: float = 0.0, status: str = "ok", **changes: str) -> dict:
    cand = Candidate(**changes)
    return {
        "key": f"{stage}|{cand.label()}",
        "stage": stage,
        "label": cand.label(),
        "status": status,
        **{k: getattr(cand, k) for k in ("weights", "kv_cache_dtype", "attention_backend", "speculative")},
        "tput_tok_s": tput,
        "tpot_ms": 10_000 / tput,
        "ttft_ms": 30.0,
        "kl": kl,
        "top1_agree": 1 - kl,
    }


SCREEN = [
    rec("screen", 1000),
    rec("screen", 1200, kl=0.01, weights="fp8"),
    rec("screen", 1300, kl=0.2, weights="org/m-AWQ"),
    rec("screen", 1010, kv_cache_dtype="int8_per_token_head"),
    rec("screen", 1100, attention_backend="TRITON_ATTN"),
    rec("screen", 600, speculative="ngram"),
]


def test_goal_rejects_quality_and_latency_violations():
    goal = Goal(max_kl=0.05, max_tpot_ms=9.5)
    assert goal.violations(SCREEN[2]) == ["KL 0.2000 > 0.05"]
    assert not goal.passes(SCREEN[0])  # 10 ms per token
    assert goal.passes(SCREEN[1])


def test_promising_keeps_single_settings_that_beat_the_baseline_within_the_goal():
    good = promising(SCREEN, Goal())
    assert good["weights"] == {"fp8": 1.2}
    assert good["attention_backend"] == {"TRITON_ATTN": 1.1}
    assert good["kv_cache_dtype"] == {} and good["speculative"] == {}


def test_latency_goal_ranks_by_time_per_token():
    good = promising(SCREEN, Goal(target="latency"))
    assert good["weights"] == {"fp8": 1.2}


def test_combinations_pair_settings_from_different_knobs():
    combos = combinations(promising(SCREEN, Goal()), limit=6)
    assert combos == [Candidate(weights="fp8", attention_backend="TRITON_ATTN")]


def test_finalists_are_the_best_passing_non_baseline_configs():
    records = [*SCREEN, rec("combine", 1250, kl=0.01, weights="fp8", attention_backend="TRITON_ATTN")]
    assert finalists(records, Goal(), top=2) == [
        Candidate(weights="fp8", attention_backend="TRITON_ATTN"),
        Candidate(weights="fp8"),
    ]
