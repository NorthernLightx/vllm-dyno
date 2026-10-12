import json
import math
import random
from dataclasses import asdict

import pytest

from vllm_dyno import fit
from vllm_dyno.fit import FitProposer, posterior, prior_mean
from vllm_dyno.load import MIN_RATE, LoadSession, Slo
from vllm_dyno.planner import Step, remaining, untried
from vllm_dyno.runner import Run
from vllm_dyno.search import Goal
from vllm_dyno.space import Candidate

FP8, INT8, TRITON = "weights=fp8", "kv_cache_dtype=int8_per_token_head", "attention_backend=TRITON_ATTN"


def cand(label: str) -> Candidate:
    return Candidate() if label == "baseline" else Candidate(**dict(kv.split("=") for kv in label.split()))


def screen(label: str, tput: float, kl: float) -> dict:
    c = cand(label)
    return {"key": f"screen|{label}", "stage": "screen", "label": label, "status": "ok", **asdict(c)} | {
        "tput_tok_s": tput,
        "kl": kl,
    }


def served(label: str, chat: float | None, batch: float) -> dict:
    """A load record of session s; chat None if even the lowest rate missed the SLO."""
    c = cand(label)
    return {"label": label, **asdict(c), "status": "ok", "error": None, "session": "s"} | {
        "capacity_rps": chat,
        "batch": {"tput_tok_s": batch},
    }


def step(path, ctx, trials: list[dict], loads: list[dict]) -> Step:
    """The first step of a session over these offline trials and load records, for chat."""
    path.mkdir(parents=True, exist_ok=True)
    run = Run(path)
    for r in trials:
        run.append(r)
    (path / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in loads))
    (path / "plan").mkdir()
    session = LoadSession(run, ctx, [[1]], Slo(), 30, {}, 600, 6000, id="s")
    options = remaining(session, Goal(), [])
    return Step(session, Goal(), "chat", [], options, untried(session, options), 10, 10, path / "plan" / "s-01.json")


def test_the_posterior_of_two_weights_matches_its_closed_form():
    # one record of w1 alone and one of w1 + w2; unit prior and noise make the precision [[3, 1], [1, 2]]
    post = posterior([[1.0, 0.0], [1.0, 1.0]], [1.0, 3.0], [1.0, 0.0], prior_sd=1.0, noise_sd=1.0)
    assert post.mean == pytest.approx([1.4, 0.8])
    assert post.cov[0] == pytest.approx([0.4, -0.2])
    assert post.cov[1] == pytest.approx([-0.2, 0.6])
    assert post.predict([1.0, 1.0]) == pytest.approx((2.2, math.sqrt(0.6)))
    draws = post.draws(random.Random(0), 20_000)
    means = [sum(d[i] for d in draws) / len(draws) for i in range(2)]
    assert means == pytest.approx([1.4, 0.8], abs=0.02)
    cov = sum((d[0] - means[0]) * (d[1] - means[1]) for d in draws) / len(draws)
    assert cov == pytest.approx(-0.2, abs=0.02)


def test_one_load_record_moves_a_weight_from_its_prior_by_the_ratio_of_their_precisions():
    post = posterior([[1.0]], [0.1], [0.3])  # precisions 1 / 0.25² = 16 and 1 / 0.04² = 625
    assert post.mean == pytest.approx([(16 * 0.3 + 625 * 0.1) / 641])
    assert post.cov[0] == pytest.approx([1 / 641])
    assert posterior([], [], [0.3]).mean == pytest.approx([0.3])


def test_a_weight_starts_from_half_the_log_offline_gain_of_its_value():
    trials = [screen("baseline", 1000, 0.0), screen(FP8, 4000, 0.007)]
    assert prior_mean(trials, ("weights", "fp8")) == pytest.approx(0.5 * math.log(4))
    assert prior_mean(trials, ("attention_backend", "TRITON_ATTN")) == 0.0


COVERAGE = [
    screen("baseline", 1000, 0.0),
    screen(FP8, 4000, 0.007),
    screen(INT8, 1100, 0.013),
    screen(TRITON, 1000, 0.0006),
]


def test_while_a_setting_is_untried_it_picks_a_config_that_settles_one_alone(tmp_path, ctx):
    loads = [served("baseline", 10, 1000), served(TRITON, 13, 1000)]
    s = step(tmp_path, ctx, COVERAGE, loads)
    assert s.untried == ["weights", "kv_cache_dtype"]
    p = FitProposer().propose(s)
    assert p.config == cand(f"{FP8} {TRITON}")
    assert p.hypothesis.startswith(f"Settles weights against {TRITON} and wins ")
    model = fit.fit(s.session.run.records, loads, s.options)
    assert p.expected_chat == pytest.approx(model.predicted_gains(cand(f"{FP8} {TRITON}"))["chat"][0])
    # the config predicted to serve chat best changes both untried settings at once, which shows neither
    assert max(s.options, key=lambda c: model.predicted_gains(c)["chat"][0]) == cand(f"{FP8} {INT8} {TRITON}")
    recorded = json.loads(s.step_file.read_text())
    assert recorded["pick"] == f"{FP8} {TRITON}"
    wins = {c["config"]: c["win"] for c in recorded["configs"]}
    assert wins[f"{FP8} {INT8} {TRITON}"] is None and wins[f"{FP8} {TRITON}"] > 0.5


SETTLED = [
    screen("baseline", 1000, 0.0),
    screen(FP8, 1000, 0.007),
    screen(INT8, 1000, 0.013),
    screen(TRITON, 1000, 0.0006),
]
SINGLES = [served("baseline", 10, 1000), served(FP8, 15, 1500), served(INT8, 8, 1100), served(TRITON, 9.5, 1000)]


def test_once_every_setting_is_known_it_stops_when_no_config_left_is_likely_to_win(tmp_path, ctx):
    s = step(tmp_path, ctx, SETTLED, SINGLES)
    assert s.untried == []
    p = FitProposer().propose(s)
    assert p.config is None
    assert p.hypothesis.startswith("No config left wins 10% of 256 draws of the fitted gains; the likeliest, ")
    assert json.loads(s.step_file.read_text())["pick"] is None


def test_it_does_not_stop_while_a_setting_is_untried_however_unlikely_the_configs_that_settle_it(tmp_path, ctx):
    slow = [*SETTLED[:-1], screen(TRITON, 300, 0.0006)]  # offline, TRITON_ATTN runs at a third of the speed
    s = step(tmp_path, ctx, slow, SINGLES[:-1])
    assert s.untried == ["attention_backend"]
    p = FitProposer().propose(s)
    assert p.config is not None and p.config.attention_backend == "TRITON_ATTN"
    assert p.hypothesis.startswith("Settles attention_backend against ")
    wins = {c["config"]: c["win"] for c in json.loads(s.step_file.read_text())["configs"]}
    assert wins[p.config.label()] < fit.STOP_P


def test_a_config_that_missed_the_slo_at_every_rate_drags_down_the_chat_of_its_combinations(tmp_path, ctx):
    loads = [served("baseline", 10, 1000), served(FP8, None, 1500)]
    s = step(tmp_path, ctx, SETTLED, loads)
    model = fit.fit(s.session.run.records, loads, s.options)
    for label in (FP8, f"{FP8} {INT8}", f"{FP8} {TRITON}"):
        assert model.predicted_gains(cand(label))["chat"][0] < 2 * MIN_RATE / 10
    p = FitProposer().propose(s)  # settles the KV cache dtype or the backend without fp8
    assert p.config is not None and p.config.weights == "base"


def test_a_config_over_the_kl_limit_never_wins(tmp_path, ctx):
    trials = [
        screen("baseline", 1000, 0.0),
        screen(FP8, 1000, 0.03),
        screen(INT8, 1000, 0.03),
        screen(TRITON, 1000, 0.0),
    ]
    loads = [served("baseline", 10, 1000), served(FP8, 12, 1200), served(INT8, 12, 1200), served(TRITON, 9, 1000)]
    s = step(tmp_path, ctx, trials, loads)
    p = FitProposer().propose(s)
    recorded = {c["config"]: c for c in json.loads(s.step_file.read_text())["configs"]}
    both = recorded[f"{FP8} {INT8}"]
    assert both["chat"][0] > 1.3 and both["kl"] == pytest.approx(0.06) and both["win"] == 0.0
    assert p.config is None


def test_the_same_run_state_gives_the_same_proposal_and_step_file(tmp_path, ctx):
    s = step(tmp_path, ctx, COVERAGE, [served("baseline", 10, 1000), served(TRITON, 13, 1000)])
    first = FitProposer().propose(s)
    text = s.step_file.read_text()
    assert FitProposer().propose(s) == first
    assert s.step_file.read_text() == text
