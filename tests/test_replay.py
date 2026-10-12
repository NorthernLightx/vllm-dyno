import json
import re
import sys
from dataclasses import asdict

import httpx
import planner_contract
import pytest

from vllm_dyno import load, planner, replay, search
from vllm_dyno.cli import main
from vllm_dyno.load import LoadSession, Slo, load_records
from vllm_dyno.planner import Llm
from vllm_dyno.report import load_recommendation
from vllm_dyno.runner import Run, trial_key
from vllm_dyno.search import SCREEN, Goal, Searcher
from vllm_dyno.space import Candidate

FP8_INT8 = "weights=fp8 kv_cache_dtype=int8_per_token_head"
FP8_INT8_TRITON = "weights=fp8 kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN"
FP8_TRITON = "weights=fp8 attention_backend=TRITON_ATTN"


@pytest.fixture(autouse=True)
def proposer_interface(monkeypatch):
    planner_contract.install(lambda obj, name, value: monkeypatch.setattr(obj, name, value, raising=False))


def cand(label: str) -> Candidate:
    return Candidate() if label == "baseline" else Candidate(**dict(kv.split("=", 1) for kv in label.split()))


def trial(stage: str, label: str, kl: float, tput: float = 1000.0, status: str = "ok") -> dict:
    return {
        "key": f"{stage}|{label}",
        "stage": stage,
        "label": label,
        "status": status,
        "kl": kl,
        "tput_tok_s": tput,
        "tpot_ms": 10.0,
        "ttft_ms": 50.0,
        "top1_agree": 0.99,
        "kv_tokens": 100_000,
        "wall_s": 60.0,
        **asdict(cand(label)),
    }


OFFLINE = [  # what `dyno run` wrote
    trial("screen", "baseline", 0.0),
    trial("screen", "weights=fp8", 0.007, tput=2000.0),
    trial("screen", "kv_cache_dtype=int8_per_token_head", 0.013, tput=1400.0),
    trial("screen", "attention_backend=TRITON_ATTN", 0.0006),
    trial("confirm", "baseline", 0.0005),
]
COMBINE_KL = {
    "kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN": 0.014,
    FP8_INT8: 0.02,
    FP8_TRITON: 0.0075,
    FP8_INT8_TRITON: 0.021,
}
SERVED = {  # chat req/s, batch tok/s, the backend the server picked
    "baseline": (10.0, 1000.0, "FLASH_ATTN"),
    "weights=fp8": (10.0, 1200.0, "FLASH_ATTN"),
    "kv_cache_dtype=int8_per_token_head": (12.0, 1200.0, "TRITON_ATTN"),
    "attention_backend=TRITON_ATTN": (12.0, 1000.0, "TRITON_ATTN"),
    FP8_INT8: (14.0, 1400.0, "TRITON_ATTN"),
    FP8_TRITON: (12.5, 1250.0, "TRITON_ATTN"),
}


def served(label: str, capacity: float, batch: float, backend: str) -> dict:
    """A load record as measure_config returns it."""
    return {
        "label": label,
        **asdict(cand(label)),
        "slo": asdict(Slo()),
        "load_sha": load.LOAD_SHA,
        "status": "ok",
        "error": None,
        "kv_tokens": 100_000,
        "backend": backend,
        "sampling": None,
        "capacity_rps": capacity,
        "capped": False,
        "chat": {"ttft_p90_ms": 300.0, "tpot_p90_ms": 40.0},
        "batch": {"tput_tok_s": batch, "kv_peak": 0.5, "preemptions": 0.0},
        "wall_s": 400.0,
    }


def write_run(path, ctx, offline=OFFLINE, load_sha=load.LOAD_SHA) -> None:
    path.mkdir()
    meta = {"context": ctx.to_json(), "goal": asdict(Goal()), "harness_sha": "abc", "free_mib_at_start": 6000}
    (path / "run.json").write_text(json.dumps(meta))
    settings = {"slo": asdict(Slo()), "probe_s": 30, "load_sha": load_sha, "env": {}, "free_mib_at_start": 6000}
    (path / "load.json").write_text(json.dumps(settings))
    run = Run(path)
    for r in offline:
        run.append(r)


def record_session(monkeypatch, path, ctx, sid: str, how, results=SERVED) -> None:
    """A dyno load session over the run at path, whose servers serve results; how(session) drives it."""

    def fake_measure(r, *args, **kwargs):
        return served(r["label"], *results[r["label"]])

    def fake_trial(run, ctx, stage, cand, tier, **kwargs):
        if found := run.find(trial_key(stage, cand)):
            return found
        r = trial(stage, cand.label(), COMBINE_KL[cand.label()])
        run.append(r)
        return r

    with monkeypatch.context() as m:
        m.setattr(load, "measure_config", fake_measure)
        m.setattr(load, "wait_for_gpu", lambda free_mib: None)
        m.setattr(search, "run_trial", fake_trial)
        how(LoadSession(Run(path), ctx, [[1]], Slo(), 30, {}, 600, 6000, id=sid))


def rules_run(tmp_path, monkeypatch, ctx, max_combos: int = 8, offline=OFFLINE):
    path = tmp_path / "rules"
    write_run(path, ctx, offline)
    record_session(monkeypatch, path, ctx, "s1", lambda s: s.search(Goal(), max_combos, 900))
    return path


def test_a_rules_replay_loads_what_the_recorded_session_loaded_and_recommends_the_same(tmp_path, monkeypatch, ctx):
    path = rules_run(tmp_path, monkeypatch, ctx)
    recorded = load_records(path)
    best = load_recommendation(recorded, Run(path).records, Goal(), "chat")
    assert best and best[0]["label"] == FP8_INT8
    result = replay.replay([path], "rules", repeats=3, max_servers=10, max_combos=8, seed=0, target="chat")
    labels = [r["label"] for r in recorded]
    assert labels == [
        "baseline",
        "weights=fp8",
        "kv_cache_dtype=int8_per_token_head",
        "attention_backend=TRITON_ATTN",
        FP8_TRITON,
        FP8_INT8,
    ]
    assert [o["loaded"] for o in result["repeats"]] == [labels] * 3
    assert [o["pick"] for o in result["repeats"]] == [FP8_INT8] * 3
    assert (result["truth"]["pick"], result["truth"]["tied"]) == (FP8_INT8, [FP8_INT8, FP8_INT8_TRITON])
    s = result["summary"]
    assert (s["gaps"], s["servers"], s["exact"], s["tied"], s["regret_worst"]) == (0, 5, 1.0, 1.0, 1.0)
    assert s["gpu_min"] == pytest.approx(6 * 400 / 60)


def loading(label: str):
    """A session that loads the baseline and then label, after its quality trial if it combines settings."""

    def drive(session: LoadSession, seed: int) -> None:
        session.baseline()
        searcher = Searcher(session.run, session.ctx, Goal(), {}, 900)
        session.measure(searcher.stage("combine", [cand(label)], SCREEN)[0])

    return drive


def test_a_server_keeps_its_recorded_gains_against_the_baseline_drawn_for_its_session(tmp_path, monkeypatch, ctx):
    night = rules_run(tmp_path, monkeypatch, ctx)
    morning = tmp_path / "morning"
    write_run(morning, ctx)
    slower = {**SERVED, "baseline": (5.0, 500.0, "FLASH_ATTN"), FP8_INT8: (8.0, 800.0, "TRITON_ATTN")}
    record_session(monkeypatch, morning, ctx, "s2", lambda s: loading(FP8_INT8)(s, 0), slower)
    rec = replay.recorded([night, morning])
    assert [r.chat for r in rec.replicates(cand(FP8_INT8))] == [1.4, 1.6]
    best = replay.truth(rec, "chat")
    sessions: list[list[dict]] = []

    def drive(session: LoadSession, seed: int) -> None:
        loading(FP8_INT8)(session, seed)
        sessions.append(load_records(session.run.path))

    seen = set()
    for i in range(40):
        outcome = replay.repeat(rec, best, drive, i, 0, "chat")
        base, measured = sessions[-1]
        assert (measured["label"], measured["backend"], measured["session"]) == (
            FP8_INT8,
            "TRITON_ATTN",
            f"replay-{i:04d}",
        )
        chat = measured["capacity_rps"] / base["capacity_rps"]
        assert measured["batch"]["tput_tok_s"] / base["batch"]["tput_tok_s"] == pytest.approx(chat)
        seen.add((base["capacity_rps"], round(chat, 6)))
        assert outcome["chat"] == pytest.approx(1.5)  # scored by the mean of both records
    assert seen == {(10.0, 1.4), (10.0, 1.6), (5.0, 1.4), (5.0, 1.6)}


def test_a_replay_repeats_itself_for_a_seed(tmp_path, monkeypatch, ctx):
    path = rules_run(tmp_path, monkeypatch, ctx)

    def run(seed: int) -> dict:
        return replay.replay([path], "random", repeats=6, max_servers=3, max_combos=8, seed=seed, target="chat")

    first = run(7)
    assert first == run(7)
    assert [o["loaded"] for o in first["repeats"]] != [o["loaded"] for o in run(8)["repeats"]]
    assert all(o["servers"] == 3 and o["gap"] is None for o in first["repeats"])


@pytest.mark.parametrize(
    ("offline", "kind"),
    [(OFFLINE, "offline"), ([*OFFLINE[:-1], trial("combine", FP8_INT8, 0.02), OFFLINE[-1]], "load")],
)
def test_a_config_no_run_recorded_ends_its_repeat_as_a_gap(tmp_path, monkeypatch, ctx, offline, kind):
    # with two combinations the rules load fp8 with TRITON_ATTN and skip int8 with it, so fp8 with int8 never ran
    path = rules_run(tmp_path, monkeypatch, ctx, max_combos=2, offline=offline)
    assert FP8_INT8 not in [r["label"] for r in load_records(path)]
    result = replay.replay([path], "all", repeats=2, max_servers=10, max_combos=8, seed=0, target="chat")
    s = result["summary"]
    assert (s["completed"], s["gaps"], s["servers"]) == (0, 2, None)
    assert s["gap_configs"] == [{"config": FP8_INT8, "kind": kind, "repeats": 2}]
    assert result["repeats"][0]["loaded"] == [
        "baseline",
        "attention_backend=TRITON_ATTN",
        "kv_cache_dtype=int8_per_token_head",
        "weights=fp8",
        FP8_TRITON,
    ]
    assert f"`{FP8_INT8}`, no {kind} record: 2" in replay.markdown(result)


def test_a_config_that_names_the_backend_its_twin_picked_runs_the_twins_engine(tmp_path, monkeypatch, ctx):
    rec = replay.recorded([rules_run(tmp_path, monkeypatch, ctx)])
    best = replay.truth(rec, "chat")
    assert rec.engine(cand(FP8_INT8_TRITON)) == cand(FP8_INT8)
    assert rec.engine(cand(FP8_TRITON)) == cand(FP8_TRITON)  # fp8 alone served on FLASH_ATTN
    alias = replay.repeat(rec, best, loading(FP8_INT8_TRITON), 0, 0, "chat")
    assert (alias["loaded"], alias["pick"], alias["exact"], alias["tied"]) == (
        ["baseline", FP8_INT8_TRITON],
        FP8_INT8_TRITON,
        True,
        True,
    )
    assert (alias["chat"], alias["regret"]) == (pytest.approx(1.4), pytest.approx(1.0))
    other = replay.repeat(rec, best, loading(FP8_TRITON), 1, 0, "chat")
    assert (other["pick"], other["exact"], other["tied"], other["regret"]) == (
        FP8_TRITON,
        False,
        False,
        pytest.approx(1.4 / 1.25),
    )


def test_servers_and_trials_are_answered_from_the_records_only_while_a_repeat_runs(tmp_path, monkeypatch, ctx):
    rec = replay.recorded([rules_run(tmp_path, monkeypatch, ctx)])
    best = replay.truth(rec, "chat")
    real = (load.measure_config, load.wait_for_gpu, search.run_trial)
    during = []

    def crash(session: LoadSession, seed: int) -> None:
        during.append((load.measure_config, load.wait_for_gpu, search.run_trial))
        session.baseline()
        raise RuntimeError("planner crashed")

    with pytest.raises(RuntimeError, match="planner crashed"):
        replay.repeat(rec, best, crash, 0, 0, "chat")
    assert (load.measure_config, load.wait_for_gpu, search.run_trial) == real
    assert all(fake is not original for fake, original in zip(during[0], real))
    replay.repeat(rec, best, loading(FP8_TRITON), 1, 0, "chat")
    assert (load.measure_config, load.wait_for_gpu, search.run_trial) == real


def test_runs_measured_with_other_load_code_are_not_pooled(tmp_path, ctx):
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    write_run(ours, ctx)
    write_run(theirs, ctx, load_sha="000000000000")
    expected = f"{ours} and {theirs} were measured differently (load code {load.LOAD_SHA} against 000000000000)"
    with pytest.raises(SystemExit, match=re.escape(expected)):
        replay.recorded([ours, theirs])


def test_optuna_loads_the_configs_best_offline_first_then_what_tpe_asks_for(tmp_path, monkeypatch, ctx):
    pytest.importorskip("optuna")
    path = rules_run(tmp_path, monkeypatch, ctx)

    def run() -> dict:
        return replay.replay([path], "optuna", repeats=2, max_servers=4, max_combos=8, seed=0, target="chat")

    result = run()
    assert result == run()
    for o in result["repeats"]:
        # fp8 with int8 KV and TRITON_ATTN is queued second, and skipped as the engine fp8 with int8 KV ran
        assert o["loaded"][:3] == ["baseline", FP8_INT8, "weights=fp8"]
        assert (o["servers"], o["gap"]) == (4, None)
        assert len(set(o["loaded"])) == len(o["loaded"])


def test_the_optuna_planner_names_the_extra_it_needs(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(SystemExit, match="install the bench extra"):
        replay.OptunaProposer(0)


def test_random_loads_any_config_left_until_its_budget_is_spent(tmp_path, monkeypatch, ctx):
    path = rules_run(tmp_path, monkeypatch, ctx)
    result = replay.replay([path], "random", repeats=20, max_servers=3, max_combos=8, seed=0, target="chat")
    for o in result["repeats"]:
        assert o["servers"] == 3 and len(set(o["loaded"])) == len(o["loaded"])
    loaded = {label for o in result["repeats"] for label in o["loaded"][1:]}
    assert loaded == {c.label() for c in planner.allowed(ctx, OFFLINE, 0.05)}


def test_an_llm_replay_asks_the_endpoint_at_every_step_and_counts_its_tokens(tmp_path, monkeypatch, ctx):
    path = rules_run(tmp_path, monkeypatch, ctx)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        pick = {"config": [FP8_TRITON, FP8_INT8][len(sent) % 2], "hypothesis": "h", "expected_chat": 1.3}
        sent.append(request)
        message = {"role": "assistant", "content": json.dumps(pick)}
        usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage})

    llm = Llm("https://example.org/v1", "org/model", transport=httpx.MockTransport(handler))
    result = replay.replay([path], "llm", repeats=2, max_servers=2, max_combos=8, seed=0, target="chat", llm=llm)
    assert [o["loaded"] for o in result["repeats"]] == [["baseline", FP8_TRITON, FP8_INT8]] * 2
    assert [p["model"] for p in result["repeats"][0]["plan"]] == ["org/model"] * 2
    assert (len(sent), result["summary"]["tokens"]) == (4, {"prompt_tokens": 400, "completion_tokens": 40})
    assert "The endpoint counted 400 prompt and 40 completion tokens in all repeats." in replay.markdown(result)


def test_replay_prints_a_summary_and_writes_every_repeat(tmp_path, monkeypatch, ctx, capsys):
    path = rules_run(tmp_path, monkeypatch, ctx)
    out = tmp_path / "replay.json"
    main(["replay", str(path), "--repeats", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert "| rules |  | 2 | 0 | 5.0 | 40 | 1.00 | 1.00 | 1.000 | 1.000 |" in text
    assert f"Truth: `{FP8_INT8}`; tied with it: `{FP8_INT8_TRITON}`." in text
    written = json.loads(out.read_text())
    assert (written["planner"], len(written["repeats"]), written["summary"]["gaps"]) == ("rules", 2, 0)


def test_replaying_the_llm_planner_needs_a_key_for_a_remote_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="needs an API key; set OPENROUTER_API_KEY"):
        main(["replay", str(tmp_path), "--planner", "llm"])
