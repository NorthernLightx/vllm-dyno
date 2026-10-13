import json
import logging
import re
import sys
from dataclasses import asdict, replace

import httpx
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
INT8_TRITON = "kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN"


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
    INT8_TRITON: 0.014,
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


def write_run(path, ctx, offline=OFFLINE, harness_sha="abc", **load_json) -> None:
    """A finished `dyno run` with its load.json; a load.json key given as None is left out."""
    path.mkdir()
    meta = {"context": ctx.to_json(), "goal": asdict(Goal()), "harness_sha": harness_sha, "free_mib_at_start": 6000}
    (path / "run.json").write_text(json.dumps(meta))
    settings = {"slo": asdict(Slo()), "probe_s": 30, "load_sha": load.LOAD_SHA, "env": {}, "free_mib_at_start": 6000}
    settings = {k: v for k, v in (settings | load_json).items() if v is not None}
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


def rules_run(tmp_path, monkeypatch, ctx, max_combos: int = 8, offline=OFFLINE, results=SERVED):
    path = tmp_path / "rules"
    write_run(path, ctx, offline)
    record_session(monkeypatch, path, ctx, "s1", lambda s: s.search(Goal(), max_combos, 900), results)
    return path


def write_load(path, records: list[dict]) -> None:
    (path / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def failed(stage: str, label: str) -> dict:
    """An offline trial vLLM refused to run."""
    return trial(stage, label, 0.0, status="unsupported") | {"kl": None}


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
    """A session that loads the baseline, then the combination label if its quality trial ran."""

    def drive(session: LoadSession, seed: int) -> None:
        session.baseline()
        searcher = Searcher(session.run, session.ctx, Goal(), {}, 900)
        r = searcher.stage("combine", [cand(label)], SCREEN)[0]
        if r["status"] == "ok":
            session.measure(r)

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


def test_a_same_engine_alias_no_run_tried_offline_takes_its_twins_trial_and_is_skipped(tmp_path, monkeypatch, ctx):
    path = tmp_path / "singles"
    write_run(path, ctx)
    int8 = next(r for r in OFFLINE if r["label"] == "kv_cache_dtype=int8_per_token_head")
    record_session(monkeypatch, path, ctx, "s1", lambda s: (s.baseline(), s.measure(int8)))
    assert trial_key("combine", cand(INT8_TRITON)) not in {r["key"] for r in Run(path).records}
    rec = replay.recorded([path])
    trials = []

    def drive(session: LoadSession, seed: int) -> None:  # the rules' combine step, for one combination
        session.baseline()
        session.measure(int8)
        r = Searcher(session.run, session.ctx, Goal(), {}, 900).stage("combine", [cand(INT8_TRITON)], SCREEN)[0]
        trials.append(r)
        if r["status"] == "ok" and not session.same_engine(r):
            session.measure(r)

    outcome = replay.repeat(rec, replay.truth(rec, "chat"), drive, 0, 0, "chat")
    assert (outcome["gap"], outcome["loaded"]) == (None, ["baseline", "kv_cache_dtype=int8_per_token_head"])
    stand_in = trials[0]
    assert (stand_in["key"], stand_in["label"], stand_in["attention_backend"]) == (
        f"combine|{INT8_TRITON}",
        INT8_TRITON,
        "TRITON_ATTN",
    )
    assert (stand_in["kl"], stand_in["wall_s"], stand_in["stand_in"]) == (0.013, 60.0, int8["label"])


class Scripted:
    """Picks the configs of a script in turn, then the first config left; raises an exception the script holds."""

    name = "scripted"

    def __init__(self, script: list) -> None:
        self.script = list(script)

    def propose(self, step):
        pick = self.script.pop(0) if self.script else step.options[0]
        if isinstance(pick, Exception):
            raise pick
        return planner.Proposal(cand(pick) if isinstance(pick, str) else pick, "scripted")


def test_a_planner_that_fails_ends_its_repeat_as_a_fallback_the_rules_do_not_finish(tmp_path, monkeypatch, ctx):
    path = rules_run(tmp_path, monkeypatch, ctx)
    script = [FP8_INT8, planner.PlannerError("HTTP 401: no key")]
    monkeypatch.setattr(replay, "proposer", lambda name, seed, llm: Scripted(script))
    search = LoadSession.search
    result = replay.replay([path], "scripted", repeats=2, max_servers=3, max_combos=8, seed=0, target="chat")
    assert LoadSession.search is search
    for o in result["repeats"]:
        assert (o["fallback"], o["loaded"], o["pick"]) == (
            {"step": 2, "error": "HTTP 401: no key"},
            ["baseline", FP8_INT8],
            None,
        )
    s = result["summary"]
    assert (s["completed"], s["gaps"], s["fallbacks"], s["servers"], s["exact"]) == (0, 0, 2, None, None)
    assert s["fallback_errors"] == [{"error": "HTTP 401: no key", "repeats": 2}]
    text = replay.markdown(result)
    assert "| scripted | 3 | 2 | 0 | 2 |  |  |  |  |  |  |" in text
    assert "- HTTP 401: no key: 2" in text


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
    rules = LoadSession.search
    monkeypatch.setattr(replay, "proposer", lambda name, seed, llm: Scripted([RuntimeError("planner crashed")]))
    drive = replay.driver("scripted", Goal(), "chat", max_servers=3, max_combos=8, llm=None)
    with pytest.raises(RuntimeError, match="planner crashed"):
        replay.repeat(rec, best, drive, 2, 0, "chat")
    assert (load.measure_config, load.wait_for_gpu, search.run_trial, LoadSession.search) == (*real, rules)


SLO_100 = asdict(Slo(tpot_ms=100))
ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


@pytest.mark.parametrize(
    ("model", "harness_sha", "load_json", "named"),
    [
        ("org/other", "abc", {}, "model org/base against org/other"),
        ("org/base", "def", {}, "offline harness abc against def"),
        ("org/base", "abc", {"load_sha": "000000000000"}, f"load code {load.LOAD_SHA} against 000000000000"),
        ("org/base", "abc", {"slo": SLO_100}, f"SLO {asdict(Slo())} against {SLO_100}"),
        ("org/base", "abc", {"probe_s": 60}, "probe window 30 against 60"),
        ("org/base", "abc", {"env": ENV}, f"environment {{}} against {ENV}"),
    ],
    ids=["model", "offline harness", "load code", "SLO", "probe window", "environment"],
)
def test_runs_measured_another_way_are_not_pooled(tmp_path, ctx, model, harness_sha, load_json, named):
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    write_run(ours, ctx)
    write_run(theirs, replace(ctx, model=model), harness_sha=harness_sha, **load_json)
    expected = f"{ours} and {theirs} were measured differently ({named})"
    with pytest.raises(SystemExit, match=re.escape(expected)):
        replay.recorded([ours, theirs])


def test_runs_whose_load_json_has_no_environment_pool_only_with_each_other(tmp_path, monkeypatch, ctx):
    old, older, new = tmp_path / "old", tmp_path / "older", tmp_path / "new"
    for path, env in ((old, None), (older, None), (new, {})):
        write_run(path, ctx, env=env)
        record_session(monkeypatch, path, ctx, "s1", lambda s: s.search(Goal(), 8, 900))
    assert "env" not in json.loads((old / "load.json").read_text())
    result = replay.replay([old, older], "rules", repeats=1, max_servers=10, max_combos=8, seed=0, target="chat")
    assert (result["summary"]["completed"], result["truth"]["pick"]) == (1, FP8_INT8)
    expected = f"{old} and {new} were measured differently (environment None against {{}})"
    with pytest.raises(SystemExit, match=re.escape(expected)):
        replay.recorded([old, new])


def test_a_run_whose_load_records_name_other_load_code_than_its_load_json_is_refused(tmp_path, ctx):
    path = tmp_path / "run"
    write_run(path, ctx)
    write_load(path, [served("baseline", 10.0, 1000.0, "FLASH_ATTN") | {"session": "a", "load_sha": "000000000000"}])
    expected = f"{path}/load.jsonl holds records of other load code than its load.json: ['000000000000']"
    with pytest.raises(SystemExit, match=re.escape(expected)):
        replay.recorded([path])


def test_only_ok_load_records_against_a_baseline_that_served_are_drawn(tmp_path, ctx):
    path = tmp_path / "run"
    write_run(path, ctx)
    awq = "weights=org/base-AWQ"  # measured, though no offline trial here allows it
    write_load(
        path,
        [
            served("baseline", 4.0, 400.0, "FLASH_ATTN") | {"session": "a", "status": "degraded"},
            served("baseline", 10.0, 1000.0, "FLASH_ATTN") | {"session": "a"},
            served("weights=fp8", 12.0, 1200.0, "FLASH_ATTN") | {"session": "a"},
            served(awq, 11.0, 1500.0, "FLASH_ATTN") | {"session": "a"},
            served("kv_cache_dtype=int8_per_token_head", 0.0, 0.0, "TRITON_ATTN")
            | {"session": "a", "status": "failed"},
            served("baseline", 0.0, 900.0, "FLASH_ATTN") | {"session": "b", "capacity_rps": None},
            served("weights=fp8", 9.0, 1100.0, "FLASH_ATTN") | {"session": "b"},
        ],
    )
    rec = replay.recorded([path])
    assert [(b["session"], b["capacity_rps"]) for b in rec.baselines] == [("a", 10.0)]
    assert {c.label(): [(r.chat, r.batch) for r in reps] for c, reps in rec.measured.items()} == {
        "weights=fp8": [(pytest.approx(1.2), pytest.approx(1.2))],
        awq: [(pytest.approx(1.1), pytest.approx(1.5))],
    }
    assert [r["label"] for r in replay.truth(rec, "chat").table] == ["baseline", "weights=fp8", awq]


def test_a_session_starts_from_the_trials_dyno_run_wrote_up_to_its_confirm_stage(tmp_path, monkeypatch, ctx):
    offline = [*OFFLINE, trial("confirm", "weights=fp8", 0.009, tput=2100.0)]
    path = rules_run(tmp_path, monkeypatch, ctx, offline=offline)
    assert {r["stage"] for r in Run(path).records[len(offline) :]} == {"combine"}  # the rules session's
    rec = replay.recorded([path])
    started = []

    def drive(session: LoadSession, seed: int) -> None:
        started.append(Run(session.run.path).records)
        session.baseline()

    replay.repeat(rec, replay.truth(rec, "chat"), drive, 0, 0, "chat")
    assert started == [offline]
    result = replay.replay([path], "rules", repeats=1, max_servers=10, max_combos=8, seed=0, target="chat")
    assert {c["config"]: c["kl"] for c in result["truth"]["configs"]}["weights=fp8"] == 0.009


def test_a_combination_that_failed_offline_fails_again_unless_another_run_ran_it(tmp_path, monkeypatch, ctx):
    refused = tmp_path / "refused"
    write_run(refused, ctx, [*OFFLINE, failed("combine", FP8_TRITON)])  # after confirm, as dyno load appends it
    record_session(monkeypatch, refused, ctx, "s1", lambda s: s.search(Goal(), 8, 900))
    assert FP8_TRITON not in [r["label"] for r in load_records(refused)]
    ran = tmp_path / "ran"
    write_run(ran, ctx)
    record_session(monkeypatch, ran, ctx, "s2", lambda s: loading(FP8_TRITON)(s, 0))
    last_trials = []

    def drive(session: LoadSession, seed: int) -> None:
        loading(FP8_TRITON)(session, seed)
        last_trials.append(Run(session.run.path).records[-1])

    both = replay.recorded([refused, ran])
    outcome = replay.repeat(both, replay.truth(both, "chat"), drive, 0, 0, "chat")
    assert (outcome["gap"], outcome["loaded"]) == (None, ["baseline", FP8_TRITON])
    assert (last_trials[-1]["label"], last_trials[-1]["status"]) == (FP8_TRITON, "ok")
    alone = replay.recorded([refused])
    outcome = replay.repeat(alone, replay.truth(alone, "chat"), drive, 0, 0, "chat")
    assert (outcome["gap"], outcome["loaded"]) == (None, ["baseline"])
    assert (last_trials[-1]["label"], last_trials[-1]["status"]) == (FP8_TRITON, "unsupported")


@pytest.fixture
def cli_logging():
    """Restores the logger levels cmd_replay sets for the rest of the process."""
    loggers = [logging.getLogger(name) for name in ("vllm_dyno", "vllm_dyno.replay")]
    levels = [lg.level for lg in loggers]
    yield
    for lg, level in zip(loggers, levels):
        lg.setLevel(level)


def test_a_batch_replay_scores_the_batch_figure(tmp_path, monkeypatch, ctx, capsys, cli_logging):
    path = rules_run(tmp_path, monkeypatch, ctx, results={**SERVED, FP8_TRITON: (12.5, 1600.0, "TRITON_ATTN")})
    main(["replay", str(path), "--repeats", "1", "--goal", "batch"])
    text = capsys.readouterr().out
    assert "| rules |  | 1 | 0 | 0 | 5.0 | 40 | 1.00 | 1.00 | 1.000 | 1.000 |" in text
    assert f"Truth: `{FP8_TRITON}`; tied with it: none." in text
    configs = [line.split(" | ")[0] for line in text.splitlines() if line.startswith("| `")]
    assert configs[0] == f"| `{FP8_TRITON}`" and configs[-1] == "| `attention_backend=TRITON_ATTN`"
    monkeypatch.setattr(replay, "proposer", lambda name, seed, llm: Scripted([FP8_INT8]))
    for target, regret in (("batch", 1.6 / 1.4), ("chat", 1.0)):
        result = replay.replay([path], "scripted", repeats=1, max_servers=1, max_combos=8, seed=0, target=target)
        assert (result["repeats"][0]["pick"], result["summary"]["regret_worst"]) == (FP8_INT8, pytest.approx(regret))


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


def test_optuna_scores_failed_combinations_infeasible_and_a_same_engine_skip_as_its_twin(tmp_path, monkeypatch, ctx):
    # combinations `dyno run` ran before its confirm stage: one over the KL limit, one vLLM refused
    offline = [*OFFLINE[:-1], trial("combine", FP8_TRITON, 0.09), failed("combine", INT8_TRITON), OFFLINE[-1]]
    path = rules_run(tmp_path, monkeypatch, ctx, offline=offline)
    told = []

    class Telling(Scripted):
        def propose(self, step):
            told.append(replay._known(step))
            return super().propose(step)

    monkeypatch.setattr(replay, "proposer", lambda name, seed, llm: Telling([FP8_INT8, FP8_INT8_TRITON]))
    result = replay.replay([path], "scripted", repeats=1, max_servers=2, max_combos=8, seed=0, target="chat")
    assert result["repeats"][0]["plan"][1]["result"].startswith(f"runs the same engine as {FP8_INT8}")
    third = {c.label(): (pytest.approx(score), feasible) for c, (score, feasible) in told[2].items()}
    twins = (replay._score(1.4, 1.4, "chat"), True)
    assert third == {
        "baseline": (0.0, True),
        FP8_INT8: twins,
        FP8_INT8_TRITON: twins,
        FP8_TRITON: (replay.INFEASIBLE, False),
        INT8_TRITON: (replay.INFEASIBLE, False),
    }
    assert replay._offline_estimate(offline)(cand(FP8_INT8)) == pytest.approx(2.0 * 1.4)


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


def test_replay_prints_a_summary_and_writes_every_repeat(tmp_path, monkeypatch, ctx, capsys, cli_logging):
    path = rules_run(tmp_path, monkeypatch, ctx)
    out = tmp_path / "replay.json"
    main(["replay", str(path), "--repeats", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert "| rules |  | 2 | 0 | 0 | 5.0 | 40 | 1.00 | 1.00 | 1.000 | 1.000 |" in text
    assert f"Truth: `{FP8_INT8}`; tied with it: `{FP8_INT8_TRITON}`." in text
    written = json.loads(out.read_text())
    assert (written["planner"], len(written["repeats"]), written["summary"]["gaps"]) == ("rules", 2, 0)


def test_replaying_the_llm_planner_needs_a_key_for_a_remote_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="needs an API key; set OPENROUTER_API_KEY"):
        main(["replay", str(tmp_path), "--planner", "llm"])
