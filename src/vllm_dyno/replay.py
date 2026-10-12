"""Replay a planner many times against recorded load measurements instead of GPU servers.

Each repeat is a fresh `dyno load` session in a temporary run directory that holds run.json and the
offline trials of the first given run, with three functions answered from the records of every
given run: a server's measurement (load.measure_config), a combination's offline quality trial
(search.run_trial) and the wait for GPU memory (load.wait_for_gpu). Nothing starts on the GPU.

A server's measurement is one load record of its config drawn at random, with its gains against
the baseline of its own session applied to a baseline record drawn once per repeat, so every ratio
a session sees is a ratio that was measured. A config naming the attention backend its auto-backend
twin's server picked anyway runs the same engine, and shares the twin's records. A config that no
run measured, under load or offline, ends its repeat as a gap: nothing is estimated. A proposer that
fails ends its repeat as a fallback: `dyno load` hands its session to the rules, whose pick is not
the proposer's.

The truth is the recommendation report.load_recommendation makes over every config's mean gains,
together with the configs it counts as tied. Each repeat is scored by the config its own records
recommend: whether that runs the truth's engine (exact) or the engine of a tied config (tied), and
the truth's mean gain on the target profile over the pick's (regret).

Only records measured the same way are pooled: one model, offline harness, load code, SLO, probe
window and environment.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import random
import tempfile
from collections import Counter
from collections.abc import Callable, Generator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING
from unittest import mock

from . import load, planner, search
from .load import LoadSession, Slo, gains, load_records, session_baseline
from .planner import Llm, Proposal
from .report import load_recommendation, markdown_table, quality
from .runner import Run, trial_key
from .search import Goal, candidate_of
from .space import DEFAULTS, KNOBS, Candidate, Context

if TYPE_CHECKING:
    from .planner import Proposer, Step

log = logging.getLogger(__name__)

TRIAL_TIMEOUT_S = 900  # handed to the search; no trial runs
OPTUNA_STARTUP = 3  # TPE samples at random until this many trials finished, the baseline among them
OPTUNA_QUEUED = 3  # configs with the best offline estimate, measured before TPE picks
OPTUNA_ASKS = 100  # answers to configs already measured before a step draws at random instead
OTHER_WEIGHT = 0.1  # weight of the other profile's log gain in the score TPE maximises
SMALLEST_GAIN = 1e-3  # stands in for no capacity at all, whose log is undefined

Drive = Callable[[LoadSession, int], None]  # runs one session; the int seeds its planner


class ReplayGap(Exception):
    """A replayed session asked for a measurement that none of the given runs recorded."""

    def __init__(self, label: str, kind: str):
        super().__init__(f"no {kind} record of {label}")
        self.label, self.kind = label, kind


class HandedOver(Exception):
    """The proposer failed, and plan() handed its session to the rules search."""


def _hand_over(session: LoadSession, goal: Goal, max_combos: int, trial_timeout_s: float) -> None:
    raise HandedOver


@dataclass(frozen=True)
class Replicate:
    record: dict  # an ok load record
    chat: float  # its gains against the baseline of its session
    batch: float


@dataclass(frozen=True)
class Recorded:
    """What the given runs measured: the offline trials a session starts from and the records it draws."""

    dirs: tuple[Path, ...]
    meta: dict  # run.json of the first run
    settings: dict  # its load.json
    snapshot: list[dict]  # the trials `dyno run` wrote in the first run
    offline: dict[str, dict]  # combine trials of every run by key, the first ok one if any
    baselines: list[dict]  # ok baseline load records of every run
    measured: dict[Candidate, list[Replicate]]  # each config's own ok load records

    def engine(self, cand: Candidate) -> Candidate:
        """The config whose engine cand runs: its auto-backend twin if the twin's server picked cand's backend."""
        if cand.attention_backend == "auto":
            return cand
        twin = cand.replace(attention_backend="auto")
        if any(rep.record.get("backend") == cand.attention_backend for rep in self.measured.get(twin, [])):
            return twin
        return cand

    def replicates(self, cand: Candidate) -> list[Replicate]:
        """The load records of cand's engine, whichever config ran it."""
        engine = self.engine(cand)
        return [rep for c, reps in self.measured.items() if self.engine(c) == engine for rep in reps]

    def mean_gains(self, cand: Candidate) -> tuple[float, float] | None:
        """cand's mean chat and batch gain over its engine's records; None if no run measured it."""
        if cand == Candidate():
            return 1.0, 1.0
        reps = self.replicates(cand)
        return (fmean(r.chat for r in reps), fmean(r.batch for r in reps)) if reps else None

    def records(self) -> list[dict]:
        """Every offline trial a recommendation may read: the snapshot, then other runs' combinations."""
        keys = {r["key"] for r in self.snapshot}
        return [*self.snapshot, *(r for key, r in self.offline.items() if key not in keys)]


def offline_run(records: list[dict]) -> list[dict]:
    """The trials `dyno run` wrote: its stages end with confirm, and combine trials after it are `dyno load`'s."""
    last = max((i for i, r in enumerate(records) if r["stage"] == "confirm"), default=None)
    return records[: last + 1] if last is not None else [r for r in records if r["stage"] == "screen"]


def _measured_by(meta: dict, settings: dict) -> dict:
    """What has to match for two runs' load records to be comparable."""
    return {
        "model": meta["context"]["model"],
        "offline harness": meta["harness_sha"],
        "load code": settings["load_sha"],
        "SLO": settings["slo"],
        "probe window": settings["probe_s"],
        "environment": settings.get("env"),  # None: load.json from before dyno load recorded it
    }


def recorded(dirs: Sequence[Path]) -> Recorded:
    """The trials and load records of runs measured the same way; the first run's offline trials start each session."""
    metas, settings = [], []
    for d in dirs:
        if not (d / "run.json").exists() or not (d / "load.json").exists():
            raise SystemExit(f"{d} holds no run measured under load")
        metas.append(Run(d).meta)
        settings.append(json.loads((d / "load.json").read_text()))
    first = _measured_by(metas[0], settings[0])
    for d, m, s in zip(dirs, metas, settings):
        theirs = _measured_by(m, s)
        if differ := [k for k in first if theirs[k] != first[k]]:
            values = "; ".join(f"{k} {first[k]} against {theirs[k]}" for k in differ)
            raise SystemExit(
                f"{dirs[0]} and {d} were measured differently ({values}); only records measured the same way are pooled"
            )
    offline: dict[str, dict] = {}
    baselines: list[dict] = []
    measured: dict[Candidate, list[Replicate]] = {}
    for d, s in zip(dirs, settings):
        for r in Run(d).records:
            if r["stage"] == "combine" and (r["key"] not in offline or offline[r["key"]]["status"] != "ok"):
                offline[r["key"]] = r
        load_ = load_records(d)
        if other := list(dict.fromkeys(r.get("load_sha") for r in load_ if r.get("load_sha") != s["load_sha"])):
            raise SystemExit(f"{d}/load.jsonl holds records of other load code than its load.json: {other}")
        for r in load_:
            if r["status"] != "ok":
                continue
            if r["label"] == "baseline":
                if r["capacity_rps"]:
                    baselines.append(r)
                continue
            chat, batch = gains(r, session_baseline(load_, r))
            if chat is not None and batch is not None:
                measured.setdefault(candidate_of(r), []).append(Replicate(r, chat, batch))
    if not baselines:
        raise SystemExit(f"no baseline in {', '.join(map(str, dirs))} served at the lowest rate")
    snapshot = offline_run(Run(dirs[0]).records)
    return Recorded(tuple(dirs), metas[0], settings[0], snapshot, offline, baselines, measured)


@dataclass(frozen=True)
class Truth:
    """The recommendation over every config's mean gains, and the configs tied with it."""

    pick: Candidate | None
    tied: list[Candidate]  # the pick among them
    table: list[dict]  # one synthetic load record per config, against one synthetic baseline


def _synthetic(cand: Candidate, capacity: float, batch: float) -> dict:
    return {
        "label": cand.label(),
        **asdict(cand),
        "status": "ok",
        "session": "truth",
        "capacity_rps": capacity,
        "batch": {"tput_tok_s": batch},
    }


def truth(rec: Recorded, target: str) -> Truth:
    """The truth for target ("chat" or "batch"): every allowed or measured config whose engine has load records."""
    goal = Goal(**rec.meta["goal"])
    ctx = Context.from_json(rec.meta["context"])
    universe = [Candidate(), *planner.allowed(ctx, rec.snapshot, goal.max_kl)]
    universe += [c for c in rec.measured if c not in universe]
    capacity = fmean(b["capacity_rps"] for b in rec.baselines)
    tput = fmean(b["batch"]["tput_tok_s"] for b in rec.baselines)
    table = []
    for cand in universe:
        if found := rec.mean_gains(cand):
            table.append(_synthetic(cand, found[0] * capacity, found[1] * tput))
    best = load_recommendation(table, rec.records(), goal, target)
    if best is None:
        return Truth(None, [], table)
    pick = candidate_of(best[0])
    return Truth(pick, [pick, *(candidate_of(r) for r in best[1])], table)


def _tokens(picks: list[dict]) -> dict[str, int] | None:
    """The tokens an endpoint counted over plan.jsonl entries; None if no entry came from one."""
    counted = [p["tokens"] for p in picks if p.get("tokens")]
    return {k: sum(t.get(k, 0) for t in counted) for k in planner.TOKENS} if counted else None


def _measure_config(rec: Recorded, rng: random.Random, base: dict) -> Callable[..., dict]:
    def measure_config(
        r: dict,
        ctx: Context,
        pool: list[list[int]],
        slo: Slo,
        probe_s: float,
        *,
        env: dict[str, str],
        out_dir: Path,
        timeout_s: float,
    ) -> dict:
        cand = candidate_of(r)
        if cand == Candidate():
            return copy.deepcopy(base)
        reps = rec.replicates(cand)
        if not reps:
            raise ReplayGap(r["label"], "load")
        rep = rng.choice(reps)
        out = copy.deepcopy(rep.record) | {
            "label": r["label"],
            **asdict(cand),
            "capacity_rps": rep.chat * base["capacity_rps"],
        }
        out["batch"]["tput_tok_s"] = rep.batch * base["batch"]["tput_tok_s"]
        return out

    return measure_config


def _run_trial(rec: Recorded) -> Callable[..., dict]:
    def run_trial(
        run: Run,
        ctx: Context,
        stage: str,
        cand: Candidate,
        tier: dict,
        *,
        write_reference: bool,
        env: dict[str, str],
        timeout_s: float,
    ) -> dict:
        key = trial_key(stage, cand)
        if found := run.find(key):
            return found
        if key not in rec.offline:
            raise ReplayGap(cand.label(), "offline")
        record = dict(rec.offline[key])
        run.append(record)
        return record

    return run_trial


@contextlib.contextmanager
def replaying(rec: Recorded, rng: random.Random, base: dict) -> Generator[None]:
    """Servers, offline trials and the wait for GPU memory answered from rec, until the block ends.

    base is the baseline record of the session; servers of other configs keep their recorded gains against it.
    """
    with (
        mock.patch.object(load, "measure_config", _measure_config(rec, rng, base)),
        mock.patch.object(load, "wait_for_gpu", lambda free_mib: None),
        mock.patch.object(search, "run_trial", _run_trial(rec)),
    ):
        yield


def repeat(rec: Recorded, best: Truth, drive: Drive, i: int, seed: int, target: str) -> dict:
    """Repeat i: one session in a fresh run directory, driven by drive, and how its recommendation scores."""
    rng = random.Random(f"{seed} {i}")  # a string seed keeps the streams of nearby seeds apart
    base = rng.choice(rec.baselines)
    gap = handed = None
    with tempfile.TemporaryDirectory(prefix="dyno-replay-") as tmp:
        path = Path(tmp)
        (path / "run.json").write_text(json.dumps(rec.meta))
        (path / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec.snapshot))
        run = Run(path)
        session = LoadSession(
            run,
            Context.from_json(rec.meta["context"]),
            [[1]],
            Slo(**rec.settings["slo"]),
            rec.settings["probe_s"],
            {},
            600,
            rec.settings["free_mib_at_start"],
            id=f"replay-{i:04d}",
        )
        try:
            with replaying(rec, rng, base):
                drive(session, rng.randrange(2**31))
        except ReplayGap as e:
            gap = {"config": e.label, "kind": e.kind}
        except HandedOver as e:
            handed = e
        served = load_records(path)
        records = list(run.records)
        picks = planner.plan_records(path)
    # plan() hands over from inside the except clause that caught the proposer's error
    fallback = {"step": len(picks) + 1, "error": str(handed.__context__)} if handed else None
    outcome = {
        "repeat": i,
        "loaded": [r["label"] for r in served],
        "servers": sum(r["label"] != "baseline" for r in served),
        "gpu_min": sum(r.get("wall_s") or 0.0 for r in served) / 60,
        "tokens": _tokens(picks),
        "plan": picks,
        "gap": gap,
        "fallback": fallback,
    }
    found = None if gap or fallback else load_recommendation(served, records, Goal(**rec.meta["goal"]), target)
    if found is None:
        return outcome | {"pick": None, "exact": False, "tied": False, "chat": None, "batch": None, "regret": None}
    pick = candidate_of(found[0])
    chat, batch = rec.mean_gains(pick) or (0.0, 0.0)
    regret = None
    if best.pick is not None:
        want = (rec.mean_gains(best.pick) or (0.0, 0.0))[0 if target == "chat" else 1]
        got = chat if target == "chat" else batch
        regret = want / got if got else math.inf
    return outcome | {
        "pick": pick.label(),
        "exact": best.pick is not None and rec.engine(pick) == rec.engine(best.pick),
        "tied": rec.engine(pick) in [rec.engine(c) for c in best.tied],
        "chat": chat,
        "batch": batch,
        "regret": regret,
    }


class RandomProposer:
    """Picks uniformly among the configs left and never stops: brute force on the same budget."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def propose(self, step: Step) -> Proposal:
        """A config drawn from step.options."""
        return Proposal(self.rng.choice(step.options), "drawn at random from the configs left")


def _candidate(label: str) -> Candidate:
    return Candidate(**dict(kv.split("=", 1) for kv in label.split())) if label != "baseline" else Candidate()


def _score(chat: float, batch: float, target: str) -> float:
    first, other = (chat, batch) if target == "chat" else (batch, chat)
    return math.log(max(first, SMALLEST_GAIN)) + OTHER_WEIGHT * math.log(max(other, SMALLEST_GAIN))


INFEASIBLE = _score(SMALLEST_GAIN, SMALLEST_GAIN, "chat")


class OptunaProposer:
    """Optuna's TPE sampler over the four settings, asked for one config per step and never stopping.

    The study is rebuilt at every step from the run's load records and this session's picks, so the
    proposer keeps no state. TPE maximises one score, the log gain on the target profile plus
    OTHER_WEIGHT times the log gain on the other; the session's recommendation still comes from its
    records. A config that failed offline, went over the KL limit or whose server failed is
    infeasible, and a config skipped as the same engine as a loaded one scores as that one.
    The OPTUNA_QUEUED configs with the highest offline estimate, the product of their settings'
    offline throughput gains, are measured first. TPE proposes configs it has seen; each is told its
    score again, and after OPTUNA_ASKS of those the step draws a config at random.
    """

    name = "optuna"

    def __init__(self, seed: int = 0) -> None:
        try:
            import optuna
        except ImportError:
            raise SystemExit(
                "the optuna planner needs Optuna; install the bench extra: uv sync --extra bench"
            ) from None
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.seed = seed

    def propose(self, step: Step) -> Proposal:
        """The next config TPE asks for among step.options."""
        import optuna
        from optuna.distributions import BaseDistribution, CategoricalDistribution
        from optuna.trial import create_trial

        known = _known(step)
        cands = [*step.options, *(c for c in known if c not in step.options)]
        dists: dict[str, BaseDistribution] = {
            k: CategoricalDistribution(sorted({DEFAULTS[k], *(getattr(c, k) for c in cands)})) for k in KNOBS
        }
        sampler = optuna.samplers.TPESampler(seed=self.seed + len(step.journal), n_startup_trials=OPTUNA_STARTUP)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        for cand, (score, feasible) in known.items():
            constraint = {"runs": 0.0 if feasible else 1.0}
            study.add_trial(create_trial(params=asdict(cand), distributions=dists, value=score, constraints=constraint))
        estimate = _offline_estimate(step.session.run.records)
        queued = sorted((c for c in cands if c.changes()), key=lambda c: (-estimate(c), c.label()))[:OPTUNA_QUEUED]
        for cand in queued:
            if cand in step.options:
                study.enqueue_trial(asdict(cand))
        for _ in range(OPTUNA_ASKS):
            trial = study.ask(dists)
            cand = Candidate(**trial.params)
            if cand in step.options:
                why = "among the best offline" if cand in queued else "TPE's pick"
                return Proposal(cand, why)
            score, feasible = known.get(cand, (INFEASIBLE, False))
            trial.set_constraint("runs", 0.0 if feasible else 1.0)
            study.tell(trial, score)
        rng = random.Random(self.seed + len(step.journal))
        return Proposal(
            rng.choice(step.options), f"TPE asked for measured configs {OPTUNA_ASKS} times; drawn at random"
        )


def _known(step: Step) -> dict[Candidate, tuple[float, bool]]:
    """Score and feasibility of every config measured or tried so far."""
    known: dict[Candidate, tuple[float, bool]] = {}
    load_ = load_records(step.session.run.path)
    for r in load_:
        chat, batch = gains(r, session_baseline(load_, r))
        if chat is not None and batch is not None:
            known[candidate_of(r)] = (_score(chat, batch, step.target), True)
        else:
            known.setdefault(candidate_of(r), (INFEASIBLE, False))
    trials = [r for r in step.session.run.records if r["stage"] in ("screen", "combine")]
    for r in trials:  # a single that failed offline added no value to the search space
        if r["stage"] == "combine" and (r["status"] != "ok" or r["kl"] > step.goal.max_kl):
            known.setdefault(candidate_of(r), (INFEASIBLE, False))
    for j in step.journal:
        cand = _candidate(j["config"])
        if cand in known:
            continue
        r = next((r for r in trials if r["label"] == j["config"] and r["status"] == "ok"), None)
        twin = step.session.same_engine(r) if r else None
        known[cand] = known.get(_candidate(twin), (INFEASIBLE, False)) if twin else (INFEASIBLE, False)
    return known


def _offline_estimate(records: list[dict]) -> Callable[[Candidate], float]:
    """A config's offline throughput over the baseline's, as the product of its settings' gains on their own."""
    base = next(r for r in records if r["stage"] == "screen" and r["label"] == "baseline")
    single = {
        next(iter(candidate_of(r).changes().items())): r["tput_tok_s"] / base["tput_tok_s"]
        for r in records
        if r["stage"] == "screen" and r["status"] == "ok" and len(candidate_of(r).changes()) == 1
    }
    return lambda cand: math.prod(single.get(item, 1.0) for item in cand.changes().items())


def proposer(name: str, seed: int, llm: Llm | None) -> Proposer:
    """A fresh proposer of the named planner, other than rules."""
    if name == "llm":
        if llm is None:
            raise ValueError("the llm planner needs an endpoint")
        return planner.LlmProposer(llm)
    if name == "fit":
        from .fit import FitProposer

        return FitProposer(seed)
    if name == "all":
        return planner.AllProposer()
    if name == "random":
        return RandomProposer(seed)
    if name == "optuna":
        return OptunaProposer(seed)
    raise ValueError(f"unknown planner {name!r}")


def driver(name: str, goal: Goal, target: str, *, max_servers: int, max_combos: int, llm: Llm | None) -> Drive:
    """How a session runs with the named planner: the rules search, or plan() with a fresh proposer.

    Under plan(), the rules search raises HandedOver instead of finishing a session the proposer failed.
    """

    def drive(session: LoadSession, seed: int) -> None:
        if name == "rules":
            session.search(goal, max_combos, TRIAL_TIMEOUT_S)
            return
        with mock.patch.object(LoadSession, "search", _hand_over):
            planner.plan(
                session,
                proposer(name, seed, llm),
                goal,
                target,
                max_servers=max_servers,
                max_combos=max_combos,
                trial_timeout_s=TRIAL_TIMEOUT_S,
            )

    return drive


def summarize(outcomes: list[dict]) -> dict:
    """Means over the repeats that ended without a gap or a fallback; the gaps' configs and the proposers' errors."""
    done = [o for o in outcomes if o["gap"] is None and o["fallback"] is None]
    regrets = [o["regret"] for o in done if o["regret"] is not None]
    gaps = Counter((o["gap"]["config"], o["gap"]["kind"]) for o in outcomes if o["gap"])
    errors = Counter(o["fallback"]["error"] for o in outcomes if o["fallback"])
    return {
        "repeats": len(outcomes),
        "completed": len(done),
        "servers": fmean(o["servers"] for o in done) if done else None,
        "gpu_min": fmean(o["gpu_min"] for o in done) if done else None,
        "exact": fmean(o["exact"] for o in done) if done else None,
        "tied": fmean(o["tied"] for o in done) if done else None,
        "regret_mean": fmean(regrets) if regrets else None,
        "regret_worst": max(regrets) if regrets else None,
        "gaps": sum(gaps.values()),
        "fallbacks": sum(errors.values()),
        "tokens": _tokens([o for o in outcomes if o["tokens"]]),
        "gap_configs": [
            {"config": config, "kind": kind, "repeats": n}
            for (config, kind), n in sorted(gaps.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "fallback_errors": [
            {"error": error, "repeats": n} for error, n in sorted(errors.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }


def replay(
    dirs: Sequence[Path],
    name: str,
    *,
    repeats: int,
    max_servers: int,
    max_combos: int,
    seed: int,
    target: str,
    llm: Llm | None = None,
) -> dict:
    """Replay the named planner repeats times over the runs' records; the truth, every repeat and their summary."""
    rec = recorded(dirs)
    best = truth(rec, target)
    goal = Goal(**rec.meta["goal"])
    drive = driver(name, goal, target, max_servers=max_servers, max_combos=max_combos, llm=llm)
    outcomes = []
    for i in range(repeats):
        outcome = repeat(rec, best, drive, i, seed, target)
        if outcome["gap"]:
            ended = f"gap: {outcome['gap']['config']}"
        elif outcome["fallback"]:
            ended = f"{name} failed at step {outcome['fallback']['step']}: {outcome['fallback']['error']}"
        else:
            ended = f"{outcome['servers']} servers, {outcome['pick']}"
        log.info("replay   %d/%d  %s", i + 1, repeats, ended)
        outcomes.append(outcome)
    records = rec.records()
    configs = []
    for r in best.table:
        cand = candidate_of(r)
        q = quality(records, r["label"])
        chat, batch = rec.mean_gains(cand) or (0.0, 0.0)
        configs.append(
            {
                "config": r["label"],
                "engine": rec.engine(cand).label(),
                "load_records": len(rec.measured.get(cand, [])) if cand.changes() else len(rec.baselines),
                "engine_records": len(rec.replicates(cand)) if cand.changes() else len(rec.baselines),
                "chat": chat,
                "batch": batch,
                "kl": q["kl"] if q else None,
            }
        )
    return {
        "runs": [str(d) for d in dirs],
        "model": rec.meta["context"]["model"],
        "load_sha": rec.settings["load_sha"],
        "planner": name,
        "goal": target,
        "seed": seed,
        "max_servers": None if name == "rules" else max_servers,
        "max_combos": max_combos,
        "truth": {
            "pick": best.pick.label() if best.pick else None,
            "tied": [c.label() for c in best.tied],
            "configs": configs,
        },
        "summary": summarize(outcomes),
        "repeats": outcomes,
    }


def _num(value: float | None, spec: str) -> str:
    return "" if value is None else format(value, spec)


def markdown(result: dict) -> str:
    """The summary of a replay as markdown: one row for the planner, then the truth it is scored against."""
    s, t = result["summary"], result["truth"]
    budget = "" if result["max_servers"] is None else str(result["max_servers"])
    lines = [
        (
            f"`{result['planner']}` over {len(result['runs'])} runs of `{result['model']}` (load code "
            f"{result['load_sha']}), goal {result['goal']}, seed {result['seed']}. Means are over the repeats "
            "that ended without a gap or a fallback to the rules; regret is the truth's mean gain on the goal "
            "over the pick's."
        ),
        "",
    ]
    header = ["planner", "max servers", "repeats", "gaps", "fallbacks", "servers", "GPU min", "exact", "tied"]
    row = [
        result["planner"],
        budget,
        str(s["repeats"]),
        str(s["gaps"]),
        str(s["fallbacks"]),
        _num(s["servers"], ".1f"),
        _num(s["gpu_min"], ".0f"),
        _num(s["exact"], ".2f"),
        _num(s["tied"], ".2f"),
        _num(s["regret_mean"], ".3f"),
        _num(s["regret_worst"], ".3f"),
    ]
    lines += markdown_table([*header, "regret mean", "regret worst"], [row])
    if s["tokens"]:
        prompt, completion = (s["tokens"][k] for k in planner.TOKENS)
        lines += ["", f"The endpoint counted {prompt:,} prompt and {completion:,} completion tokens in all repeats."]
    if s["gap_configs"]:
        lines += ["", "Repeats that asked for a config no run recorded:", ""]
        lines += [f"- `{g['config']}`, no {g['kind']} record: {g['repeats']}" for g in s["gap_configs"]]
    if s["fallback_errors"]:
        lines += ["", "Repeats whose planner failed, which `dyno load` would hand to the rules:", ""]
        lines += [f"- {f['error']}: {f['repeats']}" for f in s["fallback_errors"]]
    lines += [
        "",
        f"Truth: `{t['pick']}`; tied with it: {', '.join(f'`{c}`' for c in t['tied'][1:]) or 'none'}.",
        "",
    ]
    rows = [
        [
            f"`{c['config']}`",
            "" if c["engine"] == c["config"] else f"`{c['engine']}`",
            str(c["load_records"]),
            str(c["engine_records"]),
            f"{c['chat']:.2f}x",
            f"{c['batch']:.2f}x",
            _num(c["kl"], ".4f"),
        ]
        for c in sorted(t["configs"], key=lambda c: -c["chat"] if result["goal"] == "chat" else -c["batch"])
    ]
    lines += markdown_table(["config", "same engine as", "load records", "engine records", "chat", "batch", "KL"], rows)
    return "\n".join(lines) + "\n"
