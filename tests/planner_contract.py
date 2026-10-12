"""A stand-in for the proposer interface of planner.py, for as long as planner.py lacks it.

plan() is planner.plan with proposer.propose(step) in place of the exchange with an endpoint.
install() adds the names to vllm_dyno.planner and does nothing once planner.py has its own.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vllm_dyno import planner
from vllm_dyno.load import LoadSession, load_records
from vllm_dyno.search import Goal
from vllm_dyno.space import Candidate


@dataclass(frozen=True)
class Proposal:
    config: Candidate | None
    hypothesis: str
    expected_chat: float | None = None
    expected_batch: float | None = None
    tokens: dict[str, int] | None = None


@dataclass(frozen=True)
class Step:
    session: LoadSession
    goal: Goal
    target: str
    journal: list[dict]
    options: list[Candidate]
    untried: list[str]
    servers_left: int
    max_servers: int
    step_file: Path


class AllProposer:
    name = "all"

    def propose(self, step: Step) -> Proposal:
        return Proposal(step.options[0], "next in the allowed list")


class LlmProposer:
    def __init__(self, llm: planner.Llm) -> None:
        self.llm, self.name = llm, llm.model

    def propose(self, step: Step) -> Proposal:
        brief = planner.brief(
            step.session,
            step.goal,
            step.target,
            step.journal,
            step.options,
            step.untried,
            step.servers_left,
            step.max_servers,
        )
        messages = [{"role": "system", "content": planner.SYSTEM}, {"role": "user", "content": brief}]
        replies: list[planner.Reply] = []
        p = planner.propose(self.llm, messages, step.options, step.untried, replies)
        tokens = {k: sum(int(r.usage.get(k) or 0) for r in replies) for k in planner.TOKENS}
        return Proposal(p.config, p.hypothesis, p.expected_chat, p.expected_batch, tokens)


def plan(session, proposer, goal, target, *, max_servers, max_combos, trial_timeout_s) -> None:
    session.baseline()
    searcher = planner.Searcher(session.run, session.ctx, goal, session.env, trial_timeout_s)
    steps_dir = session.run.path / "plan"
    steps_dir.mkdir(exist_ok=True)
    journal: list[dict] = []
    servers = 0
    while servers < max_servers and len(journal) < 2 * max_servers:
        options = planner.remaining(session, goal, journal)
        if not options:
            return
        left = planner.untried(session, options)
        step_file = steps_dir / f"{session.id}-{len(journal) + 1:02d}.json"
        step = Step(session, goal, target, list(journal), options, left, max_servers - servers, max_servers, step_file)
        try:
            p = proposer.propose(step)
        except planner.PlannerError:
            session.search(goal, max_combos, trial_timeout_s)
            return
        tokens = getattr(p, "tokens", None)
        if p.config is None:
            _append(session, proposer, {"config": None, "result": "stop", "server": False, "tokens": tokens})
            return
        result, started = planner._measure(session, searcher, p.config, goal)
        servers = sum(
            r.get("session") == session.id and r["label"] != "baseline" for r in load_records(session.run.path)
        )
        entry = {
            "config": p.config.label(),
            "hypothesis": p.hypothesis,
            "expected_chat": p.expected_chat,
            "expected_batch": p.expected_batch,
            "result": result,
            "server": started,
            "tokens": tokens,
        }
        journal.append(entry)
        _append(session, proposer, entry)


def _append(session, proposer, entry: dict) -> None:
    with (session.run.path / "plan.jsonl").open("a") as f:
        f.write(json.dumps({"session": session.id, "model": proposer.name, **entry}) + "\n")


def install(put: Callable[[object, str, object], None] = setattr) -> None:
    """Add the stand-in to vllm_dyno.planner with put, unless planner.py defines Step already."""
    if hasattr(planner, "Step"):
        return
    for name, value in {"Step": Step, "AllProposer": AllProposer, "LlmProposer": LlmProposer, "plan": plan}.items():
        put(planner, name, value)
