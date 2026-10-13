"""The loop that loads, one at a time, the configs a proposer picks for `dyno load`.

Each step offers the proposer the configs it may pick: every combination of values that each ran offline
within the quality limit on its own, not loaded yet. It answers with one config and the gains it expects,
or with none to stop. A combination is checked offline for quality first, as the rules do, and only a
config within the limit starts a server, so the budget counts servers.

A stop counts only once the effect under load of every setting the proposer may change is known: offline
speed can misjudge a setting under load, and a proposer that judges by it stops before loading that setting.
An effect is known once two configs that served, or one and the baseline, differ in that setting alone,
or once a config failed to serve where the same config without that setting served; a config that
changes several untried settings at once shows none of their effects.

LlmProposer asks a chat model through any OpenAI-compatible endpoint. It shows the model the machine,
the offline trials, the load records so far and its earlier picks with what they measured. AllProposer
loads every allowed config, which measures the whole space.

If a proposer fails, such as an endpoint that fails or answers twice with something outside the list,
the rules finish the session, reusing every config already loaded.

<run>/plan.jsonl     one line per pick: hypothesis, expected and measured gains, tokens
<run>/plan/*.json    what a proposer recorded at each step; for an LLM, the messages and the replies
                     with their token usage
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections import Counter
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Protocol

import httpx

from . import load
from .load import (
    BATCH_REQUESTS,
    LOAD_MIN_GAIN,
    OUTPUT_LEN,
    PROMPT_LEN,
    LoadSession,
    gains,
    load_records,
    session_baseline,
)
from .probe import GIB, kv_bytes_per_token
from .search import SCREEN, Goal, Searcher, candidate_of
from .space import DEFAULTS, KNOBS, NGRAM, Candidate, Context, infeasible_reason

log = logging.getLogger(__name__)

RETRIES = 4
MAX_WAIT_S = 120
TOKENS = ("prompt_tokens", "completion_tokens")

SYSTEM = """\
You choose which vLLM serving config to measure next on one GPU, to find the config that serves a traffic \
profile best within a quality limit. Every config you pick starts a vLLM server and costs minutes of GPU time, \
so pick the config most likely to beat the best one so far, and stop once no remaining config is likely to.

Answer with one JSON object and nothing else:
{"config": "<a line of the allowed list, verbatim>", "hypothesis": "<one or two sentences: why, citing the \
measurements>", "expected_chat": <chat capacity as a multiple of the baseline's>, "expected_batch": <batch \
throughput as a multiple of the baseline's>}
or, to stop:
{"config": null, "hypothesis": "<why no remaining config is worth a server>"}"""


class PlannerError(Exception):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    usage: dict  # token counts as the endpoint reports them, such as prompt_tokens; empty if it reports none


def take_key(env: MutableMapping[str, str], name: str) -> str | None:
    """The value of variable name, removed from env so that processes started with env do not inherit it."""
    return env.pop(name, None) or None


@dataclass(frozen=True)
class Llm:
    """A chat model behind an OpenAI-compatible endpoint, such as OpenRouter or `vllm serve`."""

    url: str
    model: str
    key: str | None = field(default=None, repr=False)
    timeout_s: float = 300
    transport: httpx.BaseTransport | None = None

    def complete(self, messages: list[dict]) -> Reply:
        """The reply to messages, retried while the endpoint is rate limited or unavailable."""
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        # Only what every endpoint accepts: Claude models from 4.7 on refuse a temperature below 1, and
        # Anthropic's endpoint refuses response_format json_object. parse() finds the JSON in plain text.
        body = {"model": self.model, "messages": messages}
        with httpx.Client(base_url=self.url, headers=headers, timeout=self.timeout_s, transport=self.transport) as c:
            problem = ""
            for attempt in range(1, RETRIES + 1):
                wait = 15.0 * attempt
                try:
                    resp = c.post("/chat/completions", json=body)
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    problem = f"{type(e).__name__}: {e}"
                else:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = {}
                    if not isinstance(data, dict):
                        data = {}
                    if resp.status_code == 200 and data.get("choices"):
                        return Reply(data["choices"][0]["message"].get("content") or "", data.get("usage") or {})
                    # OpenRouter can answer 200 with the provider's error in the body; not every error is an object
                    error = data.get("error")
                    code = error.get("code", resp.status_code) if isinstance(error, dict) else resp.status_code
                    problem = f"HTTP {code}: {resp.text[:300]}"
                    if code != 429 and not (isinstance(code, int) and code >= 500):
                        raise PlannerError(problem)
                    after = resp.headers.get("retry-after", "")
                    wait = float(after) if after.replace(".", "", 1).isdigit() else wait
                if attempt < RETRIES:
                    log.info("plan     %s; retrying in %.0f s", problem, min(wait, MAX_WAIT_S))
                    time.sleep(min(wait, MAX_WAIT_S))
            raise PlannerError(f"{problem}, {RETRIES} tries")


@dataclass(frozen=True)
class Proposal:
    config: Candidate | None  # None: stop
    hypothesis: str
    expected_chat: float | None = None
    expected_batch: float | None = None
    tokens: dict[str, int] | None = None  # the endpoint's token counts, for proposers that call one


@dataclass(frozen=True)
class Step:
    """What a proposer is shown at one step of plan()."""

    session: LoadSession
    goal: Goal
    target: str  # "chat" or "batch"
    journal: list[dict]  # this session's earlier picks, as plan.jsonl holds them
    options: list[Candidate]  # the configs it may pick, from remaining()
    untried: list[str]  # settings whose effect under load is not known yet, from untried()
    servers_left: int
    max_servers: int
    step_file: Path  # <run>/plan/<session>-NN.json; a proposer may record its exchange there


class Proposer(Protocol):
    """Picks the config plan() loads next, or stops."""

    name: str  # in plan.jsonl and the log

    def propose(self, step: Step) -> Proposal:
        """One of step.options, or a stop; PlannerError hands the session to the rules."""
        ...


def _key(label: str) -> frozenset[str]:
    return frozenset(label.split())


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def parse(reply: str, options: list[Candidate], untried: Sequence[str] = ()) -> Proposal:
    """The proposal in a reply; its config must be one of options, its settings in any order.

    A stop counts only once no setting is left untried.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the reply holds no JSON object")
    d = json.loads(reply[start : end + 1])
    hypothesis = str(d.get("hypothesis") or "")
    if d.get("config") is None:
        if untried:
            raise ValueError(
                f"stopping is allowed once each setting's effect under load is known; not yet: {', '.join(untried)}"
            )
        return Proposal(None, hypothesis)
    by_key = {_key(c.label()): c for c in options}
    cand = by_key.get(_key(str(d["config"])))
    if cand is None:
        raise ValueError(f"{d['config']!r} is not in the allowed list")
    return Proposal(cand, hypothesis, _number(d.get("expected_chat")), _number(d.get("expected_batch")))


def allowed(ctx: Context, records: list[dict], max_kl: float) -> list[Candidate]:
    """Every combination of values that each ran offline on its own within max_kl, and that can run here."""
    values = {k: [DEFAULTS[k]] for k in KNOBS}
    for r in records:
        changes = candidate_of(r).changes()
        if r["stage"] == "screen" and len(changes) == 1 and r["status"] == "ok" and r["kl"] <= max_kl:
            knob, value = next(iter(changes.items()))
            values[knob].append(value)
    cands = [Candidate(**dict(zip(KNOBS, combo))) for combo in product(*(values[k] for k in KNOBS))]
    return [c for c in cands if c.changes() and infeasible_reason(c, ctx) is None]


def _f(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _x(gain: float | None) -> str:
    return "-" if gain is None else f"{gain:.2f}x"


def _weights_line(ctx: Context) -> str:
    ckpts = "; ".join(f"{c.repo} ({c.method}, {c.bits}-bit)" for c in ctx.checkpoints)
    return "- weights: base (the model's own weights), fp8 (quantized by vLLM while loading)" + (
        f", or a quantized checkpoint: {ckpts}" if ckpts else ""
    )


def brief(
    session: LoadSession,
    goal: Goal,
    target: str,
    journal: list[dict],
    options: list[Candidate],
    untried: Sequence[str],
    servers_left: int,
    max_servers: int,
) -> str:
    """What the model sees at each step."""
    ctx, slo, gpu = session.ctx, session.slo, session.ctx.gpu
    loaded = load_records(session.run.path)
    base = next((r for r in loaded if r["label"] == "baseline" and r.get("session") == session.id), None)
    sampling = base.get("sampling") if base else None
    other = "batch" if target == "chat" else "chat"
    lines = [
        (
            f"Machine: {gpu.name}, compute capability {gpu.compute_cap}, {gpu.total_mib:,} MiB, vLLM "
            f"{ctx.host.get('vllm')}{' under WSL' if ctx.host.get('wsl') else ''}, "
            f"{'with' if ctx.host.get('nvcc') else 'without'} nvcc. Every engine gets gpu_memory_utilization {ctx.util}."
        ),
        (
            f"Model: {ctx.model}, {ctx.base_bytes / GIB:.2f} GiB of weights, "
            f"{kv_bytes_per_token(ctx.shape) // 1024} KiB of KV cache per token, max_model_len {ctx.max_model_len}."
        ),
        "",
        "Settings:",
        _weights_line(ctx),
        "- kv_cache_dtype: auto (the model's dtype) or a smaller cache format",
        "- attention_backend: auto (vLLM picks; the backend column shows what it picked) or a named backend",
        f"- speculative: none, or ngram (prompt lookup, {NGRAM['num_speculative_tokens']} draft tokens)",
        "",
        "Traffic, "
        + (f"sampled with the model's generation_config {sampling}:" if sampling else "sampled with vLLM's defaults:"),
        (
            f"- chat: {PROMPT_LEN}-token prompts and {OUTPUT_LEN}-token answers arriving as a Poisson process. Capacity "
            f"is the highest request rate at which {slo.share:.0%} of requests get their first token within "
            f"{slo.ttft_ms:.0f} ms and the later ones at {slo.tpot_ms:.0f} ms per token or less, averaged over the answer."
        ),
        f"- batch: {BATCH_REQUESTS} such requests at once; output tokens per second until the last one finishes.",
        "",
        (
            f"Goal: the config for {target}, with the highest {target} figure. Configs within {LOAD_MIN_GAIN:.0%} of "
            f"the best count as tied; among them the {other} figure decides the same way, then the lower KL. "
            f"KL against the base model must stay at most {goal.max_kl}."
        ),
        "",
        (
            "Offline trials, in one engine that prefills every prompt up front and then decodes in pure batches. "
            "A server instead mixes the prefills of arriving requests into its decode batches, so offline speed "
            "can rank configs differently than load does; attention backends in particular differ in how much a "
            "prefill slows the decodes beside it."
        ),
        "config | KL | tok/s | KV cache tokens",
    ]
    for r in session.run.records:
        if r["stage"] not in ("screen", "combine"):
            continue
        if r["status"] == "ok":
            lines.append(f"{r['label']} | {r['kl']:.4f} | {r['tput_tok_s']:.0f} | {r['kv_tokens']:,}")
        else:
            lines.append(f"{r['label']} | {r['status']}: {r.get('error')}")
    lines += [
        "",
        "Under load, as multiples of the baseline measured in the same session:",
        (
            "config | chat req/s | chat TPOT p90 ms at capacity | batch tok/s | KV cache tokens "
            "| batch KV cache peak | batch preemptions | attention backend"
        ),
    ]
    for r in loaded:
        if r["status"] != "ok":
            lines.append(f"{r['label']} | {r['status']}: {r.get('error')}")
            continue
        chat, batch = gains(r, session_baseline(loaded, r))
        b = r["batch"]
        lines.append(
            f"{r['label']} | {_f(r['capacity_rps'], '.2f')} ({_x(chat)}) | {_f((r['chat'] or {}).get('tpot_p90_ms'), '.1f')} "
            f"| {b['tput_tok_s']:.0f} ({_x(batch)}) | {_f(r['kv_tokens'], ',')} | "
            f"{_f(None if b['kv_peak'] is None else b['kv_peak'] * 100, '.0f')}% | {_f(b['preemptions'], '.0f')} "
            f"| {r['backend']}"
        )
    if journal:
        lines += ["", "Your earlier picks in this session:", "config | expected chat, batch | outcome"]
        for j in journal:
            lines.append(f"{j['config']} | {_x(j['expected_chat'])}, {_x(j['expected_batch'])} | {j['result']}")
    walls = [r["wall_s"] for r in loaded if r.get("session") == session.id]
    each = f", about {sum(walls) / len(walls) / 60:.0f} minutes each" if walls else ""
    lines += ["", "Allowed configs, not loaded yet:", *(c.label() for c in options)]
    if untried:
        lines += [
            "",
            (
                f"Settings whose effect under load is not known yet: {', '.join(untried)}. A setting's effect is "
                "known once a config with it served, or failed to serve, where the same config without that setting "
                "served (or is the baseline). Offline results can mislead under load, so you may stop only once the "
                "effect of each setting is known."
            ),
        ]
    lines += ["", f"Servers left: {servers_left} of {max_servers}{each}."]
    return "\n".join(lines)


def ask(
    llm: Llm, messages: list[dict], options: list[Candidate], untried: Sequence[str], transcript: list[Reply]
) -> Proposal:
    """Ask for a pick, and once more if the reply names nothing allowed or stops too early. Replies go to transcript."""
    problem = ""
    for _ in range(2):
        reply = llm.complete(messages)
        transcript.append(reply)
        try:
            return parse(reply.text, options, untried)
        except ValueError as e:
            problem = str(e)
        messages = [
            *messages,
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f"{problem}. Answer again with one JSON object, as the system message says."},
        ]
    raise PlannerError(f"{llm.model} answered twice with no allowed config or stop: {problem}")


class LlmProposer:
    """A chat model that reads the brief of each step and answers with a pick; the exchange goes to the step file."""

    def __init__(self, llm: Llm) -> None:
        self.llm = llm
        self.name: str = llm.model

    def messages(self, step: Step) -> list[dict]:
        """The messages the model is sent at step."""
        text = brief(
            step.session,
            step.goal,
            step.target,
            step.journal,
            step.options,
            step.untried,
            step.servers_left,
            step.max_servers,
        )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]

    def propose(self, step: Step) -> Proposal:
        """The model's pick, with the tokens of every reply it took."""
        messages = self.messages(step)
        replies: list[Reply] = []
        try:
            p = ask(self.llm, messages, step.options, step.untried, replies)
        except PlannerError as e:
            step.step_file.write_text(_step(messages, replies, error=str(e)))
            raise
        step.step_file.write_text(_step(messages, replies))
        tokens = {k: sum(int(r.usage.get(k) or 0) for r in replies) for k in TOKENS}
        return dataclasses.replace(p, tokens=tokens)


class AllProposer:
    """Every allowed config in the order of the list. It never stops, so plan() ends once none is left."""

    name: str = "all"

    def propose(self, step: Step) -> Proposal:
        """The first allowed config."""
        return Proposal(step.options[0], "the next allowed config; every one is loaded")


def remaining(session: LoadSession, goal: Goal, journal: list[dict]) -> list[Candidate]:
    """Allowed configs not loaded in any session, not picked in this one, and not over the KL limit offline.

    A config whose every load record is degraded counts as not loaded, as LoadSession.measure serves it again.
    """
    done = {r["label"] for r in load_records(session.run.path) if r["status"] != "degraded"}
    done |= {j["config"] for j in journal}
    failed = {
        r["label"]
        for r in session.run.records
        if r["stage"] == "combine" and (r["status"] != "ok" or r["kl"] > goal.max_kl)
    }
    return [c for c in allowed(session.ctx, session.run.records, goal.max_kl) if c.label() not in done | failed]


def failed_under_load(load: list[dict]) -> set[Candidate]:
    """Configs whose server failed: a record that is neither ok nor degraded, or two degraded ones."""
    degraded = Counter(r["label"] for r in load if r["status"] == "degraded")
    return {candidate_of(r) for r in load if r["status"] not in ("ok", "degraded") or degraded[r["label"]] > 1}


def untried(session: LoadSession, options: list[Candidate]) -> list[str]:
    """Settings that a remaining config changes and whose effect under load no two loaded configs show.

    A config that served shows the effect of each setting it changes from another config that served; so
    does one that failed to serve, as a setting whose configs fail would otherwise keep a stop refused.
    """
    load = load_records(session.run.path)
    served = {candidate_of(r) for r in load if r["status"] == "ok"}
    shown = served | failed_under_load(load)
    known = {k for c in shown for k in c.changes() if c.replace(**{k: DEFAULTS[k]}) in served}
    return [k for k in KNOBS if k not in known and any(k in c.changes() for c in options)]


def _measure(session: LoadSession, searcher: Searcher, cand: Candidate, goal: Goal) -> tuple[str, bool]:
    """Run the pick: its quality trial if it combines settings, then a server if it is within the KL limit.

    The outcome as the journal records it, and whether a server started for it in this session.
    """
    if len(cand.changes()) == 1:
        r = next(r for r in session.run.records if r["stage"] == "screen" and r["label"] == cand.label())
    else:
        r = searcher.stage("combine", [cand], SCREEN)[0]
    if r["status"] != "ok":
        return f"failed offline ({r['status']}: {r.get('error')}); no server started", False
    if r["kl"] > goal.max_kl:
        return f"KL {r['kl']:.4f} offline, over the limit; no server started", False
    if same := session.same_engine(r):
        return f"runs the same engine as {same}; no server started", False
    try:
        rec = session.measure(r)
    except httpx.TransportError as e:
        # measure_config records a server that fails to start; one that dies once it answered raises out of it
        load.wait_for_gpu(session.free_mib)
        return f"the server stopped answering during measurement ({type(e).__name__}: {e}); no record", True
    started = rec.get("session") == session.id
    if rec["status"] != "ok":
        return f"{rec['status']}: {rec.get('error')}", started
    chat, batch = gains(rec, session_baseline(load_records(session.run.path), rec))
    return (
        f"chat {_f(rec['capacity_rps'], '.2f')} req/s ({_x(chat)}), batch {rec['batch']['tput_tok_s']:.0f} tok/s "
        f"({_x(batch)}), KL {r['kl']:.4f}"
    ), started


def _checked(proposer: Proposer, step: Step) -> Proposal:
    p = proposer.propose(step)
    if p.config is None and step.untried:
        raise PlannerError(f"{proposer.name} stopped before the effect of {', '.join(step.untried)} is known")
    if p.config is not None and p.config not in step.options:
        raise PlannerError(f"{proposer.name} picked {p.config.label()!r}, which is not in the allowed list")
    return p


def plan(
    session: LoadSession,
    proposer: Proposer,
    goal: Goal,
    target: str,
    *,
    max_servers: int,
    max_combos: int,
    trial_timeout_s: float,
) -> None:
    """The baseline, then the configs the proposer picks, until it stops or max_servers servers have run.

    A stop before every setting's effect is known, or a pick outside the allowed list, fails the proposer
    as a PlannerError does: the rules finish the session.
    """
    session.baseline()
    searcher = Searcher(session.run, session.ctx, goal, session.env, trial_timeout_s)
    steps_dir = session.run.path / "plan"
    steps_dir.mkdir(exist_ok=True)
    journal: list[dict] = []
    servers = 0
    while servers < max_servers and len(journal) < 2 * max_servers:
        options = remaining(session, goal, journal)
        if not options:
            log.info("plan     every allowed config is loaded")
            return
        step_file = steps_dir / f"{session.id}-{len(journal) + 1:02d}.json"
        left = untried(session, options)
        step = Step(session, goal, target, list(journal), options, left, max_servers - servers, max_servers, step_file)
        try:
            p = _checked(proposer, step)
        except PlannerError as e:
            log.warning("plan     %s; the rules finish the session", e)
            session.search(goal, max_combos, trial_timeout_s)
            return
        if p.config is None:
            log.info("plan     %s stops: %s", proposer.name, p.hypothesis)
            entry = {"config": None, "hypothesis": p.hypothesis, "expected_chat": None, "expected_batch": None}
            _append(session, proposer, entry | {"result": "stop", "server": False, "tokens": p.tokens})
            return
        label = p.config.label()
        log.info(
            "plan     %s picks %s, expecting chat %s, batch %s: %s",
            proposer.name,
            label,
            _x(p.expected_chat),
            _x(p.expected_batch),
            p.hypothesis,
        )
        result, started = _measure(session, searcher, p.config, goal)
        # a degraded server is started twice
        servers = sum(
            r.get("session") == session.id and r["label"] != "baseline" for r in load_records(session.run.path)
        )
        log.info("plan     %s: %s", label, result)
        entry = {
            "config": label,
            "hypothesis": p.hypothesis,
            "expected_chat": p.expected_chat,
            "expected_batch": p.expected_batch,
            "result": result,
            "server": started,
            "tokens": p.tokens,
        }
        journal.append(entry)
        _append(session, proposer, entry)
    log.info("plan     %d of %d servers started", servers, max_servers)


def _step(messages: list[dict], replies: list[Reply], **extra: str) -> str:
    texts, usage = [r.text for r in replies], [r.usage for r in replies]
    return json.dumps({"messages": messages, "replies": texts, "usage": usage, **extra}, indent=2)


def _append(session: LoadSession, proposer: Proposer, entry: dict) -> None:
    with (session.run.path / "plan.jsonl").open("a") as f:
        f.write(json.dumps({"session": session.id, "model": proposer.name, **entry}) + "\n")


def plan_records(run_dir: Path) -> list[dict]:
    """The lines of <run>/plan.jsonl, if any."""
    path = run_dir / "plan.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
