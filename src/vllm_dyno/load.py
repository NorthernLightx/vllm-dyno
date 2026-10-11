"""Online load through `vllm serve`: one server per config, two traffic profiles.

chat   256-token prompts with 128-token answers arriving as a Poisson process. Capacity is the
       highest request rate at which the SLO share of requests meet both the TTFT and the TPOT
       limit, found by doubling the rate and then bisecting, with the failing rate it settles
       on probed twice.
batch  512 such requests at once, more tokens than any config's KV cache holds, so the cache
       or max_num_seqs decides how many run together. Reports output tokens/s.

The search loads every setting the offline run kept within the quality limit, then combines
those that gained under load. The offline screen cannot be the filter: it prefills every prompt
up front and then decodes in pure batches, while a server mixes prefills into its decode batches,
and attention backends differ in how much a prefill slows the decodes beside it.

Every request carries its own cache_salt, so prompts from the pool can repeat without prefix
caching shortcutting prefill. The server keeps vLLM's serving defaults, which differ from the
offline engine's: below 70 GiB of GPU memory, max_num_batched_tokens is 2048 instead of 8192.

Measured requests leave sampling to the server, which applies the model's generation_config as it
would for any client that sets nothing, so the penalties and top-k/top-p a model asks for are part of
what is measured.

Each `dyno load` is a session that measures the baseline first, and gains are against the
baseline of the same session, because load results drift between sessions by more than most
settings change them.

The capacity search assumes the server serves as fast as the GPU lets it, from start to end, so
two checks mark a record degraded, which keeps it out of gains and the recommendation. Before the
search, requests sent one at a time and decoded greedily, as the offline engine decodes, must not
run DEGRADED_TPOT times slower than that engine did; a server that does is not measured further.
Speculative configs are exempt: how fast they decode depends on how much of the text their drafts
guess, and these requests' text is not the offline engine's.
After batch the server is probed at START_RATE again, and its TPOT p50 must stay within
DEGRADED_TPOT times the first probe's: a server that overflowed GPU memory into system memory
stays slow until it restarts.

<run>/load.json      SLO and settings; a resumed load must match them
<run>/load.jsonl     one line per config and session
<run>/load/NNN-*/    server.log of one config
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import random
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from .runner import Run, classify, kill_group, slug, wait_for_gpu
from .search import SCREEN, Goal, Searcher, candidate_of, combinations
from .space import KNOBS, Context, engine_args, serve_command

log = logging.getLogger(__name__)

LOAD_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

PROMPT_LEN = 256
OUTPUT_LEN = 128
BATCH_REQUESTS = 512
START_RATE = 2.0  # req/s; the doubling phase probes the same rates for every config
MIN_RATE = 0.125
MAX_RATE = 256.0
PRECISION = 0.05  # bisection stops once the passing and failing rates are this close
MIN_PROBE_REQUESTS = 20
# vLLM's Triton top-k/top-p kernels compile on first use, separately for ranges of batch sizes up
# to 64 and above it, and each compile holds the server up for a second or more. Its own JIT warmup
# leaves them out, so bursts of every size up to 64 and a few above compile them before measuring.
WARMUP_BATCHES = (*range(1, 65), 80, 128, 256)
WARMUP_PROMPT_LEN = 16
WARMUP_TOKENS = 4
KV_LINE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
BACKEND_LINE = re.compile(r"Using (?:AttentionBackendEnum\.)?([A-Z0-9_]+) (?:attention )?backend")
SAMPLING_LINE = re.compile(r"overridden by the model's `generation_config.json`: `(\{.*?\})`")
# smaller gains are within noise: a repeated measurement can land one bisection step apart,
# and the steps find_capacity takes near the limit are up to 9% wide
LOAD_MIN_GAIN = 0.1
# a lone request's TPOT over the offline engine's, or the recheck's TPOT p50 over the first probe's,
# above which the server is degraded; HTTP streaming, and the few requests that share a step at
# START_RATE, slow decoding by less than that
DEGRADED_TPOT = 1.5
GREEDY = {"temperature": 0.0, "repetition_penalty": 1.0}  # how the offline engine decodes
QUEUE = {"vllm:num_requests_running", "vllm:num_requests_waiting"}
KV_USAGE = {"vllm:kv_cache_usage_perc"}
PREEMPTIONS = {"vllm:num_preemptions_total"}


@dataclass(frozen=True)
class Timing:
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    tokens: int = 0
    error: str | None = None


@dataclass(frozen=True)
class Slo:
    ttft_ms: float = 1000
    tpot_ms: float = 50
    share: float = 0.9  # of requests that must meet both limits

    def met(self, t: Timing) -> bool:
        return t.error is None and t.ttft_s * 1000 <= self.ttft_ms and t.tpot_s * 1000 <= self.tpot_ms

    def needed(self, n: int) -> int:
        """Requests out of n that must meet the SLO."""
        return math.ceil(self.share * n - 1e-9)  # 0.7 * 10 is 7.000000000000001 in floating point


def prompt_pool(data: dict) -> list[list[int]]:
    """PROMPT_LEN-token prompts cut from the run's speed prompts."""
    return [p[i : i + PROMPT_LEN] for p in data["speed_prompts"] for i in range(0, len(p) - PROMPT_LEN + 1, PROMPT_LEN)]


def poisson_starts(rate: float, n: int, seed: int = 0) -> list[float]:
    """Send times of n requests arriving at `rate` per second on average.

    A fixed seed draws the same pattern at every rate, scaled by 1 / rate.
    """
    rng = random.Random(seed)
    starts, t = [], 0.0
    for _ in range(n):
        starts.append(t)
        t += rng.expovariate(rate)
    return starts


async def stream_one(
    client: httpx.AsyncClient, model: str, prompt: list[int], max_tokens: int, params: dict | None = None
) -> Timing:
    """Send one streamed completion, with sampling params if any; time its first token and the tokens after it."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": uuid.uuid4().hex,
        **(params or {}),
    }
    t0 = time.perf_counter()
    first, tokens = None, 0
    try:
        async with client.stream("POST", "/v1/completions", json=body) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="replace")
                return Timing(error=f"HTTP {resp.status_code}: {text[:200]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("choices") and first is None:
                    first = time.perf_counter()
                if chunk.get("usage"):
                    tokens = chunk["usage"]["completion_tokens"]
    except httpx.HTTPError as e:
        return Timing(error=f"{type(e).__name__}: {e}")
    end = time.perf_counter()
    if first is None or tokens < 2:
        return Timing(tokens=tokens, error=f"streamed {tokens} tokens")
    return Timing(first - t0, (end - first) / (tokens - 1), tokens)


@dataclass(frozen=True)
class Sent:
    timings: list[Timing]  # of the requests that finished
    wall_s: float
    cpu_s: float  # the client's
    kv_peak: float | None = None  # highest KV cache usage sampled, 0 to 1
    preemptions: float | None = None


def metric(text: str, names: set[str]) -> float:
    """Sum of the named series in Prometheus text, over all their labels."""
    total = 0.0
    for line in text.splitlines():
        if line.split("{", 1)[0].split(" ", 1)[0] in names:
            total += float(line.rsplit(" ", 1)[1])
    return total


async def watch(client: httpx.AsyncClient, stop: asyncio.Event) -> tuple[float, float]:
    """Peak KV cache usage and the preemptions counted until `stop` is set, from /metrics."""
    first = count = metric((await client.get("/metrics")).text, PREEMPTIONS)
    peak = 0.0
    while not stop.is_set():
        await asyncio.sleep(0.5)
        text = (await client.get("/metrics")).text
        peak, count = max(peak, metric(text, KV_USAGE)), metric(text, PREEMPTIONS)
    return peak, count - first


async def send(
    url: str,
    model: str,
    prompts: list[list[int]],
    starts: list[float],
    max_tokens: int,
    slo: Slo | None = None,
    *,
    watch_server: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
    params: dict | None = None,
) -> Sent:
    """Send prompts[i] at starts[i] seconds. With an SLO, stop once too many requests have missed it."""
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(None, connect=10)
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=timeout, transport=transport) as client:
        stop = asyncio.Event()
        watcher = asyncio.create_task(watch(client, stop)) if watch_server else None
        t0, cpu0 = time.perf_counter(), time.process_time()

        async def one(prompt: list[int], start: float) -> Timing:
            await asyncio.sleep(max(0.0, t0 + start - time.perf_counter()))
            return await stream_one(client, model, prompt, max_tokens, params)

        tasks = [asyncio.create_task(one(p, s)) for p, s in zip(prompts, starts)]
        allowed = len(tasks) - slo.needed(len(tasks)) if slo else len(tasks)
        done: list[Timing] = []
        missed = 0
        for finished in asyncio.as_completed(tasks):
            t = await finished
            done.append(t)
            if slo and not slo.met(t):
                missed += 1
                if missed > allowed:
                    break
        for task in tasks:
            task.cancel()  # closing a stream makes vLLM abort the request
        await asyncio.gather(*tasks, return_exceptions=True)
        wall_s, cpu_s = time.perf_counter() - t0, time.process_time() - cpu0
        stop.set()
        if watcher is None:
            return Sent(done, wall_s, cpu_s)
        kv_peak, preemptions = await watcher
        return Sent(done, wall_s, cpu_s, kv_peak, preemptions)


def quantile(xs: list[float], q: float) -> float | None:
    """Nearest-rank quantile; None for no values."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000


def summary(timings: list[Timing], n: int) -> dict:
    """Latency percentiles and errors of one batch of requests, n of which were sent."""
    ok = [t for t in timings if t.error is None]
    errors = [t.error for t in timings if t.error]
    return {
        "requests": n,
        "finished": len(timings),
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
        "ttft_p50_ms": _ms(quantile([t.ttft_s for t in ok], 0.5)),
        "ttft_p90_ms": _ms(quantile([t.ttft_s for t in ok], 0.9)),
        "tpot_p50_ms": _ms(quantile([t.tpot_s for t in ok], 0.5)),
        "tpot_p90_ms": _ms(quantile([t.tpot_s for t in ok], 0.9)),
    }


def warm_up(url: str, model: str, pool: list[list[int]]) -> None:
    """Send short requests in bursts of each size in WARMUP_BATCHES, sampled as the server samples."""
    for n in WARMUP_BATCHES:
        prompts = [pool[i % len(pool)][:WARMUP_PROMPT_LEN] for i in range(n)]
        asyncio.run(send(url, model, prompts, [0.0] * n, WARMUP_TOKENS))


def chat_probe(url: str, model: str, pool: list[list[int]], rate: float, slo: Slo, probe_s: float) -> dict:
    """Poisson arrivals at `rate` over `probe_s` seconds; the share of requests that met the SLO."""
    n = max(MIN_PROBE_REQUESTS, round(rate * probe_s))
    prompts = [pool[i % len(pool)] for i in range(n)]
    sent = asyncio.run(send(url, model, prompts, poisson_starts(rate, n), OUTPUT_LEN, slo, watch_server=True))
    met = sum(slo.met(t) for t in sent.timings)  # requests cut off by an early stop count as missed
    return {
        "rate": rate,
        "met": met,
        "share_met": met / n,
        "stopped_early": len(sent.timings) < n,
        **_server_side(sent),
        **summary(sent.timings, n),
    }


def batch_run(url: str, model: str, pool: list[list[int]], n: int = BATCH_REQUESTS) -> dict:
    """n requests sent at once; output tokens per second until the last one finishes."""
    prompts = [pool[i % len(pool)] for i in range(n)]
    sent = asyncio.run(send(url, model, prompts, [0.0] * n, OUTPUT_LEN, watch_server=True))
    tokens = sum(t.tokens for t in sent.timings if t.error is None)
    return {"tput_tok_s": tokens / sent.wall_s, **_server_side(sent), **summary(sent.timings, n)}


def _server_side(sent: Sent) -> dict:
    return {"client_cpu": sent.cpu_s / sent.wall_s, "kv_peak": sent.kv_peak, "preemptions": sent.preemptions}


def _f(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _cache_note(p: dict) -> str:
    peak = None if p["kv_peak"] is None else p["kv_peak"] * 100
    return f"KV cache peak {_f(peak, '.0f')}%, {_f(p['preemptions'], '.0f')} preemptions"


def find_capacity(passes: Callable[[float], bool]) -> tuple[float | None, float | None]:
    """The highest rate found to pass and the lowest found to fail.

    Doubles from START_RATE until a rate fails, or halves until one passes, then bisects until
    the two are within PRECISION. The failing rate is then probed once more, and while it
    passes the search goes on above it and ends with the same check, so the failing rate
    returned failed twice. The passing rate is None if MIN_RATE fails; the failing one is None
    if MAX_RATE passes.
    """
    failed: list[float] = []

    def probe(rate: float) -> bool:
        if passes(rate):
            return True
        failed.append(rate)
        return False

    lo: float | None = None
    hi: float | None = None
    if probe(START_RATE):
        lo, hi = _double(probe, START_RATE)
    else:
        hi = START_RATE
        while hi / 2 >= MIN_RATE:
            if probe(hi / 2):
                lo = hi / 2
                break
            hi /= 2
    if lo is None or hi is None:
        return lo, hi
    lo, hi = _bisect(probe, lo, hi)
    # a single failure can be a passing stall of the server, such as a kernel compiled on first use
    while passes(hi):
        lo = hi
        hi = min((rate for rate in failed if rate > lo), default=None)
        if hi is None:
            lo, hi = _double(probe, lo)
            if hi is None:
                return lo, None
        lo, hi = _bisect(probe, lo, hi)
    return lo, hi


def _double(probe: Callable[[float], bool], lo: float) -> tuple[float, float | None]:
    while lo * 2 <= MAX_RATE:
        if not probe(lo * 2):
            return lo, lo * 2
        lo *= 2
    return lo, None


def _bisect(probe: Callable[[float], bool], lo: float, hi: float) -> tuple[float, float]:
    while hi / lo > 1 + PRECISION:
        mid = math.sqrt(lo * hi)  # the bracket narrows as a ratio, so split it geometrically
        if probe(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


def slow_start(offline_ms: float, lone_ms: float | None) -> str | None:
    """How a server decoded requests sent one at a time slower than the same engine offline; None if within DEGRADED_TPOT."""
    if lone_ms is None or lone_ms <= DEGRADED_TPOT * offline_ms:
        return None
    return f"a lone request decoded at {lone_ms:.1f} ms per token, against {offline_ms:.1f} ms offline"


def slowdown(first: dict, recheck: dict) -> str | None:
    """How the server slowed down between its first probe and the recheck after batch; None if it did not."""
    before, after = first["tpot_p50_ms"], recheck["tpot_p50_ms"]
    rate = f"{recheck['rate']:g} req/s"
    if before is None or after is None:
        return f"no request at {rate} finished without error {'at the start' if before is None else 'after batch'}"
    if after > DEGRADED_TPOT * before:
        return f"TPOT p50 at {rate} rose from {before:.1f} to {after:.1f} ms during the measurement"
    return None


def wait_idle(url: str, timeout_s: float = 60) -> None:
    """Block until the server holds no requests, such as those aborted by an early stop."""
    deadline = time.monotonic() + timeout_s
    while metric(httpx.get(url + "/metrics", timeout=5).text, QUEUE) > 0:
        if time.monotonic() > deadline:
            log.warning("server still busy %.0f s after a probe", timeout_s)
            return
        time.sleep(0.5)


class ServerFailed(Exception):
    def __init__(self, status: str, error: str):
        super().__init__(error)
        self.status, self.error = status, error


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _healthy(url: str) -> bool:
    try:
        return httpx.get(url + "/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@contextlib.contextmanager
def serving(argv: list[str], log_path: Path, env: dict[str, str], timeout_s: float) -> Generator[str]:
    """Run `vllm serve` and yield its base URL once /health answers. Its process group dies on exit."""
    url = f"http://127.0.0.1:{_free_port()}"
    cmd = [*argv, "--host", "127.0.0.1", "--port", url.rsplit(":", 1)[1], "--disable-uvicorn-access-log"]
    with log_path.open("w") as out:
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, env={**env, "PYTHONUNBUFFERED": "1"}, start_new_session=True
        )
        try:
            deadline = time.monotonic() + timeout_s
            while not _healthy(url):
                if proc.poll() is not None:
                    raise ServerFailed(*classify(log_path.read_text(errors="replace")))
                if time.monotonic() > deadline:
                    raise ServerFailed("timeout", f"/health did not answer within {timeout_s:.0f} s")
                time.sleep(2)
            yield url
        finally:
            kill_group(proc)
            proc.wait()


def targets(records: list[dict], labels: list[str]) -> list[dict]:
    """The trial record of each named config."""
    out = []
    for label in labels:
        r = next((r for r in records if r["label"] == label and r["status"] == "ok"), None)
        if r is None:
            raise SystemExit(f"no successful trial of {label!r} in this run")
        out.append(r)
    return out


def attention_backends(log_text: str) -> str | None:
    """The attention backends a server log says it picked, in order, joined by "+".

    turboquant KV keeps some layers on FLASH_ATTN, so its server names two.
    """
    return "+".join(dict.fromkeys(BACKEND_LINE.findall(log_text))) or None


def load_goal(run_dir: Path) -> str:
    """The traffic profile, "chat" or "batch", that the last `dyno load` of the run recommends a config for."""
    path = run_dir / "load.json"
    return json.loads(path.read_text()).get("goal", "chat") if path.exists() else "chat"


def load_records(run_dir: Path) -> list[dict]:
    """The lines of <run>/load.jsonl, if any."""
    path = run_dir / "load.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    """Start a server for the trial record's config and run both profiles against it."""
    cand = candidate_of(r)
    model = engine_args(cand, ctx)["model"]
    log_path = out_dir / "server.log"
    started = time.monotonic()
    record = {"label": r["label"], **asdict(cand), "slo": asdict(slo), "load_sha": LOAD_SHA}
    probes: list[dict] = []
    try:
        with serving(shlex.split(serve_command(cand, ctx)), log_path, env, timeout_s) as url:
            started_log = log_path.read_text(errors="replace")
            kv, backends = KV_LINE.search(started_log), attention_backends(started_log)
            sampling = SAMPLING_LINE.search(started_log)
            warm_up(url, model, pool)
            # the faster of two, so a kernel compiled on first use does not decide
            lone = [asyncio.run(send(url, model, pool[:1], [0.0], OUTPUT_LEN, params=GREEDY)) for _ in range(2)]
            lone_ms = min((t.tpot_s * 1000 for s in lone for t in s.timings if t.error is None), default=None)
            log.info(
                "load     %-55s lone request: TPOT %s ms, %.1f ms offline", r["label"], _f(lone_ms, ".1f"), r["tpot_ms"]
            )
            if cand.speculative == "none" and (slow := slow_start(r["tpot_ms"], lone_ms)):
                raise ServerFailed("degraded", slow)

            def passes(rate: float) -> bool:
                p = chat_probe(url, model, pool, rate, slo, probe_s)
                probes.append(p)
                if p["stopped_early"]:
                    outcome = f"stopped once {p['finished'] - p['met']} of {p['finished']} finished missed the SLO"
                else:
                    outcome = f"{p['share_met']:.0%} within SLO"
                log.info(
                    "load     %-55s chat %6.2f req/s: %s, TTFT p90 %s ms, TPOT p90 %s ms, %s",
                    r["label"],
                    rate,
                    outcome,
                    _f(p["ttft_p90_ms"], ".0f"),
                    _f(p["tpot_p90_ms"], ".1f"),
                    _cache_note(p),
                )
                if p["client_cpu"] > 0.8:
                    log.warning(
                        "load     the client used %.0f%% of a core; it may be the bottleneck", p["client_cpu"] * 100
                    )
                wait_idle(url)
                return p["met"] >= slo.needed(p["requests"])

            lo, hi = find_capacity(passes)
            batch = batch_run(url, model, pool)
            log.info(
                "load     %-55s batch %d requests: %.0f tok/s, TPOT p50 %s ms, %s",
                r["label"],
                batch["requests"],
                batch["tput_tok_s"],
                _f(batch["tpot_p50_ms"], ".1f"),
                _cache_note(batch),
            )
            recheck = chat_probe(url, model, pool, START_RATE, slo, probe_s)
            wait_idle(url)
            first = next(p for p in probes if p["rate"] == START_RATE)
            slowed = slowdown(first, recheck)
            log.log(
                logging.WARNING if slowed else logging.INFO,
                "load     %-55s chat %6.2f req/s after batch: TPOT p50 %s ms, %s ms at the start",
                r["label"],
                START_RATE,
                _f(recheck["tpot_p50_ms"], ".1f"),
                _f(first["tpot_p50_ms"], ".1f"),
            )
    except ServerFailed as e:
        return record | {"status": e.status, "error": e.error, "wall_s": round(time.monotonic() - started, 1)}
    at_capacity = next((p for p in reversed(probes) if p["rate"] == lo), None)  # lo may have failed once
    return record | {
        "status": "degraded" if slowed else "ok",
        "error": slowed,
        "kv_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "backend": backends,
        "sampling": sampling.group(1) if sampling else None,  # None: vLLM's own defaults
        "lone_tpot_ms": lone_ms,
        "capacity_rps": lo,
        "capped": hi is None,
        "chat": at_capacity,
        "probes": probes,
        "batch": batch,
        "recheck": recheck,
        "wall_s": round(time.monotonic() - started, 1),
    }


def session_baseline(load: list[dict], r: dict) -> dict | None:
    """The baseline measured in the same session as load record r."""
    return next((b for b in load if b["label"] == "baseline" and b.get("session") == r.get("session")), None)


def gains(r: dict, base: dict | None) -> tuple[float | None, float | None]:
    """Chat capacity and batch throughput of load record r as multiples of its session baseline's."""
    if base is None or r["status"] != "ok" or base["status"] != "ok":
        return None, None
    chat = (r["capacity_rps"] or 0.0) / base["capacity_rps"] if base["capacity_rps"] else None
    return chat, r["batch"]["tput_tok_s"] / base["batch"]["tput_tok_s"]


def promising_under_load(load: list[dict], min_gain: float) -> dict[str, dict[str, float]]:
    """Per setting, the values that on their own beat their session baseline by min_gain on chat or batch."""
    out: dict[str, dict[str, float]] = {k: {} for k in KNOBS}
    for r in load:
        changes = candidate_of(r).changes()
        best = max((g for g in gains(r, session_baseline(load, r)) if g is not None), default=0.0)
        if len(changes) == 1 and best >= 1 + min_gain:
            knob, value = next(iter(changes.items()))
            out[knob][value] = best
    return out


@dataclass(frozen=True)
class LoadSession:
    """One `dyno load` invocation, writing to <run>/load.jsonl."""

    run: Run
    ctx: Context
    pool: list[list[int]]
    slo: Slo
    probe_s: float
    env: dict[str, str]
    server_timeout_s: float
    free_mib: int  # GPU memory free before the session; each server must give it back
    id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))

    def measure(self, r: dict) -> dict:
        """The load record of trial r's config: one from an earlier session if any, else measured now.

        The baseline is measured again in every session.
        """
        done = load_records(self.run.path)
        earlier = next((x for x in done if x["label"] == r["label"]), None)
        if earlier and r["label"] != "baseline":
            log.info("load     %s recorded earlier", r["label"])
            return earlier
        out_dir = self.run.path / "load" / f"{len(done):03d}-{slug(r['label'])}"
        out_dir.mkdir(parents=True, exist_ok=True)
        record = measure_config(
            r,
            self.ctx,
            self.pool,
            self.slo,
            self.probe_s,
            env=self.env,
            out_dir=out_dir,
            timeout_s=self.server_timeout_s,
        )
        record |= {"session": self.id, "dir": str(out_dir.relative_to(self.run.path))}
        with (self.run.path / "load.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if record["status"] != "ok":
            log.info("load     %s %s: %s", r["label"], record["status"], record["error"])
        wait_for_gpu(self.free_mib)
        return record

    def baseline(self) -> dict:
        r = next((r for r in self.run.records if r["label"] == "baseline" and r["status"] == "ok"), None)
        if r is None:
            raise SystemExit(f"{self.run.path} has no successful baseline trial")
        record = self.measure(r)
        if record["status"] != "ok":
            where = self.run.path / record["dir"]
            raise SystemExit(f"the baseline did not serve ({record['status']}: {record['error']}); see {where}")
        return record

    def same_engine(self, r: dict) -> str | None:
        """A loaded config that differs from r only in naming the attention backend its server picked anyway."""
        cand = candidate_of(r)
        if cand.attention_backend == "auto":
            return None
        auto = cand.replace(attention_backend="auto").label()
        done = load_records(self.run.path)
        if any(x["label"] == auto and x["status"] == "ok" and x.get("backend") == cand.attention_backend for x in done):
            return auto
        return None

    def search(self, goal: Goal, max_combos: int, trial_timeout_s: float) -> None:
        """The baseline, every single setting within the quality limit, then combinations of those that gained."""
        self.baseline()
        for r in self.run.records:
            single = r["stage"] == "screen" and len(candidate_of(r).changes()) == 1
            if single and r["status"] == "ok" and r["kl"] <= goal.max_kl:
                self.measure(r)
        good = promising_under_load(load_records(self.run.path), LOAD_MIN_GAIN)
        # pairs first, so a triple whose explicit backend changes nothing is seen to match a pair already loaded
        combos = sorted(combinations(good, limit=sys.maxsize), key=lambda c: len(c.changes()))[:max_combos]
        if not combos:
            log.info("load     fewer than two settings gained %.0f%%; nothing to combine", LOAD_MIN_GAIN * 100)
            return
        quality = Searcher(self.run, self.ctx, goal, self.env, trial_timeout_s).stage("combine", combos, SCREEN)
        for r in quality:
            if r["status"] != "ok" or r["kl"] > goal.max_kl:
                continue
            if same := self.same_engine(r):
                log.info("load     %s runs the same engine as %s", r["label"], same)
                continue
            self.measure(r)
