"""Online load through `vllm serve`: one server per config, two traffic profiles.

chat   256-token prompts with 128-token answers arriving as a Poisson process. Capacity is the
       highest request rate at which the SLO share of requests meet both the TTFT and the TPOT
       limit, found by doubling the rate and then bisecting.
batch  512 such requests at once, more tokens than any config's KV cache holds, so the cache
       decides how many run together. Reports output tokens/s.

Every request carries its own cache_salt, so prompts from the pool can repeat without prefix
caching shortcutting prefill. The server keeps vLLM's serving defaults, which differ from the
offline engine's: below 70 GiB of GPU memory, max_num_batched_tokens is 2048 instead of 8192.

<run>/load.json      SLO and settings; a resumed load must match them
<run>/load.jsonl     one line per config
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
import time
import uuid
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .runner import classify, kill_group
from .search import candidate_of
from .space import Context, engine_args, serve_command

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
KV_LINE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
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


async def stream_one(client: httpx.AsyncClient, model: str, prompt: list[int], max_tokens: int) -> Timing:
    """Send one streamed completion; time its first token and the tokens after it."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": uuid.uuid4().hex,
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
            return await stream_one(client, model, prompt, max_tokens)

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
    the two are within PRECISION. The passing rate is None if MIN_RATE fails; the failing one
    is None if MAX_RATE passes.
    """
    lo: float | None = None
    hi: float | None = None
    if passes(START_RATE):
        lo = START_RATE
        while lo * 2 <= MAX_RATE:
            if not passes(lo * 2):
                hi = lo * 2
                break
            lo *= 2
    else:
        hi = START_RATE
        while hi / 2 >= MIN_RATE:
            if passes(hi / 2):
                lo = hi / 2
                break
            hi /= 2
    if lo is None or hi is None:
        return lo, hi
    while hi / lo > 1 + PRECISION:
        mid = math.sqrt(lo * hi)  # the bracket narrows as a ratio, so split it geometrically
        if passes(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


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
    """Trial records to load: the named configs, or every config that finished the confirm stage."""
    if not labels:
        return [r for r in records if r["stage"] == "confirm" and r["status"] == "ok"]
    out = []
    for label in labels:
        r = next((r for r in records if r["label"] == label and r["status"] == "ok"), None)
        if r is None:
            raise SystemExit(f"no successful trial of {label!r} in this run")
        out.append(r)
    return out


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
            kv = KV_LINE.search(log_path.read_text(errors="replace"))
            asyncio.run(send(url, model, pool[:4], [0.0] * 4, 16))  # first requests pay one-off setup costs

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
    except ServerFailed as e:
        return record | {"status": e.status, "error": e.error, "wall_s": round(time.monotonic() - started, 1)}
    at_capacity = next((p for p in probes if p["rate"] == lo), None)
    return record | {
        "status": "ok",
        "error": None,
        "kv_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "capacity_rps": lo,
        "capped": hi is None,
        "chat": at_capacity,
        "probes": probes,
        "batch": batch,
        "wall_s": round(time.monotonic() - started, 1),
    }
