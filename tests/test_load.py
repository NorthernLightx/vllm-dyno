import asyncio
import contextlib
import json
from dataclasses import asdict

import httpx
import pytest

from vllm_dyno import load
from vllm_dyno.load import (
    GREEDY,
    QUEUE,
    LoadSession,
    Sent,
    Slo,
    Timing,
    find_capacity,
    gains,
    load_records,
    metric,
    poisson_starts,
    promising_under_load,
    quantile,
    send,
    session_baseline,
    targets,
)
from vllm_dyno.runner import Run
from vllm_dyno.search import Goal
from vllm_dyno.space import KNOBS, Candidate


def completion(tokens: int, text: str = "x") -> bytes:
    """A streamed completion as vLLM sends it: one chunk per token, then usage, then [DONE]."""
    chunks: list[dict] = [{"choices": [{"index": 0, "text": text}]} for _ in range(tokens)]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": tokens}})
    # like vLLM's pydantic, json escapes only control characters below U+0020 when ensure_ascii is off
    return b"".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


class Pieces(httpx.AsyncByteStream):
    def __init__(self, data: bytes, size: int):
        self.data, self.size = data, size

    async def __aiter__(self):
        for i in range(0, len(self.data), self.size):
            yield self.data[i : i + self.size]


def run_send(handler, prompts, slo=None, watch_server=False):
    starts = [0.0] * len(prompts)
    transport = httpx.MockTransport(handler)
    return asyncio.run(
        send("http://server", "org/base", prompts, starts, 8, slo, watch_server=watch_server, transport=transport)
    )


def test_requests_send_token_ids_each_with_its_own_cache_salt_and_leave_sampling_to_the_server():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=completion(8))

    timings = run_send(handler, [[1, 2, 3], [4, 5, 6]]).timings
    assert [(t.tokens, t.error) for t in timings] == [(8, None), (8, None)]
    assert sorted(b["prompt"] for b in bodies) == [[1, 2, 3], [4, 5, 6]]
    assert all(b["ignore_eos"] and b["stream"] for b in bodies)
    assert not {"temperature", "top_k", "top_p", "repetition_penalty"} & {k for b in bodies for k in b}
    assert bodies[0]["cache_salt"] != bodies[1]["cache_salt"]


def test_the_health_check_decodes_greedily_like_the_offline_engine():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=completion(8))

    transport = httpx.MockTransport(handler)
    asyncio.run(send("http://server", "org/base", [[1]], [0.0], 8, transport=transport, params=GREEDY))
    assert {k: bodies[0][k] for k in GREEDY} == GREEDY == {"temperature": 0.0, "repetition_penalty": 1.0}


@pytest.mark.parametrize("piece", [7, None])
def test_a_token_whose_text_holds_a_unicode_line_separator_stays_in_its_event(piece):
    body = completion(8, text="a\x85b\u2028c\u2029d")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Pieces(body, piece or len(body)))

    assert [(t.tokens, t.error) for t in run_send(handler, [[1]]).timings] == [(8, None)]


def test_http_errors_and_short_streams_are_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["prompt"] == [0]:
            return httpx.Response(500, content=b"boom")
        return httpx.Response(200, content=completion(1))

    assert sorted(t.error or "" for t in run_send(handler, [[0], [1]]).timings) == [
        "HTTP 500: boom",
        "streamed 1 tokens",
    ]


def test_sending_stops_once_too_many_requests_missed_the_slo():
    # 10% of 20 requests may miss; every request here fails, so the third miss ends the probe
    timings = run_send(lambda request: httpx.Response(200, content=completion(1)), [[1]] * 20, Slo()).timings
    assert len(timings) == 3


def test_slo_needs_both_limits_and_no_error():
    assert [Slo(share=s).needed(10) for s in (0.7, 0.9, 0.95)] == [7, 9, 10]
    slo = Slo(ttft_ms=100, tpot_ms=10)
    assert slo.met(Timing(0.1, 0.01, 128))
    assert not slo.met(Timing(0.11, 0.01, 128))
    assert not slo.met(Timing(0.1, 0.011, 128))
    assert not slo.met(Timing(error="HTTP 500"))


def test_poisson_arrivals_keep_their_pattern_across_rates():
    slow, fast = poisson_starts(2, 1000), poisson_starts(4, 1000)
    assert slow[0] == 0
    assert fast == pytest.approx([t / 2 for t in slow])
    assert slow[-1] / 999 == pytest.approx(0.5, rel=0.1)


@pytest.mark.parametrize("limit", [0.3, 7.3, 100.0])
def test_capacity_brackets_the_limit_within_precision(limit):
    rates = []

    def passes(rate: float) -> bool:
        rates.append(rate)
        return rate <= limit

    lo, hi = find_capacity(passes)
    assert lo is not None and hi is not None
    assert lo <= limit < hi and hi / lo <= 1 + load.PRECISION
    assert rates[0] == load.START_RATE


def test_capacity_reports_the_ends_of_the_range():
    assert find_capacity(lambda rate: False) == (None, load.MIN_RATE)
    assert find_capacity(lambda rate: True) == (load.MAX_RATE, None)


@pytest.mark.parametrize("stall", [load.START_RATE, load.START_RATE * 2, 16.0])
def test_one_failure_that_does_not_repeat_does_not_cap_the_capacity(stall):
    rates = []

    def passes(rate: float) -> bool:
        rates.append(rate)
        return rate <= 20 and not (rate == stall and rates.count(rate) == 1)

    lo, hi = find_capacity(passes)
    assert lo is not None and hi is not None
    assert lo <= 20 < hi and hi / lo <= 1 + load.PRECISION
    assert rates.count(stall) == 2


def test_stalls_at_two_rates_do_not_cap_the_capacity():
    rates = []

    def passes(rate: float) -> bool:
        rates.append(rate)
        return rate <= 20 and not (rate in (4.0, 8.0) and rates.count(rate) == 1)

    lo, hi = find_capacity(passes)
    assert lo is not None and hi is not None
    assert lo <= 20 < hi and hi / lo <= 1 + load.PRECISION
    assert rates.count(4.0) == rates.count(8.0) == rates.count(hi) == 2


def test_the_lowest_failing_rate_is_probed_once_more_and_then_stands():
    rates = []

    def passes(rate: float) -> bool:
        rates.append(rate)
        return rate <= 7.3

    lo, hi = find_capacity(passes)
    assert (lo, hi) == pytest.approx(find_capacity(lambda rate: rate <= 7.3))
    assert rates.count(hi) == 2
    assert rates[-1] == hi


def test_metrics_sum_the_named_series_only():
    metrics = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 7.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0
"""
    assert metric(metrics, QUEUE) == 5


def test_quantile_is_nearest_rank():
    assert quantile([5, 1, 4, 2, 3], 0.5) == 3
    assert quantile(list(range(1, 11)), 0.9) == 9
    assert quantile([], 0.9) is None


def test_targets_are_the_named_configs_from_any_stage():
    records = [
        {"stage": "screen", "label": "baseline", "status": "ok"},
        {"stage": "screen", "label": "kv_cache_dtype=turboquant_k8v4", "status": "ok"},
        {"stage": "screen", "label": "kv_cache_dtype=fp8", "status": "skipped"},
    ]
    assert targets(records, ["kv_cache_dtype=turboquant_k8v4"]) == [records[1]]
    with pytest.raises(SystemExit):
        targets(records, ["kv_cache_dtype=fp8"])


@pytest.mark.parametrize(
    ("lines", "backends"),
    [
        (
            ["Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN']."],
            "FLASH_ATTN",
        ),
        (["Using AttentionBackendEnum.TRITON_ATTN backend."], "TRITON_ATTN"),
        (
            [
                "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN'].",
                "Using TURBOQUANT attention backend out of potential backends: ['TURBOQUANT'].",
            ],
            "FLASH_ATTN+TURBOQUANT",
        ),
        (["Starting vLLM API server"], None),
    ],
)
def test_the_server_log_names_the_attention_backends(lines, backends):
    log_text = "\n".join(f"(EngineCore pid=1) INFO 10-10 15:49:51 [cuda.py:526] {line}" for line in lines)
    assert load.attention_backends(log_text) == backends


def test_the_server_log_names_the_sampling_defaults_from_generation_config():
    line = (
        "(APIServer pid=527) WARNING 10-10 23:17:15 [model.py:1850] Default vLLM sampling parameters have been "
        "overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.1, 'temperature': 0.7, "
        "'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with "
        "`--generation-config vllm`."
    )
    found = load.SAMPLING_LINE.search(line)
    assert found and found.group(1) == "{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}"
    assert load.SAMPLING_LINE.search("Starting vLLM API server") is None


def test_watching_the_server_reports_peak_kv_usage_and_new_preemptions():
    samples = [(0.1, 3), (0.8, 5)]
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/metrics":
            return httpx.Response(200, content=completion(8))
        kv, preemptions = samples[min(len(calls), len(samples) - 1)]
        calls.append(request)
        lines = [
            f'vllm:kv_cache_usage_perc{{engine="0"}} {kv}',
            f'vllm:num_preemptions_total{{engine="0"}} {preemptions}',
        ]
        return httpx.Response(200, text="\n".join(lines))

    sent = run_send(handler, [[1]] * 4, watch_server=True)
    assert len(sent.timings) == 4
    assert (sent.kv_peak, sent.preemptions) == (0.8, 2)


def session(tmp_path, monkeypatch, ctx, sid: str) -> tuple[LoadSession, list[str]]:
    """A load session over a run with a baseline and an fp8 trial; no server starts."""
    measured: list[str] = []

    def fake_measure(r, *args, **kwargs):
        measured.append(r["label"])
        return {"label": r["label"], "status": "ok"}

    monkeypatch.setattr(load, "measure_config", fake_measure)
    monkeypatch.setattr(load, "wait_for_gpu", lambda free_mib: None)
    run = Run(tmp_path)
    if not run.records:
        run.append({"key": "screen|baseline", "stage": "screen", "label": "baseline", "status": "ok"})
        run.append({"key": "screen|weights=fp8", "stage": "screen", "label": "weights=fp8", "status": "ok"})
    return LoadSession(run, ctx, [[1]], Slo(), 30, {}, 600, 6000, id=sid), measured


def test_every_session_measures_the_baseline_again_and_reuses_other_configs(tmp_path, monkeypatch, ctx):
    first, measured = session(tmp_path, monkeypatch, ctx, "night")
    first.baseline()
    first.measure(first.run.records[1])
    second, measured_again = session(tmp_path, monkeypatch, ctx, "morning")
    second.baseline()
    assert second.measure(second.run.records[1])["session"] == "night"
    assert measured == ["baseline", "weights=fp8"]
    assert measured_again == ["baseline"]
    assert [(r["label"], r["session"], r["dir"]) for r in load_records(tmp_path)] == [
        ("baseline", "night", "load/000-baseline"),
        ("weights=fp8", "night", "load/001-weights-fp8"),
        ("baseline", "morning", "load/002-baseline"),
    ]


def test_a_degraded_server_is_started_once_more_and_its_config_measured_again_in_later_sessions(
    tmp_path, monkeypatch, ctx
):
    night, _ = session(tmp_path, monkeypatch, ctx, "night")
    morning, _ = session(tmp_path, monkeypatch, ctx, "morning")
    statuses = iter(["ok", "degraded", "degraded", "degraded", "ok", "ok"])
    monkeypatch.setattr(
        load, "measure_config", lambda r, *a, **k: {"label": r["label"], "status": next(statuses), "error": "slow"}
    )
    night.baseline()
    assert night.measure(night.run.records[1])["status"] == "degraded"
    morning.baseline()
    fp8 = morning.measure(morning.run.records[1])
    records = load_records(tmp_path)
    assert [(r["label"], r["session"], r["status"]) for r in records] == [
        ("baseline", "night", "ok"),
        ("weights=fp8", "night", "degraded"),
        ("weights=fp8", "night", "degraded"),
        ("baseline", "morning", "degraded"),
        ("baseline", "morning", "ok"),
        ("weights=fp8", "morning", "ok"),
    ]
    assert session_baseline(records, fp8) == records[4]


@pytest.mark.parametrize("status", ["oom", "degraded"])
def test_a_baseline_that_does_not_serve_stops_the_session(tmp_path, monkeypatch, ctx, status):
    s, _ = session(tmp_path, monkeypatch, ctx, "night")
    monkeypatch.setattr(
        load, "measure_config", lambda r, *a, **k: {"label": r["label"], "status": status, "error": "x"}
    )
    with pytest.raises(SystemExit, match=f"the baseline did not serve \\({status}: x\\)"):
        s.baseline()


def loaded(label: str, session: str, capacity: float, batch: float, backend: str = "FLASH_ATTN") -> dict:
    cand = Candidate() if label == "baseline" else Candidate(**dict(kv.split("=") for kv in label.split()))
    return {
        "label": label,
        **{k: getattr(cand, k) for k in KNOBS},
        "status": "ok",
        "session": session,
        "capacity_rps": capacity,
        "batch": {"tput_tok_s": batch},
        "backend": backend,
    }


def test_settings_gain_against_their_own_session_on_chat_or_batch():
    records = [
        loaded("baseline", "night", 10, 1000),
        loaded("attention_backend=TRITON_ATTN", "night", 12, 1000),  # chat +20%
        loaded("baseline", "morning", 12, 1000),
        loaded("weights=fp8", "morning", 12, 1150),  # batch +15%
        loaded("kv_cache_dtype=turboquant_k8v4", "morning", 12.5, 1050),  # under 10% on both
        loaded("weights=fp8 attention_backend=TRITON_ATTN", "morning", 20, 2000),  # a combination, not a setting
    ]
    good = promising_under_load(records, 0.1)
    assert good["attention_backend"] == {"TRITON_ATTN": pytest.approx(1.2)}
    assert good["weights"] == {"fp8": pytest.approx(1.15)}
    assert good["kv_cache_dtype"] == {} and good["speculative"] == {}


def trial(stage: str, label: str, kl: float, status: str = "ok") -> dict:
    cand = Candidate() if label == "baseline" else Candidate(**dict(kv.split("=") for kv in label.split()))
    return {"key": f"{stage}|{label}", "stage": stage, "label": label, "status": status, "kl": kl, **asdict(cand)}


def test_search_loads_settings_within_quality_then_combines_those_that_gained(tmp_path, monkeypatch, ctx):
    run = Run(tmp_path)
    for r in [
        trial("screen", "baseline", 0.0),
        trial("screen", "weights=fp8", 0.007),
        trial("screen", "weights=org/base-AWQ", 0.07),  # over the quality limit
        trial("screen", "kv_cache_dtype=int8_per_token_head", 0.013),
        trial("screen", "attention_backend=TRITON_ATTN", 0.0006),
        trial("screen", "speculative=ngram", 0.0),
        trial("screen", "kv_cache_dtype=fp8", 0.0, status="skipped"),
    ]:
        run.append(r)
    results = {  # chat req/s, batch tok/s, attention backend the server picked
        "baseline": (10, 1000, "FLASH_ATTN"),
        "weights=fp8": (10, 1200, "FLASH_ATTN"),
        "kv_cache_dtype=int8_per_token_head": (12, 1200, "TRITON_ATTN"),
        "attention_backend=TRITON_ATTN": (12, 1000, "TRITON_ATTN"),
        "speculative=ngram": (5, 600, "FLASH_ATTN"),
    }
    measured = []

    def fake_measure(r, *args, **kwargs):
        measured.append(r["label"])
        cap, batch, backend = results.get(r["label"], (13, 1300, "TRITON_ATTN"))
        return loaded(r["label"], "", cap, batch, backend)

    class FakeSearcher:
        def __init__(self, *args):
            pass

        def stage(self, stage, cands, tier):
            # fp8 weights with int8 KV miss the quality limit, whatever backend is named
            def kl(c: Candidate) -> float:
                return 0.06 if c.weights == "fp8" and c.kv_cache_dtype == "int8_per_token_head" else 0.01

            return [trial(stage, c.label(), kl(c)) for c in cands]

    monkeypatch.setattr(load, "measure_config", fake_measure)
    monkeypatch.setattr(load, "wait_for_gpu", lambda free_mib: None)
    monkeypatch.setattr(load, "Searcher", FakeSearcher)
    LoadSession(run, ctx, [[1]], Slo(), 30, {}, 600, 6000, id="s").search(Goal(), max_combos=8, trial_timeout_s=900)
    assert measured == [
        "baseline",
        "weights=fp8",
        "kv_cache_dtype=int8_per_token_head",
        "attention_backend=TRITON_ATTN",
        "speculative=ngram",
        # int8 KV already ran TRITON_ATTN, so naming it again is not loaded
        "weights=fp8 attention_backend=TRITON_ATTN",
    ]


def measure(
    tmp_path,
    monkeypatch,
    ctx,
    after: float | None,
    lone: float = 8.5,
    label: str = "weights=fp8",
    compiles: int = 0,
    starts: list | None = None,
    dies_at: float | None = None,
) -> dict:
    """measure_config on a fake server that serves up to 20 req/s at TPOT p50 8.9 ms, and at `after` ms once batch ran.

    Requests sent one at a time decode at `lone` ms per token, and the engine did 8.0 offline. The first
    `compiles` starts compile the model's graph and get a smaller KV cache; each start is appended to `starts`.
    A server given `dies_at` logs a CUDA error and stops answering when probed at that rate.
    """
    batched, lone_params, warmup_sizes = [], [], []
    starts = [] if starts is None else starts

    @contextlib.contextmanager
    def fake_serving(argv, log_path, env, timeout_s):
        starts.append(argv)
        if len(starts) <= compiles:
            log_path.write_text(f"{load.COMPILED_LINE} to /cache\nGPU KV cache size: 28,000 tokens\n")
        else:
            log_path.write_text("GPU KV cache size: 44,624 tokens\n")
        yield "http://server"

    async def fake_send(url, model, prompts, starts, max_tokens, slo=None, *, params=None, **kwargs):
        if max_tokens == load.OUTPUT_LEN:  # the warmup asks for fewer
            lone_params.append(params)
        else:
            warmup_sizes.append(len(prompts))
        return Sent([Timing(0.05, lone / 1000, max_tokens)] * len(prompts), 1.0, 0.0)

    def fake_probe(url, model, pool, rate, slo, probe_s):
        if dies_at is not None and rate >= dies_at:
            with (tmp_path / "server.log").open("a") as f:
                f.write("(EngineCore pid=7) RuntimeError: CUDA error: an illegal memory access was encountered\n")
            raise httpx.ConnectError("[Errno 111] Connection refused")
        n, tpot = 60, after if batched else 8.9
        met = n if rate <= 20 else 0
        return {
            "rate": rate,
            "requests": n,
            "finished": n,
            "met": met,
            "share_met": met / n,
            "stopped_early": False,
            "ttft_p90_ms": 200.0,
            "tpot_p50_ms": tpot,
            "tpot_p90_ms": tpot,
            "client_cpu": 0.1,
            "kv_peak": 0.5,
            "preemptions": 0.0,
        }

    def fake_batch(url, model, pool):
        batched.append(True)
        return {"requests": 512, "tput_tok_s": 2900.0, "tpot_p50_ms": 75.0, "kv_peak": 0.9, "preemptions": 0.0}

    monkeypatch.setattr(load, "serving", fake_serving)
    monkeypatch.setattr(load, "send", fake_send)
    monkeypatch.setattr(load, "chat_probe", fake_probe)
    monkeypatch.setattr(load, "batch_run", fake_batch)
    monkeypatch.setattr(load, "wait_idle", lambda url: None)
    r = trial("screen", label, 0.007) | {"tpot_ms": 8.0}
    record = load.measure_config(r, ctx, [[1]], Slo(), 30, env={}, out_dir=tmp_path, timeout_s=600)
    assert lone_params == [GREEDY, GREEDY]
    assert warmup_sizes == list(load.WARMUP_BATCHES)
    return record


@pytest.mark.parametrize(
    ("after", "status", "error"),
    [
        (10.6, "ok", None),  # 1.19x the first probe
        (11.0, "degraded", "TPOT p50 at 2 req/s rose from 8.9 to 11.0 ms during the measurement"),
        (597.0, "degraded", "TPOT p50 at 2 req/s rose from 8.9 to 597.0 ms during the measurement"),
        (None, "degraded", "no request at 2 req/s finished without error after batch"),
    ],
)
def test_a_server_slower_after_batch_than_at_its_first_probe_is_degraded(
    tmp_path, monkeypatch, ctx, after, status, error
):
    record = measure(tmp_path, monkeypatch, ctx, after)
    assert (record["status"], record["error"]) == (status, error)
    assert (record["recheck"]["rate"], record["recheck"]["tpot_p50_ms"]) == (load.START_RATE, after)
    # a degraded record keeps its numbers for diagnosis
    assert record["capacity_rps"] == pytest.approx(20, rel=load.PRECISION)
    assert (record["batch"]["tput_tok_s"], record["kv_tokens"], record["lone_tpot_ms"]) == (2900, 44624, 8.5)


def test_a_server_that_stops_answering_once_it_served_is_recorded_with_the_cause_its_log_gives(
    tmp_path, monkeypatch, ctx
):
    record = measure(tmp_path, monkeypatch, ctx, 8.9, dies_at=8.0)
    assert record["status"] == "crash"
    assert record["error"] == (
        "the server stopped answering (ConnectError: [Errno 111] Connection refused); its log ends with "
        "RuntimeError: CUDA error: an illegal memory access was encountered"
    )
    assert record["wall_s"] >= 0 and "capacity_rps" not in record


@pytest.mark.parametrize(("compiles", "kv_tokens", "started"), [(0, 44624, 1), (1, 44624, 2), (2, 28000, 2)])
def test_a_server_whose_start_compiled_the_graph_is_measured_after_a_second_start(
    tmp_path, monkeypatch, ctx, compiles, kv_tokens, started
):
    starts = []
    record = measure(tmp_path, monkeypatch, ctx, 8.9, compiles=compiles, starts=starts)
    assert (record["status"], record["kv_tokens"], len(starts)) == ("ok", kv_tokens, started)
    assert len(set(map(tuple, starts))) == 1
    assert (tmp_path / "server-compile.log").exists() == (compiles > 0)


def test_a_server_that_decodes_a_lone_request_much_slower_than_offline_is_not_measured(tmp_path, monkeypatch, ctx):
    assert measure(tmp_path, monkeypatch, ctx, 8.9, lone=11.9)["status"] == "ok"  # 1.49x offline
    record = measure(tmp_path, monkeypatch, ctx, 8.9, lone=94.1)
    assert (record["status"], record["error"]) == (
        "degraded",
        "a lone request decoded at 94.1 ms per token, against 8.0 ms offline",
    )
    assert "capacity_rps" not in record and "batch" not in record


def test_speculative_configs_decode_lone_requests_at_a_speed_the_check_cannot_judge(tmp_path, monkeypatch, ctx):
    record = measure(tmp_path, monkeypatch, ctx, 8.9, lone=94.1, label="speculative=ngram")
    assert (record["status"], record["lone_tpot_ms"]) == ("ok", 94.1)


def test_a_degraded_config_gains_nothing_and_does_not_count_as_the_same_engine(tmp_path, monkeypatch, ctx):
    s, _ = session(tmp_path, monkeypatch, ctx, "s")
    base = loaded("baseline", "s", 10, 1000)
    slow = loaded("weights=fp8", "s", 30, 3000, backend="TRITON_ATTN") | {"status": "degraded"}
    (tmp_path / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (base, slow)))
    assert gains(slow, base) == (None, None)
    assert promising_under_load([base, slow], 0.1)["weights"] == {}
    assert s.same_engine(trial("combine", "weights=fp8 attention_backend=TRITON_ATTN", 0.01)) is None
