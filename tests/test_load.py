import asyncio
import json

import httpx
import pytest

from vllm_dyno import load
from vllm_dyno.load import (
    QUEUE,
    LoadSession,
    Slo,
    Timing,
    find_capacity,
    load_records,
    metric,
    poisson_starts,
    quantile,
    send,
    targets,
)
from vllm_dyno.runner import Run


def completion(tokens: int) -> bytes:
    """A streamed completion as vLLM sends it: one chunk per token, then usage, then [DONE]."""
    chunks: list[dict] = [{"choices": [{"index": 0, "text": "x"}]} for _ in range(tokens)]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": tokens}})
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def run_send(handler, prompts, slo=None, watch_server=False):
    starts = [0.0] * len(prompts)
    transport = httpx.MockTransport(handler)
    return asyncio.run(
        send("http://server", "org/base", prompts, starts, 8, slo, watch_server=watch_server, transport=transport)
    )


def test_requests_send_token_ids_each_with_its_own_cache_salt():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=completion(8))

    timings = run_send(handler, [[1, 2, 3], [4, 5, 6]]).timings
    assert [(t.tokens, t.error) for t in timings] == [(8, None), (8, None)]
    assert sorted(b["prompt"] for b in bodies) == [[1, 2, 3], [4, 5, 6]]
    assert all(b["ignore_eos"] and b["stream"] and b["temperature"] == 0 for b in bodies)
    assert all(b["repetition_penalty"] == 1 for b in bodies)
    assert bodies[0]["cache_salt"] != bodies[1]["cache_salt"]


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


def test_targets_default_to_confirmed_configs_and_take_named_ones_from_any_stage():
    records = [
        {"stage": "screen", "label": "baseline", "status": "ok"},
        {"stage": "screen", "label": "kv_cache_dtype=turboquant_k8v4", "status": "ok"},
        {"stage": "screen", "label": "kv_cache_dtype=fp8", "status": "skipped"},
        {"stage": "confirm", "label": "baseline", "status": "ok"},
    ]
    assert targets(records, []) == [records[3]]
    assert targets(records, ["kv_cache_dtype=turboquant_k8v4"]) == [records[1]]
    with pytest.raises(SystemExit):
        targets(records, ["kv_cache_dtype=fp8"])


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


def test_a_baseline_that_does_not_serve_stops_the_session(tmp_path, monkeypatch, ctx):
    s, _ = session(tmp_path, monkeypatch, ctx, "night")
    monkeypatch.setattr(load, "measure_config", lambda r, *a, **k: {"label": r["label"], "status": "oom", "error": "x"})
    with pytest.raises(SystemExit, match="the baseline did not serve"):
        s.baseline()
