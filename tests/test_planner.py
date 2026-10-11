import json
from dataclasses import asdict

import httpx
import pytest

from vllm_dyno import load, planner
from vllm_dyno.load import LoadSession, Slo
from vllm_dyno.planner import Llm, PlannerError, allowed, parse, plan
from vllm_dyno.runner import Run
from vllm_dyno.search import Goal
from vllm_dyno.space import KNOBS, Candidate


def cand(label: str) -> Candidate:
    return Candidate() if label == "baseline" else Candidate(**dict(kv.split("=") for kv in label.split()))


def trial(stage: str, label: str, kl: float, status: str = "ok") -> dict:
    return {
        "key": f"{stage}|{label}",
        "stage": stage,
        "label": label,
        "status": status,
        "kl": kl,
        "tput_tok_s": 3000.0,
        "kv_tokens": 100_000,
        **asdict(cand(label)),
    }


SCREEN = [
    trial("screen", "baseline", 0.0),
    trial("screen", "weights=fp8", 0.007),
    trial("screen", "weights=org/base-AWQ", 0.07),  # over the quality limit
    trial("screen", "kv_cache_dtype=fp8", 0.0, status="skipped"),
    trial("screen", "kv_cache_dtype=int8_per_token_head", 0.013),
    trial("screen", "attention_backend=TRITON_ATTN", 0.0006),
]


def test_the_planner_may_pick_combinations_of_values_that_each_passed_quality(ctx):
    labels = [c.label() for c in allowed(ctx, SCREEN, max_kl=0.05)]
    assert labels == [
        "attention_backend=TRITON_ATTN",
        "kv_cache_dtype=int8_per_token_head",
        "kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN",
        "weights=fp8",
        "weights=fp8 attention_backend=TRITON_ATTN",
        "weights=fp8 kv_cache_dtype=int8_per_token_head",
        "weights=fp8 kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN",
    ]


def test_a_reply_names_an_allowed_config_in_any_order_or_stops():
    options = [cand("weights=fp8 attention_backend=TRITON_ATTN"), cand("weights=fp8")]
    reply = (
        '```json\n{"config": "attention_backend=TRITON_ATTN weights=fp8", "hypothesis": "h", "expected_chat": 1.2}\n```'
    )
    p = parse(reply, options)
    assert (p.config, p.hypothesis, p.expected_chat, p.expected_batch) == (options[0], "h", 1.2, None)
    assert parse('{"config": null, "hypothesis": "done"}', options).config is None
    with pytest.raises(ValueError, match="not in the allowed list"):
        parse('{"config": "baseline"}', options)
    with pytest.raises(ValueError, match="no JSON object"):
        parse("I would try fp8.", options)


def endpoint(responses: list[httpx.Response], sent: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return responses.pop(0)

    return httpx.MockTransport(handler)


def answer(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_the_endpoint_is_asked_again_while_it_is_rate_limited(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(planner.time, "sleep", waits.append)
    sent: list[httpx.Request] = []
    responses = [
        httpx.Response(429, json={"error": {"code": 429, "message": "slow down"}}, headers={"retry-after": "3"}),
        httpx.Response(200, json={"error": {"code": 502, "message": "provider down"}}),
        answer("{}"),
    ]
    llm = Llm("https://example.org/api/v1", "org/model:free", "secret", transport=endpoint(responses, sent))
    assert llm.complete([{"role": "user", "content": "hi"}]) == "{}"
    assert waits == [3.0, 30.0]
    assert str(sent[0].url) == "https://example.org/api/v1/chat/completions"
    assert sent[0].headers["authorization"] == "Bearer secret"
    assert json.loads(sent[0].content) == {
        "model": "org/model:free",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    }


def test_an_endpoint_that_refuses_the_request_is_not_asked_again():
    sent: list[httpx.Request] = []
    llm = Llm("https://example.org/v1", "m", transport=endpoint([httpx.Response(401, json={})], sent))
    with pytest.raises(PlannerError, match="HTTP 401"):
        llm.complete([])
    assert len(sent) == 1 and "authorization" not in sent[0].headers


def setup(tmp_path, monkeypatch, ctx, replies: list[str], kl: dict[str, float] | None = None):
    """A load session over SCREEN whose servers serve 10 req/s and 1000 tok/s, more with TRITON_ATTN."""
    run = Run(tmp_path)
    for r in SCREEN:
        run.append(r)
    measured: list[str] = []

    def fake_measure(r, *args, **kwargs):
        measured.append(r["label"])
        triton = "TRITON_ATTN" in r["label"]
        return {
            "label": r["label"],
            **{k: r[k] for k in KNOBS},
            "status": "ok",
            "capacity_rps": 12.0 if triton else 10.0,
            "chat": {"tpot_p90_ms": 40.0},
            "batch": {"tput_tok_s": 1000.0, "kv_peak": 0.5, "preemptions": 0},
            "kv_tokens": 100_000,
            "backend": "TRITON_ATTN" if triton else "FLASH_ATTN",
            "wall_s": 400.0,
        }

    class FakeSearcher:
        def __init__(self, *args):
            pass

        def stage(self, stage, cands, tier):
            return [trial(stage, c.label(), (kl or {}).get(c.label(), 0.01)) for c in cands]

    monkeypatch.setattr(load, "measure_config", fake_measure)
    monkeypatch.setattr(load, "wait_for_gpu", lambda free_mib: None)
    monkeypatch.setattr(planner, "Searcher", FakeSearcher)
    sent: list[httpx.Request] = []
    llm = Llm("https://example.org/v1", "org/model", transport=endpoint([answer(r) for r in replies], sent))
    return LoadSession(run, ctx, [[1]], Slo(), 30, {}, 600, 6000, id="s"), llm, measured, sent


def pick(label: str | None, chat: float = 1.2) -> str:
    return json.dumps({"config": label, "hypothesis": f"try {label}", "expected_chat": chat, "expected_batch": 1.0})


def test_the_planner_loads_its_picks_after_the_baseline_and_combinations_after_their_quality_trial(
    tmp_path, monkeypatch, ctx
):
    replies = [
        pick("attention_backend=TRITON_ATTN"),
        pick("weights=fp8 kv_cache_dtype=int8_per_token_head"),  # over the KL limit offline
        pick("weights=fp8 attention_backend=TRITON_ATTN"),
        pick(None),
    ]
    s, llm, measured, sent = setup(
        tmp_path, monkeypatch, ctx, replies, kl={"weights=fp8 kv_cache_dtype=int8_per_token_head": 0.06}
    )
    plan(s, llm, Goal(), "chat", max_servers=10, max_combos=8, trial_timeout_s=900)
    assert measured == ["baseline", "attention_backend=TRITON_ATTN", "weights=fp8 attention_backend=TRITON_ATTN"]
    journal = [json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()]
    assert [(j["config"], j["server"]) for j in journal] == [
        ("attention_backend=TRITON_ATTN", True),
        ("weights=fp8 kv_cache_dtype=int8_per_token_head", False),
        ("weights=fp8 attention_backend=TRITON_ATTN", True),
        (None, False),
    ]
    assert journal[0]["result"] == "chat 12.00 req/s (1.20x), batch 1000 tok/s (1.00x), KL 0.0006"
    assert journal[1]["result"] == "KL 0.0600 offline, over the limit; no server started"
    last = json.loads(sent[-1].content)["messages"][1]["content"]
    assert "attention_backend=TRITON_ATTN | 1.20x, 1.00x | chat 12.00 req/s" in last
    assert "Servers left: 8 of 10, about 7 minutes each." in last
    allowed_now = last.split("Allowed configs, not loaded yet:\n")[1].split("\n\n")[0].splitlines()
    assert "weights=fp8 kv_cache_dtype=int8_per_token_head" not in allowed_now
    assert "weights=fp8 attention_backend=TRITON_ATTN" not in allowed_now
    assert len(list((tmp_path / "plan").iterdir())) == 4


def test_the_planner_stops_at_its_server_budget(tmp_path, monkeypatch, ctx):
    replies = [pick("attention_backend=TRITON_ATTN"), pick("weights=fp8")]
    s, llm, measured, _ = setup(tmp_path, monkeypatch, ctx, replies)
    plan(s, llm, Goal(), "chat", max_servers=1, max_combos=8, trial_timeout_s=900)
    assert measured == ["baseline", "attention_backend=TRITON_ATTN"]


def test_the_rules_finish_the_session_when_the_model_names_no_allowed_config_twice(tmp_path, monkeypatch, ctx):
    replies = [pick("weights=org/base-AWQ"), "fp8, I think"]
    s, llm, _, sent = setup(tmp_path, monkeypatch, ctx, replies)
    searched = []
    monkeypatch.setattr(LoadSession, "search", lambda self, goal, max_combos, timeout: searched.append(max_combos))
    plan(s, llm, Goal(), "chat", max_servers=10, max_combos=8, trial_timeout_s=900)
    assert searched == [8]
    retry = json.loads(sent[1].content)["messages"]
    assert retry[-1]["content"].startswith("'weights=org/base-AWQ' is not in the allowed list.")
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    step = json.loads(next((tmp_path / "plan").iterdir()).read_text())
    assert "answered twice without an allowed config" in step["error"]
