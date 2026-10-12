import json
import sys
from collections.abc import Callable
from dataclasses import asdict

import httpx
import pytest

from vllm_dyno import load, planner
from vllm_dyno.fit import FitProposer
from vllm_dyno.load import LoadSession, Slo, load_records
from vllm_dyno.planner import AllProposer, Llm, LlmProposer, PlannerError, Proposal, Step, allowed, parse, plan
from vllm_dyno.report import load_recommendation
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


def test_the_planner_may_not_name_a_backend_for_a_turboquant_kv_cache(ctx):
    screen = [*SCREEN, trial("screen", "kv_cache_dtype=turboquant_k8v4", 0.0009)]
    labels = [c.label() for c in allowed(ctx, screen, max_kl=0.05)]
    assert "weights=fp8 kv_cache_dtype=turboquant_k8v4" in labels
    assert [label for label in labels if "turboquant" in label and "TRITON_ATTN" in label] == []


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
    message = {"role": "assistant", "content": content}
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage})


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
    reply = llm.complete([{"role": "user", "content": "hi"}])
    assert (reply.text, reply.usage["completion_tokens"]) == ("{}", 10)
    assert waits == [3.0, 30.0]
    assert str(sent[0].url) == "https://example.org/api/v1/chat/completions"
    assert sent[0].headers["authorization"] == "Bearer secret"
    assert json.loads(sent[0].content) == {"model": "org/model:free", "messages": [{"role": "user", "content": "hi"}]}


def test_an_endpoint_that_refuses_the_request_is_not_asked_again():
    sent: list[httpx.Request] = []
    llm = Llm("https://example.org/v1", "m", transport=endpoint([httpx.Response(401, json={})], sent))
    with pytest.raises(PlannerError, match="HTTP 401"):
        llm.complete([])
    assert len(sent) == 1 and "authorization" not in sent[0].headers


@pytest.mark.parametrize("body", [[{"error": {"code": 502}}], {"error": "upstream request failed"}])
def test_a_body_that_is_an_array_or_holds_a_string_error_fails_as_its_status_says(monkeypatch, body):
    sent: list[httpx.Request] = []
    llm = Llm("https://example.org/v1", "m", transport=endpoint([httpx.Response(200, json=body)], sent))
    with pytest.raises(PlannerError, match="HTTP 200"):
        llm.complete([])
    assert len(sent) == 1
    waits: list[float] = []
    monkeypatch.setattr(planner.time, "sleep", waits.append)
    responses = [httpx.Response(503, json=body), answer("{}")]
    assert Llm("https://example.org/v1", "m", transport=endpoint(responses, sent)).complete([]).text == "{}"
    assert waits == [15.0]


def triton_faster(label: str) -> tuple[float, float]:
    return (12.0 if "TRITON_ATTN" in label else 10.0), 1000.0


def setup(
    tmp_path,
    monkeypatch,
    ctx,
    replies: list[str],
    kl: dict[str, float] | None = None,
    slow: frozenset[str] = frozenset(),
    serve: Callable[[str], tuple[float, float]] = triton_faster,
):
    """A load session over SCREEN whose servers serve the req/s and tok/s serve gives for their config.

    By default 10 req/s and 1000 tok/s, more with TRITON_ATTN. A combination's quality trial measures KL 0.01
    unless kl says otherwise. The first server of each config in slow is degraded.
    """
    run = Run(tmp_path)
    for r in SCREEN:
        run.append(r)
    measured: list[str] = []

    def fake_measure(r, *args, **kwargs):
        measured.append(r["label"])
        chat, batch = serve(r["label"])
        return {
            "label": r["label"],
            **{k: r[k] for k in KNOBS},
            "status": "degraded" if r["label"] in slow and measured.count(r["label"]) == 1 else "ok",
            "error": None,
            "capacity_rps": chat,
            "chat": {"tpot_p90_ms": 40.0},
            "batch": {"tput_tok_s": batch, "kv_peak": 0.5, "preemptions": 0},
            "kv_tokens": 100_000,
            "backend": "TRITON_ATTN" if "TRITON_ATTN" in r["label"] else "FLASH_ATTN",
            "wall_s": 400.0,
        }

    class FakeSearcher:
        def __init__(self, *args):
            pass

        def stage(self, stage, cands, tier):
            out = []
            for c in cands:
                r = run.find(f"{stage}|{c.label()}")
                if r is None:
                    r = trial(stage, c.label(), (kl or {}).get(c.label(), 0.01))
                    run.append(r)  # as runner.run_trial does
                out.append(r)
            return out

    monkeypatch.setattr(load, "measure_config", fake_measure)
    monkeypatch.setattr(load, "wait_for_gpu", lambda free_mib: None)
    monkeypatch.setattr(planner, "Searcher", FakeSearcher)
    sent: list[httpx.Request] = []
    llm = Llm("https://example.org/v1", "org/model", transport=endpoint([answer(r) for r in replies], sent))
    return LoadSession(run, ctx, [[1]], Slo(), 30, {}, 600, 6000, id="s"), LlmProposer(llm), measured, sent


def pick(label: str | None, chat: float = 1.2) -> str:
    return json.dumps({"config": label, "hypothesis": f"try {label}", "expected_chat": chat, "expected_batch": 1.0})


def test_the_planner_loads_its_picks_after_the_baseline_and_combinations_after_their_quality_trial(
    tmp_path, monkeypatch, ctx
):
    replies = [
        pick("attention_backend=TRITON_ATTN"),
        pick("weights=fp8 kv_cache_dtype=int8_per_token_head"),  # over the KL limit offline
        pick("weights=fp8 attention_backend=TRITON_ATTN"),
        pick(None),  # refused: no loaded config has changed the KV cache dtype
        pick("kv_cache_dtype=int8_per_token_head"),
        pick(None),
    ]
    s, proposer, measured, sent = setup(
        tmp_path, monkeypatch, ctx, replies, kl={"weights=fp8 kv_cache_dtype=int8_per_token_head": 0.06}
    )
    plan(s, proposer, Goal(), "chat", max_servers=10, max_combos=8, trial_timeout_s=900)
    assert measured == [
        "baseline",
        "attention_backend=TRITON_ATTN",
        "weights=fp8 attention_backend=TRITON_ATTN",
        "kv_cache_dtype=int8_per_token_head",
    ]
    journal = [json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()]
    assert [(j["config"], j["server"]) for j in journal] == [
        ("attention_backend=TRITON_ATTN", True),
        ("weights=fp8 kv_cache_dtype=int8_per_token_head", False),
        ("weights=fp8 attention_backend=TRITON_ATTN", True),
        ("kv_cache_dtype=int8_per_token_head", True),
        (None, False),
    ]
    assert {j["model"] for j in journal} == {"org/model"}
    assert journal[0]["result"] == "chat 12.00 req/s (1.20x), batch 1000 tok/s (1.00x), KL 0.0006"
    assert journal[1]["result"] == "KL 0.0600 offline, over the limit; no server started"
    assert journal[3]["tokens"] == {"prompt_tokens": 200, "completion_tokens": 20}  # the refused stop counts
    first = json.loads(sent[0].content)["messages"][1]["content"]
    assert "Settings whose effect under load is not known yet: weights, kv_cache_dtype, attention_backend." in first
    refused = json.loads(sent[4].content)["messages"]
    assert refused[-1]["content"].startswith(
        "stopping is allowed once each setting's effect under load is known; not yet: kv_cache_dtype."
    )
    last = json.loads(sent[-1].content)["messages"][1]["content"]
    assert "attention_backend=TRITON_ATTN | 1.20x, 1.00x | chat 12.00 req/s" in last
    assert "Servers left: 7 of 10, about 7 minutes each." in last
    assert "Settings whose effect under load is not known yet" not in last
    allowed_now = last.split("Allowed configs, not loaded yet:\n")[1].split("\n\n")[0].splitlines()
    assert "weights=fp8 kv_cache_dtype=int8_per_token_head" not in allowed_now
    assert "weights=fp8 attention_backend=TRITON_ATTN" not in allowed_now
    assert len(list((tmp_path / "plan").iterdir())) == 5


def test_a_setting_is_known_once_two_configs_that_served_differ_in_it_alone(tmp_path, monkeypatch, ctx):
    replies = [
        pick("weights=fp8 kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN"),
        pick(None),  # refused: the three settings changed at once
        pick("weights=fp8 kv_cache_dtype=int8_per_token_head"),
        pick(None),  # refused: only the attention backend is known
        pick("weights=fp8"),
        pick(None),
    ]
    s, proposer, measured, sent = setup(tmp_path, monkeypatch, ctx, replies)
    plan(s, proposer, Goal(), "chat", max_servers=10, max_combos=8, trial_timeout_s=900)
    assert measured == [
        "baseline",
        "weights=fp8 kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN",
        "weights=fp8 kv_cache_dtype=int8_per_token_head",
        "weights=fp8",
    ]
    refusals = [json.loads(sent[i].content)["messages"][-1]["content"] for i in (2, 4)]
    assert [r.split(". Answer")[0].split("not yet: ")[1] for r in refusals] == [
        "weights, kv_cache_dtype, attention_backend",
        "weights, kv_cache_dtype",
    ]


def test_the_planner_stops_at_its_server_budget(tmp_path, monkeypatch, ctx):
    replies = [pick("attention_backend=TRITON_ATTN"), pick("weights=fp8")]
    s, proposer, measured, _ = setup(tmp_path, monkeypatch, ctx, replies)
    plan(s, proposer, Goal(), "chat", max_servers=1, max_combos=8, trial_timeout_s=900)
    assert measured == ["baseline", "attention_backend=TRITON_ATTN"]


def test_a_server_started_again_after_it_degraded_counts_against_the_budget(tmp_path, monkeypatch, ctx):
    replies = [pick("attention_backend=TRITON_ATTN"), pick("weights=fp8")]
    s, proposer, measured, _ = setup(
        tmp_path, monkeypatch, ctx, replies, slow=frozenset({"attention_backend=TRITON_ATTN"})
    )
    plan(s, proposer, Goal(), "chat", max_servers=2, max_combos=8, trial_timeout_s=900)
    assert measured == ["baseline", "attention_backend=TRITON_ATTN", "attention_backend=TRITON_ATTN"]


def test_the_rules_finish_the_session_when_the_model_names_no_allowed_config_twice(tmp_path, monkeypatch, ctx):
    replies = [pick("weights=org/base-AWQ"), "fp8, I think"]
    s, proposer, _, sent = setup(tmp_path, monkeypatch, ctx, replies)
    searched = []
    monkeypatch.setattr(LoadSession, "search", lambda self, goal, max_combos, timeout: searched.append(max_combos))
    plan(s, proposer, Goal(), "chat", max_servers=10, max_combos=8, trial_timeout_s=900)
    assert searched == [8]
    retry = json.loads(sent[1].content)["messages"]
    assert retry[-1]["content"].startswith("'weights=org/base-AWQ' is not in the allowed list.")
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    step = json.loads(next((tmp_path / "plan").iterdir()).read_text())
    assert "answered twice with no allowed config or stop" in step["error"]


class Scripted:
    """A proposer that answers with the given proposals in turn, keeping each step it was shown."""

    name = "scripted"

    def __init__(self, *proposals: Proposal):
        self.proposals = list(proposals)
        self.steps: list[Step] = []

    def propose(self, step: Step) -> Proposal:
        self.steps.append(step)
        return self.proposals.pop(0)


@pytest.mark.parametrize(
    "bad",
    [
        Proposal(None, "done"),  # the weights and the KV cache dtype are not known yet
        Proposal(cand("weights=org/base-AWQ"), "over the KL limit offline"),
    ],
)
def test_the_rules_finish_the_session_when_any_proposer_stops_too_early_or_picks_outside_the_list(
    tmp_path, monkeypatch, ctx, bad
):
    s, _, measured, _ = setup(tmp_path, monkeypatch, ctx, [])
    searched = []
    monkeypatch.setattr(LoadSession, "search", lambda self, goal, max_combos, timeout: searched.append(max_combos))
    proposer = Scripted(Proposal(cand("attention_backend=TRITON_ATTN"), "h", 1.2), bad)
    plan(s, proposer, Goal(), "chat", max_servers=10, max_combos=8, trial_timeout_s=900)
    assert (measured, searched) == (["baseline", "attention_backend=TRITON_ATTN"], [8])
    first, second = proposer.steps
    assert first.untried == ["weights", "kv_cache_dtype", "attention_backend"]
    assert second.untried == ["weights", "kv_cache_dtype"]
    assert (first.servers_left, second.servers_left, second.max_servers) == (10, 9, 10)
    assert [j["config"] for j in second.journal] == ["attention_backend=TRITON_ATTN"]
    assert (first.step_file, second.step_file) == (tmp_path / "plan" / "s-01.json", tmp_path / "plan" / "s-02.json")
    assert "attention_backend=TRITON_ATTN" in [c.label() for c in first.options]
    assert "attention_backend=TRITON_ATTN" not in [c.label() for c in second.options]
    journal = [json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()]
    assert [(j["model"], j["config"], j["expected_chat"], j["tokens"]) for j in journal] == [
        ("scripted", "attention_backend=TRITON_ATTN", 1.2, None)
    ]


def test_the_all_planner_loads_every_allowed_config_in_order(tmp_path, monkeypatch, ctx):
    over = "weights=fp8 kv_cache_dtype=int8_per_token_head"
    s, _, measured, _ = setup(tmp_path, monkeypatch, ctx, [], kl={over: 0.06})
    plan(s, AllProposer(), Goal(), "chat", max_servers=sys.maxsize, max_combos=8, trial_timeout_s=900)
    assert measured == [
        "baseline",
        "attention_backend=TRITON_ATTN",
        "kv_cache_dtype=int8_per_token_head",
        "kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN",
        "weights=fp8",
        "weights=fp8 attention_backend=TRITON_ATTN",
        "weights=fp8 kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN",
    ]
    journal = [json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()]
    assert [j["config"] for j in journal] == [c.label() for c in allowed(ctx, SCREEN, max_kl=0.05)]
    assert next(j for j in journal if j["config"] == over)["server"] is False
    assert {(j["model"], j["tokens"], j["expected_chat"]) for j in journal} == {("all", None, None)}


def test_the_all_planner_keeps_to_a_server_budget_it_is_given(tmp_path, monkeypatch, ctx):
    s, _, measured, _ = setup(tmp_path, monkeypatch, ctx, [])
    plan(s, AllProposer(), Goal(), "chat", max_servers=2, max_combos=8, trial_timeout_s=900)
    assert measured == ["baseline", "attention_backend=TRITON_ATTN", "kv_cache_dtype=int8_per_token_head"]


PLANTED = {  # chat and batch gains of each value; they multiply, and weights=fp8 with TRITON_ATTN serves best
    "weights=fp8": (1.15, 1.25),
    "kv_cache_dtype=int8_per_token_head": (0.9, 1.1),
    "attention_backend=TRITON_ATTN": (1.3, 1.0),
}


def planted(label: str) -> tuple[float, float]:
    chat, batch = 10.0, 1000.0
    for value in label.split():
        c, b = PLANTED.get(value, (1.0, 1.0))
        chat, batch = chat * c, batch * b
    return chat, batch


def test_the_fit_planner_finds_the_best_config_of_a_space_whose_gains_multiply(tmp_path, monkeypatch, ctx):
    s, _, measured, _ = setup(tmp_path, monkeypatch, ctx, [], serve=planted)
    monkeypatch.setattr(LoadSession, "search", lambda *args: pytest.fail("the rules finished the session"))
    plan(s, FitProposer(), Goal(), "chat", max_servers=5, max_combos=8, trial_timeout_s=900)
    journal = [json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()]
    assert "weights=fp8 attention_backend=TRITON_ATTN" in measured
    assert journal[-1]["config"] is None and len(measured) - 1 < 5  # it stopped on its own
    picked = load_recommendation(load_records(tmp_path), s.run.records, Goal(), "chat")
    assert picked is not None and picked[0]["label"] == "weights=fp8 attention_backend=TRITON_ATTN"
    assert {j["model"] for j in journal} == {"fit"}
    assert all(j["expected_chat"] and j["expected_batch"] for j in journal[:-1])
    assert len(list((tmp_path / "plan").iterdir())) == len(journal)


def test_a_stop_counts_only_once_no_setting_is_left_untried():
    options = [cand("weights=fp8")]
    with pytest.raises(ValueError, match="not yet: weights, speculative"):
        parse('{"config": null, "hypothesis": "h"}', options, ["weights", "speculative"])
    assert parse('{"config": null, "hypothesis": "h"}', options, []).config is None


def test_the_key_is_taken_out_of_the_environment_and_never_shown():
    env = {"OPENROUTER_API_KEY": "secret", "HF_HOME": "/hf"}
    assert planner.take_key(env, "OPENROUTER_API_KEY") == "secret"
    assert env == {"HF_HOME": "/hf"}
    assert planner.take_key({"K": ""}, "K") is None
    assert "secret" not in repr(Llm("https://example.org/v1", "m", "secret"))
