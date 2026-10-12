"""report.md from run.json, trials.jsonl, load.jsonl and plan.jsonl."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .load import (
    BATCH_REQUESTS,
    LOAD_MIN_GAIN,
    MIN_RATE,
    OUTPUT_LEN,
    PRECISION,
    PROMPT_LEN,
    gains,
    session_baseline,
)
from .planner import TOKENS
from .probe import serve_env
from .search import Goal, candidate_of
from .space import Candidate, Context, serve_command


def recommendation(records: list[dict], goal: Goal) -> dict | None:
    confirmed = [r for r in records if r["stage"] == "confirm" and goal.passes(r)]
    return max(confirmed, key=goal.score) if confirmed else None


def near_ties(records: list[dict], best: dict, goal: Goal) -> list[tuple[dict, float, float]]:
    """Confirmed configs within the goal that trail the winner by less than the noise, with the gap and the noise.

    The noise is the smallest gain the search counts, or either config's throughput CV across repeats if larger.
    TPOT has no measured spread, so for the latency goal only the smallest counted gain applies.
    """
    out = []
    for r in records:
        if r["stage"] != "confirm" or r["key"] == best["key"] or not goal.passes(r):
            continue
        cvs = [x.get("tput_cv") or 0 for x in (best, r)] if goal.target == "throughput" else []
        gap, noise = goal.gain(best, r) - 1, max([goal.min_gain, *cvs])
        if gap < noise:
            out.append((r, gap, noise))
    return sorted(out, key=lambda t: t[1])


def pareto_keys(records: list[dict]) -> set[str]:
    """Trials no other trial beats on throughput, TPOT and KL at once."""
    ok = [r for r in records if r["status"] == "ok"]

    def dominates(a: dict, b: dict) -> bool:
        no_worse = a["tput_tok_s"] >= b["tput_tok_s"] and a["tpot_ms"] <= b["tpot_ms"] and a["kl"] <= b["kl"]
        better = a["tput_tok_s"] > b["tput_tok_s"] or a["tpot_ms"] < b["tpot_ms"] or a["kl"] < b["kl"]
        return no_worse and better

    return {r["key"] for r in ok if not any(dominates(o, r) for o in ok)}


def _f(value: float | None, spec: str) -> str:
    return "" if value is None else format(value, spec)


def _ratio(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}x"


def _cost(r: dict, gpu_price: float | None) -> str:
    if not gpu_price or r.get("status") != "ok":
        return ""
    return f"{gpu_price / (r['tput_tok_s'] * 3600) * 1e6:.3f}"


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


_LOAD_TARGETS = {"chat": "highest chat capacity under the SLO", "batch": "highest batch throughput"}


def load_scores(
    load: Sequence[dict], records: list[dict], goal: Goal
) -> dict[str, tuple[dict, tuple[float, float, float]]]:
    """Per label, the load record load_recommendation weighs, with its offline KL, and its chat gain, batch gain and KL.

    A label's last record that has both gains counts. Only configs within the KL limit do.
    """
    scored: dict[str, tuple[dict, tuple[float, float, float]]] = {}
    for r in load:
        q = quality(records, r["label"])
        chat, batch = gains(r, session_baseline(list(load), r))
        if q is None or q["kl"] > goal.max_kl or chat is None or batch is None:
            continue
        scored[r["label"]] = (r | {"kl": q["kl"]}, (chat, batch, q["kl"]))
    return scored


def load_winner(scores: dict[str, tuple[float, float, float]], target: str) -> tuple[str, list[str]]:
    """The label to serve for `target` given each label's chat gain, batch gain and KL, and the labels tied with it.

    Labels within LOAD_MIN_GAIN of the best on the target are tied with it. Among them the other profile
    decides the same way, and then the lowest KL, and then the higher gain on the target.
    """
    first = 0 if target == "chat" else 1

    def tied(labels: list[str], i: int) -> list[str]:
        top = max(scores[k][i] for k in labels)
        return [k for k in labels if scores[k][i] * (1 + LOAD_MIN_GAIN) >= top]

    on_target = tied(list(scores), first)
    best = min(tied(on_target, 1 - first), key=lambda k: (scores[k][2], -scores[k][first]))
    return best, [k for k in on_target if k != best]


def load_recommendation(
    load: Sequence[dict], records: list[dict], goal: Goal, target: str
) -> tuple[dict, list[dict]] | None:
    """The load record to serve for `target` ("chat" or "batch"), and those the measurements did not separate from it.

    Configs within LOAD_MIN_GAIN of the best on the target are tied with it. Among them the other profile
    decides the same way, and then the lowest KL. Only configs within the KL limit count.
    """
    scored = load_scores(load, records, goal)
    if not scored:
        return None
    best, ties = load_winner({label: s for label, (_, s) in scored.items()}, target)
    return scored[best][0], [scored[label][0] for label in ties]


def quality(records: list[dict], label: str) -> dict | None:
    """The offline trial of label that measured its KL on the most tokens."""
    ok = [r for r in records if r["label"] == label and r["status"] == "ok"]
    confirmed = [r for r in ok if r["stage"] == "confirm"]
    return next(iter(confirmed or ok), None)


def _sampling_note(r: dict) -> str:
    if "sampling" not in r:  # load records from before requests left sampling to the server
        return "Requests decoded greedily without penalties."
    if r["sampling"] is None:
        return "Requests left sampling to the server, which used vLLM's defaults."
    return f"Requests left sampling to the server, which applied the model's generation_config: `{r['sampling']}`."


def _chat_batch(r: dict, load: list[dict]) -> str:
    chat, batch = gains(r, session_baseline(load, r))
    return (
        f"chat {r['capacity_rps']:.2f} req/s ({_ratio(chat)} the baseline), "
        f"batch {r['batch']['tput_tok_s']:,.0f} tok/s ({_ratio(batch)})"
    )


def _load_lines(best: dict, load: list[dict], target: str) -> list[str]:
    base = session_baseline(load, best) or best
    return [
        (
            f"`{best['label']}` for {target} traffic: {_chat_batch(best, load)}, against the baseline loaded "
            f"in the same session ({base['capacity_rps']:.2f} req/s, {base['batch']['tput_tok_s']:,.0f} tok/s). "
            f"KL {best['kl']:.4f}, KV cache {_f(best.get('kv_tokens'), ',')} tokens as the server reports them."
        ),
        "",
    ]


def _load_tie_lines(ties: list[dict], load: list[dict], target: str) -> list[str]:
    if not ties:
        return []
    metric, other = ("chat capacity", "batch throughput") if target == "chat" else ("batch throughput", "chat capacity")
    return [
        "",
        (
            f"These configs are within {LOAD_MIN_GAIN:.0%} of it on {metric}, which the measurements do not "
            f"separate; among them {other} within the same margin decided, then the lowest KL:"
        ),
        "",
        *(f"- `{r['label']}`: {_chat_batch(r, load)}, KL {r['kl']:.4f}." for r in ties),
    ]


def _offline_lines(best: dict, b: dict) -> list[str]:
    return [
        (
            f"`{best['label']}`: {best['tput_tok_s']:,.0f} tok/s, "
            f"{best['tput_tok_s'] / b['tput_tok_s']:.2f}x the baseline ({b['tput_tok_s']:,.0f}). "
            f"TPOT {best['tpot_ms']:.1f} ms (baseline {b['tpot_ms']:.1f}), "
            f"TTFT {best['ttft_ms']:.0f} ms (baseline {b['ttft_ms']:.0f}), "
            f"KV cache {best['kv_tokens']:,} tokens (baseline {b['kv_tokens']:,}). "
            f"KL {best['kl']:.4f}, top-1 agreement {best['top1_agree']:.3f}, "
            f"perplexity {best['ppl']:.3f} (baseline {b['ppl']:.3f})."
        ),
        "",
        (
            f"The baseline rerun scored KL {b['kl']:.4f} and top-1 agreement {b['top1_agree']:.3f} against its own "
            f"reference, and its throughput varied {_f((b.get('tput_cv') or 0) * 100, '.1f')}% across repeats. "
            "Differences smaller than these are noise."
        ),
    ]


def _offline_tie_lines(records: list[dict], best: dict, goal: Goal) -> list[str]:
    ties = near_ties(records, best, goal)
    if not ties:
        return []
    worse = "lower throughput" if goal.target == "throughput" else "higher TPOT"
    return [
        "",
        (
            f"These configs trail `{best['label']}` by less than the noise, so the measurements do not "
            "separate them from it; choose among them on quality and memory:"
        ),
        "",
        *(
            (
                f"- `{r['label']}`: {gap * 100:.1f}% {worse}, noise {noise * 100:.1f}%. "
                f"KL {r['kl']:.4f} (winner {best['kl']:.4f}), TPOT {r['tpot_ms']:.1f} ms "
                f"(winner {best['tpot_ms']:.1f}), KV cache {r['kv_tokens']:,} tokens "
                f"(winner {best['kv_tokens']:,})."
            )
            for r, gap, noise in ties
        ),
    ]


def _commands(cand: Candidate, ctx: Context, run_dir: Path) -> list[str]:
    env_vars = [f"{k}={v}" for k, v in serve_env(ctx.host, ctx.util).items()]
    lines = [
        "Start it with the environment the trials ran in:",
        "",
        "```",
        f"dyno serve {run_dir}",
        "```",
        "",
        "or directly:",
        "",
        "```",
        " ".join([*env_vars, serve_command(cand, ctx)]),
        "```",
    ]
    if ctx.host.get("unsearchable_path"):
        lines += [
            "",
            (
                f"The trials ran with {ctx.host['unsearchable_path']} PATH entries removed that Linux cannot "
                "search (WSL adds Windows directories). With them in PATH, torch fails while probing for nvcc; "
                "`dyno serve` removes them."
            ),
        ]
    return lines


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _expected(p: dict) -> str:
    if not p["config"] or (p["expected_chat"] is None and p["expected_batch"] is None):
        return ""
    return f"{_ratio(p['expected_chat'])}, {_ratio(p['expected_batch'])}"


def _plan_lines(picks: Sequence[dict]) -> list[str]:
    models = ", ".join(f"`{m}`" for m in dict.fromkeys(p["model"] for p in picks))
    expects = any(_expected(p) for p in picks)
    expected = ", with the gains it expected before they ran" if expects else ""
    lines = [
        "",
        "## Planner",
        "",
        (
            f"{models} picked the configs to load after the baseline, in this order{expected}. "
            "A combination it picked ran offline for quality before a server started."
        ),
        "",
    ]
    rows = [
        [
            str(i),
            f"`{p['config']}`" if p["config"] else "stop",
            *([_expected(p)] if expects else []),
            _cell(p["result"]),
            _cell(p["hypothesis"]),
        ]
        for i, p in enumerate(picks, 1)
    ]
    header = ["#", "config", *(["expected chat, batch"] if expects else []), "measured", "hypothesis"]
    lines += markdown_table(header, rows)
    counted = [p["tokens"] for p in picks if p.get("tokens")]
    if counted:
        prompt, completion = (sum(t[k] for t in counted) for k in TOKENS)
        lines += ["", f"The endpoint counted {prompt:,} prompt tokens and {completion:,} completion tokens."]
    return lines


def render(
    meta: dict,
    records: list[dict],
    run_dir: Path,
    load: Sequence[dict] = (),
    load_goal: str = "chat",
    picks: Sequence[dict] = (),
) -> str:
    ctx = Context.from_json(meta["context"])
    goal = Goal(**meta["goal"])
    price = meta.get("gpu_price")
    gpu = ctx.gpu
    base_screen = next((r for r in records if r["stage"] == "screen" and r["label"] == "baseline"), None)
    base_confirm = next((r for r in records if r["stage"] == "confirm" and r["label"] == "baseline"), None)
    best = recommendation(records, goal)
    under_load = load_recommendation(load, records, goal, load_goal)
    picked, load_ties = under_load if under_load else (None, [])

    target = "highest throughput" if goal.target == "throughput" else "lowest time per output token"
    limits = [f"KL ≤ {goal.max_kl}"]
    if goal.max_ttft_ms is not None:
        limits.append(f"TTFT ≤ {goal.max_ttft_ms:.0f} ms")
    if goal.max_tpot_ms is not None:
        limits.append(f"TPOT ≤ {goal.max_tpot_ms:.1f} ms")
    lines = [
        "# vllm-dyno report",
        "",
        (
            f"`{ctx.model}` on {gpu.name} (compute capability {gpu.compute_cap}, {gpu.total_mib:,} MiB), "
            f"vLLM {ctx.host.get('vllm')}, `gpu_memory_utilization` {ctx.util}, `max_model_len` {ctx.max_model_len}. "
            f"Started {meta['created']}."
        ),
        "",
        (
            f"Goal: {_LOAD_TARGETS[load_goal]} with KL ≤ {goal.max_kl}; without load, {target} with {', '.join(limits)}."
            if picked
            else f"Goal: {target} with {', '.join(limits)}."
        )
        + " KL is measured against the baseline's next-token distribution.",
        "",
        "## Recommendation",
        "",
    ]
    if picked:
        lines += _load_lines(picked, list(load), load_goal)
        lines += _commands(candidate_of(picked), ctx, run_dir)
        lines += _load_tie_lines(load_ties, list(load), load_goal)
        if best and base_confirm:
            lines += ["", "### Without load", "", *_offline_lines(best, base_confirm)]
            lines += _offline_tie_lines(records, best, goal)
    elif best and base_confirm:
        offline = _offline_lines(best, base_confirm)
        lines += [*offline[:2], *_commands(candidate_of(best), ctx, run_dir), "", *offline[2:]]
        lines += _offline_tie_lines(records, best, goal)
    else:
        lines.append("No config finished the confirm stage within the goal.")

    confirm = [r for r in records if r["stage"] == "confirm" and r["status"] == "ok"]
    if confirm:
        lines += ["", "## Confirmed", ""]
        header = ["config", "tok/s", "CV %", "TPOT ms", "TTFT ms", "KV tokens", "KL", "top-1", "ppl", "$/1M tok"]
        rows = [
            [
                f"`{r['label']}`",
                f"{r['tput_tok_s']:,.0f}",
                _f((r.get("tput_cv") or 0) * 100, ".1f"),
                f"{r['tpot_ms']:.1f}",
                f"{r['ttft_ms']:.0f}",
                f"{r['kv_tokens']:,}",
                f"{r['kl']:.4f}",
                f"{r['top1_agree']:.3f}",
                f"{r['ppl']:.3f}",
                _cost(r, price),
            ]
            for r in sorted(confirm, key=goal.score, reverse=True)
        ]
        lines += markdown_table(header, rows)

    if load:
        slo = load[0]["slo"]
        lines += [
            "",
            "## Under load",
            "",
            (
                f"Through `vllm serve` with its serving defaults. Chat: {PROMPT_LEN}-token prompts with "
                f"{OUTPUT_LEN}-token answers arriving as a Poisson process; capacity is the highest rate at which "
                f"{slo['share']:.0%} of requests had TTFT ≤ {slo['ttft_ms']:.0f} ms and TPOT ≤ {slo['tpot_ms']:.0f} ms, "
                f"found to within {PRECISION:.0%}. Batch: {BATCH_REQUESTS} such requests at once. "
                f"{_sampling_note(load[0])} "
                "Attention is the backend the server picked, and KV tokens the cache size it reports. Ratios "
                "are against the baseline measured at the start of the same `dyno load`."
            ),
            "",
        ]
        rows = []
        for r in load:
            q = quality(records, r["label"])
            kl = _f(q["kl"] if q else None, ".4f")
            if r["status"] != "ok":
                rows.append([f"`{r['label']}`", "", r["status"], "", "", "", "", "", "", kl])
                continue
            cap, chat, batch = r["capacity_rps"], r.get("chat") or {}, r["batch"]
            chat_x, batch_x = gains(r, session_baseline(list(load), r))
            rows.append(
                [
                    f"`{r['label']}`",
                    r.get("backend") or "",
                    f"< {MIN_RATE:g}" if cap is None else f"≥ {cap:g}" if r["capped"] else f"{cap:.2f}",
                    _ratio(chat_x),
                    _f(chat.get("ttft_p90_ms"), ".0f"),
                    _f(chat.get("tpot_p90_ms"), ".1f"),
                    f"{batch['tput_tok_s']:,.0f}",
                    _ratio(batch_x),
                    _f(r.get("kv_tokens"), ","),
                    kl,
                ]
            )
        header = [
            "config",
            "attention",
            "chat req/s",
            "vs baseline",
            "TTFT p90 ms",
            "TPOT p90 ms",
            "batch tok/s",
            "vs baseline",
            "KV tokens",
            "KL",
        ]
        lines += markdown_table(header, rows)

    if picks:
        lines += _plan_lines(picks)

    singles = [r for r in records if r["stage"] == "screen" and len(candidate_of(r).changes()) == 1]
    if base_screen and base_screen["status"] == "ok" and singles:
        b = base_screen
        lines += [
            "",
            "## Effect of each setting on its own",
            "",
            "Ratios against the baseline in the screen stage.",
            "",
        ]
        rows = []
        for r in singles:
            if r["status"] != "ok":
                rows.append([f"`{r['label']}`", "", "", "", "", r["status"]])
                continue
            rows.append(
                [
                    f"`{r['label']}`",
                    f"{r['tput_tok_s'] / b['tput_tok_s']:.2f}x",
                    f"{r['tpot_ms'] / b['tpot_ms']:.2f}x",
                    f"{r['kv_tokens'] / b['kv_tokens']:.2f}x",
                    f"{r['kl']:.4f}",
                    "ok" if goal.passes(r) else "; ".join(goal.violations(r)),
                ]
            )
        lines += markdown_table(["setting", "tok/s", "TPOT", "KV tokens", "KL", "verdict"], rows)

    front = set().union(
        *(pareto_keys([r for r in records if r["stage"] == s]) for s in ("screen", "combine", "confirm"))
    )
    lines += [
        "",
        "## All trials",
        "",
        "Pareto marks trials that no other trial of the same stage beats on throughput, TPOT and KL at once.",
        "",
    ]
    rows = [
        [
            str(i),
            r["stage"],
            f"`{r['label']}`",
            r["status"],
            _f(r.get("tput_tok_s"), ",.0f"),
            _f(r.get("tpot_ms"), ".1f"),
            _f(r.get("ttft_ms"), ".0f"),
            _f(r.get("kv_tokens"), ","),
            _f(r.get("kl"), ".4f"),
            _f(r.get("top1_agree"), ".3f"),
            _f(r.get("load_s"), ".0f"),
            "yes" if r["key"] in front else "",
        ]
        for i, r in enumerate(records)
    ]
    header = [
        "#",
        "stage",
        "config",
        "status",
        "tok/s",
        "TPOT ms",
        "TTFT ms",
        "KV tokens",
        "KL",
        "top-1",
        "load s",
        "Pareto",
    ]
    lines += markdown_table(header, rows)

    failed = [r for r in records if r["status"] != "ok"]
    failed_under_load = [r for r in load if r["status"] != "ok"]
    if failed or failed_under_load:
        lines += ["", "## Skipped and failed", ""]
        for r in failed:
            where = f" (`{r['dir']}/engine.log`)" if r.get("dir") else ""
            lines.append(f"- `{r['label']}`, {r['stage']}: {r['status']}, {r.get('error')}{where}")
        for r in failed_under_load:
            where = f" (`{r['dir']}/server.log`)" if r.get("dir") else ""
            lines.append(f"- `{r['label']}`, load: {r['status']}, {r.get('error')}{where}")
    return "\n".join(lines) + "\n"
