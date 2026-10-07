"""report.md from run.json and trials.jsonl."""

from __future__ import annotations

from pathlib import Path

from .probe import serve_env
from .search import Goal, candidate_of
from .space import Context, serve_command


def recommendation(records: list[dict], goal: Goal) -> dict | None:
    confirmed = [r for r in records if r["stage"] == "confirm" and goal.passes(r)]
    return max(confirmed, key=goal.score) if confirmed else None


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


def _cost(r: dict, gpu_price: float | None) -> str:
    if not gpu_price or r.get("status") != "ok":
        return ""
    return f"{gpu_price / (r['tput_tok_s'] * 3600) * 1e6:.3f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render(meta: dict, records: list[dict], run_dir: Path) -> str:
    ctx = Context.from_json(meta["context"])
    goal = Goal(**meta["goal"])
    price = meta.get("gpu_price")
    gpu = ctx.gpu
    base_screen = next((r for r in records if r["stage"] == "screen" and r["label"] == "baseline"), None)
    base_confirm = next((r for r in records if r["stage"] == "confirm" and r["label"] == "baseline"), None)
    best = recommendation(records, goal)

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
        f"Goal: {target} with {', '.join(limits)}. KL is measured against the baseline's next-token distribution.",
        "",
        "## Recommendation",
        "",
    ]
    if best and base_confirm:
        b = base_confirm
        env_vars = [f"{k}={v}" for k, v in serve_env(ctx.host.get("nvcc", False)).items()]
        lines += [
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
            "Start it with the environment the trials ran in:",
            "",
            "```",
            f"dyno serve {run_dir}",
            "```",
            "",
            "or directly:",
            "",
            "```",
            " ".join([*env_vars, serve_command(candidate_of(best), ctx)]),
            "```",
            "",
        ]
        if ctx.host.get("unsearchable_path"):
            lines += [
                (
                    f"The trials ran with {ctx.host['unsearchable_path']} PATH entries removed that Linux cannot "
                    "search (WSL adds Windows directories). With them in PATH, torch fails while probing for nvcc; "
                    "`dyno serve` removes them."
                ),
                "",
            ]
        lines += [
            (
                f"The baseline rerun scored KL {b['kl']:.4f} and top-1 agreement {b['top1_agree']:.3f} against its own "
                f"reference, and its throughput varied {_f((b.get('tput_cv') or 0) * 100, '.1f')}% across repeats. "
                "Differences smaller than these are noise."
            ),
        ]
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
        lines += _table(header, rows)

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
        lines += _table(["setting", "tok/s", "TPOT", "KV tokens", "KL", "verdict"], rows)

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
    lines += _table(header, rows)

    failed = [r for r in records if r["status"] != "ok"]
    if failed:
        lines += ["", "## Skipped and failed", ""]
        for r in failed:
            where = f" (`{r['dir']}/engine.log`)" if r.get("dir") else ""
            lines.append(f"- `{r['label']}`, {r['stage']}: {r['status']}, {r.get('error')}{where}")
    return "\n".join(lines) + "\n"
