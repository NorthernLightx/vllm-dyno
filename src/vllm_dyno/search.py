"""Which trials to run, in three stages.

screen   the baseline, then one changed setting at a time, so a failure or a gain points at one setting
combine  settings that each beat the baseline, tried together
confirm  the best few and the baseline again, on more tokens with repeated speed runs;
         the baseline rerun against its own reference shows the noise floor
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import product

from .runner import Run, run_trial, skip, trial_key
from .space import DEFAULTS, KNOBS, Candidate, Context, infeasible_reason, screen_candidates

log = logging.getLogger(__name__)

SCREEN = {"eval_tokens": 4096, "ttft_n": 3, "tpot_n": 2, "tpot_tokens": 128, "tput_repeats": 2}
CONFIRM = {"eval_tokens": 16384, "ttft_n": 5, "tpot_n": 3, "tpot_tokens": 256, "tput_repeats": 3}
POOL_SIZE = 208  # warm-up + TTFT + TPOT prompts + 64 throughput prompts per repeat, for the CONFIRM tier


@dataclass(frozen=True)
class Goal:
    target: str = "throughput"  # "throughput" maximises batch tokens/s; "latency" minimises time per output token
    max_kl: float = 0.05
    max_ttft_ms: float | None = None
    max_tpot_ms: float | None = None
    min_gain: float = 0.05  # single-repeat throughput of identical runs differed by 4% on an RTX 3070

    def score(self, r: dict) -> float:
        return r["tput_tok_s"] if self.target == "throughput" else -r["tpot_ms"]

    def gain(self, r: dict, base: dict) -> float:
        """How many times better than the baseline on the target metric."""
        return r["tput_tok_s"] / base["tput_tok_s"] if self.target == "throughput" else base["tpot_ms"] / r["tpot_ms"]

    def violations(self, r: dict) -> list[str]:
        out = []
        if r["kl"] > self.max_kl:
            out.append(f"KL {r['kl']:.4f} > {self.max_kl}")
        if self.max_ttft_ms is not None and r["ttft_ms"] > self.max_ttft_ms:
            out.append(f"TTFT {r['ttft_ms']:.0f} ms > {self.max_ttft_ms:.0f}")
        if self.max_tpot_ms is not None and r["tpot_ms"] > self.max_tpot_ms:
            out.append(f"TPOT {r['tpot_ms']:.1f} ms > {self.max_tpot_ms:.1f}")
        return out

    def passes(self, r: dict) -> bool:
        return r["status"] == "ok" and not self.violations(r)


def candidate_of(r: dict) -> Candidate:
    return Candidate(**{k: r[k] for k in KNOBS})


def promising(screen: list[dict], goal: Goal) -> dict[str, dict[str, float]]:
    """Per setting, the values that beat the baseline on their own, with their gain."""
    base = next(r for r in screen if r["label"] == "baseline")
    out: dict[str, dict[str, float]] = {k: {} for k in KNOBS}
    for r in screen:
        changes = candidate_of(r).changes()
        if len(changes) == 1 and goal.passes(r) and goal.gain(r, base) >= 1 + goal.min_gain:
            knob, value = next(iter(changes.items()))
            out[knob][value] = goal.gain(r, base)
    return out


def combinations(good: dict[str, dict[str, float]], limit: int) -> list[Candidate]:
    """Pairs, triples, ... of promising values, ordered by the product of their single gains."""
    options = [[DEFAULTS[k], *good[k]] for k in KNOBS]
    cands = [Candidate(**dict(zip(KNOBS, combo))) for combo in product(*options)]
    multi = [c for c in cands if len(c.changes()) >= 2]
    estimate = {c: math.prod(good[k][v] for k, v in c.changes().items()) for c in multi}
    return sorted(multi, key=estimate.__getitem__, reverse=True)[:limit]


def finalists(records: list[dict], goal: Goal, top: int) -> list[Candidate]:
    ranked = sorted((r for r in records if goal.passes(r) and r["label"] != "baseline"), key=goal.score, reverse=True)
    out: list[Candidate] = []
    for r in ranked:
        cand = candidate_of(r)
        if cand not in out:
            out.append(cand)
    return out[:top]


def _log_trial(i: int, n: int, r: dict, goal: Goal, recorded: bool) -> None:
    head = f"{r['stage']:<8}{i}/{n}  {r['label']:<55}"
    if r["status"] != "ok":
        log.info("%s %s: %s", head, r["status"], r.get("error"))
        return
    note = "" if goal.passes(r) else "  fails: " + ", ".join(goal.violations(r))
    log.info(
        "%s ok  %6.0f tok/s  TPOT %5.1f ms  TTFT %5.0f ms  KL %.4f  top-1 %.3f  (%s)%s",
        head,
        r["tput_tok_s"],
        r["tpot_ms"],
        r["ttft_ms"],
        r["kl"],
        r["top1_agree"],
        "recorded earlier" if recorded else f"{r['wall_s']:.0f} s",
        note,
    )


@dataclass(frozen=True)
class Searcher:
    run: Run
    ctx: Context
    goal: Goal
    env: dict[str, str]
    timeout_s: float

    def stage(self, stage: str, cands: list[Candidate], tier: dict) -> list[dict]:
        records = []
        for i, cand in enumerate(cands, 1):
            recorded = self.run.find(trial_key(stage, cand)) is not None
            reason = infeasible_reason(cand, self.ctx)
            if reason:
                r = skip(self.run, stage, cand, reason)
            else:
                write_ref = stage == "screen" and cand.label() == "baseline"
                r = run_trial(
                    self.run,
                    self.ctx,
                    stage,
                    cand,
                    tier,
                    write_reference=write_ref,
                    env=self.env,
                    timeout_s=self.timeout_s,
                )
            _log_trial(i, len(cands), r, self.goal, recorded)
            records.append(r)
        return records

    def search(self, confirm_top: int, max_combos: int) -> None:
        screen = self.stage("screen", screen_candidates(self.ctx), SCREEN)
        base = screen[0]
        if base["status"] != "ok":
            where = self.run.path / base.get("dir", "")
            raise SystemExit(f"the baseline did not run ({base['status']}: {base.get('error')}); see {where}")

        combos = combinations(promising(screen, self.goal), max_combos)
        if combos:
            combined = self.stage("combine", combos, SCREEN)
        else:
            combined = []
            log.info("combine  gains came from fewer than two settings; nothing to combine")

        self.stage("confirm", [Candidate(), *finalists(screen + combined, self.goal, confirm_top)], CONFIRM)
