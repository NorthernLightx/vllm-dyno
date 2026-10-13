"""A planner for `dyno load` that fits each setting's gain under load and loads the config likeliest to win.

A config's log chat gain, and its log batch gain, is the sum of one weight per value it changes from the
baseline. A weight starts from PRIOR_SHRINK times the log offline throughput gain of its value's screen
trial, and every load record of the run, against the baseline of its own session, updates it by Bayesian
linear regression. vLLM picks the attention backend by the KV cache format, so a config that leaves the
backend to vLLM counts as naming the backend that servers with its KV cache format picked: otherwise the
weight of that format would hold the backend's effect too, and a config naming both would count it twice.

At each step the planner draws gains of the configs it may pick from the fit, keeps the measured gains of
the configs that served, and applies the report's recommendation rule to each draw. A config's win
probability is the share of draws that recommend it when every config it may pick is in the draw, and
its change probability the share that recommend it when it alone joins the configs that served. While
the effect under load of a setting is not known, the planner picks among the configs that differ from one
that served in such a setting alone, so that its stop counts; after that among all. It stops once no
config would change the recommendation in STOP_P of the draws: similar configs split the win
probability between them, so a stop on it would come while one of them is still likely to win. It leaves
out a config that names the attention backend the server of its auto twin picked anyway, which starts no
server.

<run>/plan/*.json    the fitted weights, and each config's predicted gains, KL, and win and change
                     probabilities, which are null for a config the step did not let it pick
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

from .load import MIN_RATE, gains, load_records, session_baseline
from .planner import Proposal, Step
from .report import load_scores, load_winner, quality
from .search import Goal, candidate_of
from .space import DEFAULTS, KNOBS, VALUES, Candidate

# the share of a value's log offline throughput gain its weight starts from: offline speed ranks
# configs, but a server mixes prefills into its decode batches and can rank them differently
PRIOR_SHRINK = 0.5
PRIOR_SD = 0.25  # of a weight before any load record
NOISE_SD = 0.04  # of a log gain between loads of one config
DRAWS = 256
STOP_P = 0.1
Z90 = NormalDist().inv_cdf(0.95)

Value = tuple[str, str]  # a setting and the value a config gives it


def _cholesky(a: list[list[float]]) -> list[list[float]]:
    """The lower triangular L with L Lᵀ = a, for a symmetric positive definite a."""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            low[i][j] = math.sqrt(s) if i == j else s / low[j][j]
    return low


def _solve(low: list[list[float]], b: list[float]) -> list[float]:
    """x with L Lᵀ x = b, for the lower triangular L."""
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
    return x


def _dot(x: list[float], w: list[float]) -> float:
    return sum(a * b for a, b in zip(x, w))


@dataclass(frozen=True)
class Posterior:
    """A normal distribution of weights."""

    mean: list[float]
    cov: list[list[float]]

    def predict(self, x: list[float]) -> tuple[float, float]:
        """The mean and the standard deviation of x · w."""
        var = sum(x[i] * self.cov[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))
        return _dot(x, self.mean), math.sqrt(max(var, 0.0))

    def draws(self, rng: random.Random, n: int) -> list[list[float]]:
        """n draws of the weights."""
        low = _cholesky(self.cov)
        out = []
        for _ in range(n):
            z = [rng.gauss(0.0, 1.0) for _ in self.mean]
            out.append([m + _dot(low[i][: i + 1], z) for i, m in enumerate(self.mean)])
        return out


def posterior(
    rows: list[list[float]], ys: list[float], prior: list[float], prior_sd: float = PRIOR_SD, noise_sd: float = NOISE_SD
) -> Posterior:
    """The weights w of ys = rows · w + noise, from w ~ N(prior, prior_sd² I) and independent noise of sd noise_sd.

    Precision A = I / prior_sd² + XᵀX / noise_sd², mean A⁻¹ (prior / prior_sd² + Xᵀy / noise_sd²), covariance A⁻¹.
    """
    d = len(prior)
    a = [
        [float(i == j) / prior_sd**2 + sum(r[i] * r[j] for r in rows) / noise_sd**2 for j in range(d)] for i in range(d)
    ]
    b = [prior[i] / prior_sd**2 + sum(r[i] * y for r, y in zip(rows, ys)) / noise_sd**2 for i in range(d)]
    low = _cholesky(a)
    return Posterior(_solve(low, b), [_solve(low, [float(k == i) for k in range(d)]) for i in range(d)])


def _single(value: Value) -> Candidate:
    return Candidate(**{value[0]: value[1]})


def _screen(records: list[dict], label: str) -> dict | None:
    return next((r for r in records if r["stage"] == "screen" and r["label"] == label and r["status"] == "ok"), None)


def prior_mean(records: list[dict], value: Value) -> float:
    """PRIOR_SHRINK times the log throughput gain of the value's screen trial over the screen baseline; 0 without one."""
    base, single = _screen(records, "baseline"), _screen(records, _single(value).label())
    if base is None or single is None:
        return 0.0
    return PRIOR_SHRINK * math.log(single["tput_tok_s"] / base["tput_tok_s"])


def picked_backends(load: list[dict]) -> dict[str, str]:
    """Per KV cache format other than the default, a named backend that a server of a config with it picked."""
    return {
        r["kv_cache_dtype"]: r["backend"]
        for r in load
        if r["status"] == "ok"
        and r["kv_cache_dtype"] != DEFAULTS["kv_cache_dtype"]
        and r["attention_backend"] == DEFAULTS["attention_backend"]
        and r.get("backend") in VALUES["attention_backend"]
    }


def engine(c: Candidate, backends: dict[str, str]) -> Candidate:
    """c naming the backend that servers with its KV cache format picked, if it leaves the backend to vLLM."""
    if c.attention_backend == DEFAULTS["attention_backend"] and c.kv_cache_dtype in backends:
        return c.replace(attention_backend=backends[c.kv_cache_dtype])
    return c


@dataclass(frozen=True)
class Fit:
    """One weight per value for the log chat gain and one for the log batch gain."""

    values: list[Value]
    chat: Posterior
    batch: Posterior
    backends: dict[str, str] = field(default_factory=dict)  # from picked_backends()

    def x(self, c: Candidate) -> list[float]:
        """Which of the values c's engine sets, as 0 or 1."""
        changed = set(engine(c, self.backends).changes().items())
        return [float(v in changed) for v in self.values]

    def predicted_gains(self, c: Candidate) -> dict[str, list[float]]:
        """c's predicted chat and batch gains: the posterior mean and a central 90% interval, as multiples."""
        out = {}
        for name, post in (("chat", self.chat), ("batch", self.batch)):
            m, sd = post.predict(self.x(c))
            out[name] = [math.exp(m), math.exp(m - Z90 * sd), math.exp(m + Z90 * sd)]
        return out


def fit(records: list[dict], load: list[dict], cands: list[Candidate]) -> Fit:
    """Weights fitted to every load record that served, but the baselines, for the values of those and of cands."""
    served = [r for r in load if r["status"] == "ok" and r["label"] != "baseline"]
    backends = picked_backends(load)
    found = {v for c in [*cands, *map(candidate_of, served)] for v in engine(c, backends).changes().items()}
    values = sorted(found, key=lambda v: (KNOBS.index(v[0]), v[1]))
    prior = [prior_mean(records, v) for v in values]
    rows: dict[str, tuple[list[list[float]], list[float]]] = {"chat": ([], []), "batch": ([], [])}
    for r in served:
        changed = set(engine(candidate_of(r), backends).changes().items())
        base = session_baseline(load, r)
        chat, batch = gains(r, base)
        if chat == 0 and base is not None:
            # even MIN_RATE failed, so the gain is below MIN_RATE over the baseline's: the bound stands in for it
            chat = MIN_RATE / base["capacity_rps"]
        for name, g in (("chat", chat), ("batch", batch)):
            if g:  # None without a baseline to compare with; a batch that finished no request has no log gain
                rows[name][0].append([float(v in changed) for v in values])
                rows[name][1].append(math.log(g))
    return Fit(values, posterior(*rows["chat"], prior), posterior(*rows["batch"], prior), backends)


def predicted_kl(records: list[dict], c: Candidate) -> float:
    """c's KL as the recommendation weighs it once c served; the sum of its values' alone if c has no trial yet."""
    q = quality(records, c.label())
    if q is not None:
        return q["kl"]
    singles = [quality(records, _single(v).label()) for v in c.changes().items()]
    return sum(s["kl"] if s else math.inf for s in singles)


def _settles(c: Candidate, served: set[Candidate], untried: list[str]) -> list[str]:
    """The untried settings whose effect under load serving c would show."""
    return [k for k in untried if k in c.changes() and c.replace(**{k: DEFAULTS[k]}) in served]


def _x(gain: list[float]) -> str:
    return f"{gain[0]:.2f}x (90% interval {gain[1]:.2f}x to {gain[2]:.2f}x)"


@dataclass(frozen=True)
class Shares:
    """Per config, shares of DRAWS draws of the fitted gains that recommend it."""

    win: dict[Candidate, float]  # with every config it may pick in the draw
    change: dict[Candidate, float]  # with it alone joining the configs that served


def win_shares(
    model: Fit,
    served: dict[str, tuple[float, float, float]],
    pool: list[Candidate],
    kl: dict[Candidate, float],
    goal: Goal,
    target: str,
    seed: int,
) -> Shares:
    """Per config of pool, the shares of DRAWS draws of its gains from the fit that recommend it for target.

    served holds the chat gain, batch gain and KL of each config that served, which keep their measured
    values in every draw. A config of pool over the KL limit is never recommended.
    """
    contenders = {c.label(): c for c in pool if kl[c] <= goal.max_kl}
    wins, changes = dict.fromkeys(pool, 0), dict.fromkeys(pool, 0)
    if contenders:
        rng = random.Random(seed)
        chat, batch = model.chat.draws(rng, DRAWS), model.batch.draws(rng, DRAWS)
        xs = {label: model.x(c) for label, c in contenders.items()}
        for wc, wb in zip(chat, batch):
            drawn = {
                label: (math.exp(_dot(x, wc)), math.exp(_dot(x, wb)), kl[contenders[label]]) for label, x in xs.items()
            }
            winner = load_winner(served | drawn, target)[0]
            if winner in contenders:
                wins[contenders[winner]] += 1
            for label, score in drawn.items():
                if load_winner(served | {label: score}, target)[0] == label:
                    changes[contenders[label]] += 1
    return Shares({c: n / DRAWS for c, n in wins.items()}, {c: n / DRAWS for c, n in changes.items()})


class FitProposer:
    """Picks the config likeliest to be recommended under a fitted model of each setting's gain."""

    name: str = "fit"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, step: Step) -> Proposal:
        """While a setting is untried, the likeliest winner that settles one; then the likeliest winner, or a stop."""
        records, load = step.session.run.records, load_records(step.session.run.path)
        served = {candidate_of(r) for r in load if r["status"] == "ok"}
        # plan() starts no server for a config that names the backend its auto twin's server picked anyway;
        # once only such configs are left, picking them runs the list out and ends the session
        options = [c for c in step.options if not step.session.same_engine(asdict(c))] or step.options
        settles = {c: _settles(c, served, step.untried) for c in options}
        # none settles a setting alone once each config that would has failed under load
        pool = [c for c in options if settles[c]] or options
        model = fit(records, load, step.options)
        kl = {c: predicted_kl(records, c) for c in step.options}
        scores = {label: s for label, (_, s) in load_scores(load, records, step.goal).items()}
        shares = win_shares(model, scores, pool, kl, step.goal, step.target, self.seed)
        p = shares.win
        best = max(pool, key=p.__getitem__)  # the first of equals, in the order of the options
        g = model.predicted_gains(best)
        changer = max(pool, key=shares.change.__getitem__)
        if not step.untried and shares.change[changer] < STOP_P:
            proposal = Proposal(
                None,
                f"No config left, loaded next, would be recommended in {STOP_P:.0%} of {DRAWS} draws of the "
                f"fitted gains; the likeliest, {changer.label()}, would in {shares.change[changer]:.0%}.",
            )
        else:
            odds = f"{p[best]:.0%} of {DRAWS} draws of the fitted gains: chat {_x(g['chat'])}, batch {_x(g['batch'])}."
            if settles[best]:
                ref = best.replace(**{settles[best][0]: DEFAULTS[settles[best][0]]}).label()
                ref = "the baseline" if ref == "baseline" else ref
                hypothesis = f"Settles {', '.join(settles[best])} against {ref} and wins {odds}"
            else:
                hypothesis = f"Wins {odds}"
            proposal = Proposal(best, hypothesis, g["chat"][0], g["batch"][0])
        step.step_file.write_text(_step(model, step.options, kl, shares, proposal))
        return proposal


def _step(model: Fit, options: list[Candidate], kl: dict[Candidate, float], shares: Shares, pick: Proposal) -> str:
    weights = {}
    for i, (k, v) in enumerate(model.values):
        unit = [float(j == i) for j in range(len(model.values))]
        weights[f"{k}={v}"] = {"chat": model.chat.predict(unit), "batch": model.batch.predict(unit)}
    configs = [
        {
            "config": c.label(),
            "win": shares.win.get(c),
            "change": shares.change.get(c),
            "kl": kl[c] if math.isfinite(kl[c]) else None,
            **model.predicted_gains(c),
        }
        for c in options
    ]
    return json.dumps({"weights": weights, "configs": configs, "pick": pick.config and pick.config.label()}, indent=2)
