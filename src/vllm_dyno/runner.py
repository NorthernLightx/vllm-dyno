"""Run directory and trial execution.

<run>/run.json        settings and machine facts; a resumed run reuses them
<run>/trials.jsonl    one line per trial, including skipped and failed ones
<run>/trials/NNN-*/   spec.json, engine.log and metrics.json of one trial
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import harness
from .probe import query_gpu
from .space import Candidate, Context, engine_args

log = logging.getLogger(__name__)

HARNESS_SHA = hashlib.sha256(Path(harness.__file__).read_bytes()).hexdigest()[:12]

OOM_PATTERNS = (
    "out of memory",
    "outofmemoryerror",
    "no available memory for the cache blocks",
    "available kv cache memory",
    "less than desired gpu memory utilization",
)
UNSUPPORTED_PATTERNS = ("not supported", "notimplementederror", "unsupported", "does not support")
ERROR_LINE = re.compile(r"^.*?(\w*(?:Error|Exception)): (.+)$", re.MULTILINE)
# raised in the main process when the engine core process dies; the cause is logged above it
WRAPPERS = ("engine core initialization failed", "enginecore", "enginedeaderror")


def classify(log_text: str) -> tuple[str, str]:
    """Status and a one-line error for a trial process that exited without metrics.

    Only exception lines decide "unsupported": a successful engine logs plenty of
    "not supported" warnings too.
    """
    errors = [f"{name}: {msg.strip()}" for name, msg in ERROR_LINE.findall(log_text)]
    causes = [e for e in errors if not any(w in e.lower() for w in WRAPPERS)] or errors
    error = (causes[-1] if causes else (log_text.strip().splitlines() or ["no output"])[-1])[:300]
    error_text = " ".join(errors).lower()
    if any(p in error_text for p in OOM_PATTERNS):
        return "oom", error
    if any(p in error_text for p in UNSUPPORTED_PATTERNS):
        return "unsupported", error
    if any(p in log_text.lower() for p in OOM_PATTERNS):
        return "oom", error
    return "crash", error


class Run:
    def __init__(self, path: Path):
        self.path = path
        self.trials_file = path / "trials.jsonl"
        self.records: list[dict] = []
        if self.trials_file.exists():
            self.records = [json.loads(line) for line in self.trials_file.read_text().splitlines() if line.strip()]

    @property
    def meta(self) -> dict:
        return json.loads((self.path / "run.json").read_text())

    def write_meta(self, meta: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "run.json").write_text(json.dumps(meta, indent=2))

    def find(self, key: str) -> dict | None:
        return next((r for r in self.records if r["key"] == key), None)

    def append(self, record: dict) -> None:
        self.records.append(record)
        with self.trials_file.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:60]


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # vLLM's engine core is a child process; take it down too
    except ProcessLookupError:
        pass


def wait_for_gpu(free_mib: int, slack_mib: int = 400, timeout_s: float = 60) -> None:
    """Block until GPU memory is back near the level measured before the run."""
    deadline = time.monotonic() + timeout_s
    while (now := query_gpu().free_mib) < free_mib - slack_mib:
        if time.monotonic() > deadline:
            log.warning("GPU memory not released: %d MiB free, expected about %d", now, free_mib)
            return
        time.sleep(2)


def trial_key(stage: str, cand: Candidate) -> str:
    return f"{stage}|{cand.label()}"


def skip(run: Run, stage: str, cand: Candidate, reason: str) -> dict:
    key = trial_key(stage, cand)
    if record := run.find(key):
        return record
    record = {"key": key, "stage": stage, "label": cand.label(), **asdict(cand), "status": "skipped", "error": reason}
    run.append(record)
    return record


def run_trial(
    run: Run,
    ctx: Context,
    stage: str,
    cand: Candidate,
    tier: dict,
    *,
    write_reference: bool,
    env: dict[str, str],
    timeout_s: float,
) -> dict:
    key = trial_key(stage, cand)
    if record := run.find(key):
        return record
    trial_dir = run.path / "trials" / f"{len(run.records):03d}-{stage}-{_slug(cand.label())}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "engine": engine_args(cand, ctx),
        "tier": tier,
        "data": str(run.path / "data.json"),
        "reference": str(run.path / "reference.npz"),
        "write_reference": write_reference,
        "out": str(trial_dir / "metrics.json"),
    }
    (trial_dir / "spec.json").write_text(json.dumps(spec, indent=2))

    started = time.monotonic()
    status, error, metrics = "ok", None, {}
    with (trial_dir / "engine.log").open("w") as out:
        cmd = [sys.executable, "-m", "vllm_dyno.harness", str(trial_dir / "spec.json")]
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            code = None
        except KeyboardInterrupt:
            _kill_group(proc)
            raise
        finally:
            _kill_group(proc)
    if code is None:
        status, error = "timeout", f"no result after {timeout_s:.0f} s"
    elif (trial_dir / "metrics.json").exists():
        metrics = json.loads((trial_dir / "metrics.json").read_text())
    else:
        status, error = classify((trial_dir / "engine.log").read_text(errors="replace"))

    record = {
        "key": key,
        "stage": stage,
        "label": cand.label(),
        **asdict(cand),
        "status": status,
        "error": error,
        **metrics,
        "wall_s": round(time.monotonic() - started, 1),
        "harness_sha": HARNESS_SHA,
        "dir": str(trial_dir.relative_to(run.path)),
    }
    run.append(record)
    wait_for_gpu(run.meta["free_mib_at_start"])
    return record
