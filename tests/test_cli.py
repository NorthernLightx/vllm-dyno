import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict

import pytest

from vllm_dyno import cli, planner, probe
from vllm_dyno.cli import main
from vllm_dyno.fit import FitProposer
from vllm_dyno.planner import AllProposer, LlmProposer
from vllm_dyno.runner import HARNESS_SHA, Run
from vllm_dyno.search import Goal


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_server_stops_with_dyno_when_dyno_is_terminated(tmp_path, sig):
    pid_file, log_file = tmp_path / "server.pid", tmp_path / "server.log"
    server = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    dyno = (
        "import os, sys; from pathlib import Path; from vllm_dyno import cli, load; cli._unwind_on_signals()\n"
        f"with load.serving([sys.executable, '-c', {server!r}], Path({str(log_file)!r}), dict(os.environ), 60):\n"
        "    pass"
    )
    proc = subprocess.Popen([sys.executable, "-c", dyno])
    deadline = time.monotonic() + 30
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < deadline and proc.poll() is None
        time.sleep(0.1)
    pid = int(pid_file.read_text())
    proc.send_signal(sig)
    assert proc.wait(timeout=30) == 128 + sig
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_engines_start_with_the_hub_offline_once_every_model_the_run_loads_is_cached(monkeypatch, ctx):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    cached = {"org/base"}

    def lookup(repo, revision=None, local_files_only=False):
        if not local_files_only or repo not in cached:
            raise LocalEntryNotFoundError(repo)
        return f"/cache/{repo}"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lookup)
    assert cli._hub_offline({"PATH": "/bin"}, ctx) == {"PATH": "/bin"}
    cached.add("org/base-AWQ")  # the 3-bit GPTQ checkpoint cannot run on sm86, so it is never loaded
    assert cli._hub_offline({"PATH": "/bin"}, ctx) == {"PATH": "/bin", "HF_HUB_OFFLINE": "1"}


def test_version_prints_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("dyno 0.")


def test_no_command_prints_usage(capsys):
    main([])
    assert "usage: dyno" in capsys.readouterr().out


@pytest.mark.parametrize(
    "meta",
    [
        {"harness_sha": "000000000000"},
        {"harness_sha": HARNESS_SHA, "context": {"host": {"nvcc": False, "wsl": False}, "util": 0.6}, "env": {}},
    ],
)
def test_load_search_refuses_a_run_measured_with_other_offline_code_or_environment(tmp_path, meta):
    (tmp_path / "run.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="offline measurement code or environment changed"):
        main(["load", str(tmp_path)])


def test_load_takes_the_planner_key_out_of_the_environment_before_starting_anything(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    with pytest.raises(SystemExit, match="is not a run directory"):
        main(["load", str(tmp_path / "missing"), "--planner", "llm"])
    assert "OPENROUTER_API_KEY" not in os.environ


@pytest.mark.parametrize(
    ("argv", "kind", "name", "budget"),
    [
        (["--planner", "llm", "--planner-url", "http://localhost:8000/v1"], LlmProposer, "org/model", 10),
        (["--planner", "fit"], FitProposer, "fit", 10),
        (["--planner", "fit", "--max-servers", "3"], FitProposer, "fit", 3),
        (["--planner", "all"], AllProposer, "all", sys.maxsize),
        (["--planner", "all", "--max-servers", "3"], AllProposer, "all", 3),
    ],
)
def test_load_hands_the_planning_loop_the_planner_it_names_and_its_server_budget(
    tmp_path, monkeypatch, ctx, argv, kind, name, budget
):
    meta = {"created": "2026-10-12 10:00", "context": ctx.to_json(), "goal": asdict(Goal()), "harness_sha": HARNESS_SHA}
    Run(tmp_path).write_meta(meta | {"env": cli._trial_env(meta)})
    (tmp_path / "data.json").write_text(json.dumps({"speed_prompts": []}))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # fit and all need none
    monkeypatch.setattr(probe, "query_gpu", lambda: ctx.gpu)
    monkeypatch.setattr(cli, "_hub_offline", lambda env, ctx: env)
    monkeypatch.setattr(cli, "_setup_logging", lambda logfile: None)
    calls = []
    monkeypatch.setattr(planner, "plan", lambda session, proposer, goal, target, **kw: calls.append((proposer, kw)))
    main(["load", str(tmp_path), "--planner-model", "org/model", *argv])
    [(proposer, kw)] = calls
    assert (type(proposer), proposer.name, kw["max_servers"]) == (kind, name, budget)


@pytest.mark.parametrize("value", [None, ""])
def test_the_llm_planner_needs_a_key_for_an_endpoint_elsewhere(tmp_path, monkeypatch, value):
    monkeypatch.delenv("CLAUDE_KEY", raising=False)
    if value is not None:
        monkeypatch.setenv("CLAUDE_KEY", value)
    argv = ["load", str(tmp_path), "--planner", "llm", "--planner-key-env", "CLAUDE_KEY"]
    with pytest.raises(SystemExit, match="https://api.anthropic.com/v1 needs an API key; set CLAUDE_KEY"):
        main([*argv, "--planner-url", "https://api.anthropic.com/v1"])
    with pytest.raises(SystemExit, match="is not a run directory"):
        main([*argv, "--planner-url", "http://localhost:8000/v1"])
