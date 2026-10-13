import dataclasses
import json
from pathlib import Path

import pytest

from vllm_dyno import runner
from vllm_dyno.runner import Run, classify, run_trial, skip
from vllm_dyno.search import SCREEN
from vllm_dyno.space import Candidate

WRAPPED = """\
(EngineCore pid=123) ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""
PROFILED = """\
(EngineCore pid=36895) INFO 10-12 12:03:58 [model_runner.py:1087] Graph capturing finished in 7 secs, took 0.28 GiB
(EngineCore pid=36895) INFO 10-12 12:03:58 [gpu_worker.py:692] Available KV cache memory: 1.34 GiB
(EngineCore pid=36895) INFO 10-12 12:03:58 [kv_cache_utils.py:2464] GPU KV cache size: 12,528 tokens, \
Maximum concurrency for 2,048 tokens per request: 6.12x
"""


def failed(cause: str) -> str:
    """The end of a log whose engine core died of cause."""
    return (
        f"(EngineCore pid=36895) ERROR 10-12 12:04:01 [core.py:1433] {cause}\n"
        "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}\n"
    )


def test_out_of_memory_is_reported_with_its_cause_not_the_wrapper():
    status, error = classify(WRAPPED)
    assert status == "oom"
    assert error.startswith("OutOfMemoryError: CUDA out of memory")


def test_a_crash_after_memory_profiling_is_not_an_oom():
    cause = "RuntimeError: CUDA driver error: device not ready"
    assert classify(PROFILED + failed(cause)) == ("crash", cause)


def test_an_oom_that_aborts_in_cpp_is_found_without_an_exception_line():
    aborted = (
        "terminate called after throwing an instance of 'c10::OutOfMemoryError'\n"
        "  what():  CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 8.00 GiB "
        "of which 0 bytes is free.\n"
    )
    status, error = classify(PROFILED + aborted)
    assert status == "oom"
    assert "CUDA out of memory" in error


VLLM_MEMORY_ERRORS = [
    (
        "ValueError: To serve at least one request with the model's max seq len (2048), (0.22 GiB KV cache is "
        "needed, which is larger than the available KV cache memory (0.11 GiB). Based on the available memory, "
        "the estimated maximum model length is 1024. Try increasing `gpu_memory_utilization`"
    ),
    "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`",
    (
        "ValueError: Free memory on device cuda:0 (5.12/8.00 GiB) on startup is less than desired GPU memory "
        "utilization (0.74, 5.92 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes."
    ),
]


@pytest.mark.parametrize("cause", VLLM_MEMORY_ERRORS)
def test_vllm_memory_errors_are_oom(cause):
    assert classify(failed(cause)) == ("oom", cause[:300])


def test_unsupported_comes_from_the_exception_line():
    log = "INFO loading\nValueError: fp8 KV cache is not supported on this platform\n"
    assert classify(log) == ("unsupported", "ValueError: fp8 KV cache is not supported on this platform")


def test_warnings_saying_not_supported_do_not_mark_a_crash_unsupported():
    log = "WARNING cuda graphs not supported for X, falling back\nKeyError: 'lm_head.weight'\n"
    assert classify(log) == ("crash", "KeyError: 'lm_head.weight'")


def test_run_records_survive_a_restart_and_skips_are_recorded_once(tmp_path):
    run = Run(tmp_path)
    cand = Candidate(kv_cache_dtype="fp8")
    first = skip(run, "screen", cand, "no kernel")
    assert skip(run, "screen", cand, "no kernel") is first
    reloaded = Run(tmp_path)
    assert reloaded.records == [first]
    assert (reloaded.find("screen|kv_cache_dtype=fp8") or {}).get("status") == "skipped"


def test_a_trial_waits_for_the_gpu_memory_free_before_it_started(tmp_path, monkeypatch, ctx):
    run = Run(tmp_path)
    run.write_meta({"free_mib_at_start": 7000})  # the desktop has taken more since

    class Finished:
        pid = 0

        def wait(self, timeout=None):
            return 0

    def popen(cmd, **kwargs):
        spec = json.loads(Path(cmd[-1]).read_text())
        Path(spec["out"]).write_text(json.dumps({"tput_tok_s": 1000.0}))
        return Finished()

    waits: list[int] = []
    monkeypatch.setattr(runner, "query_gpu", lambda: dataclasses.replace(ctx.gpu, free_mib=5200))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "kill_group", lambda proc: None)
    monkeypatch.setattr(runner, "wait_for_gpu", waits.append)
    r = run_trial(run, ctx, "screen", Candidate(), SCREEN, write_reference=False, env={}, timeout_s=60)
    assert (r["status"], r["tput_tok_s"], waits) == ("ok", 1000.0, [5200])
