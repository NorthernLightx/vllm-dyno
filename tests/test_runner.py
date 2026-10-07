from vllm_dyno.runner import Run, classify, skip
from vllm_dyno.space import Candidate

WRAPPED = """\
(EngineCore pid=123) ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""


def test_out_of_memory_is_reported_with_its_cause_not_the_wrapper():
    status, error = classify(WRAPPED)
    assert status == "oom"
    assert error.startswith("OutOfMemoryError: CUDA out of memory")


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
