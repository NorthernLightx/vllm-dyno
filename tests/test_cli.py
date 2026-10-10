import json

import pytest

from vllm_dyno.cli import main
from vllm_dyno.runner import HARNESS_SHA


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
        {"harness_sha": HARNESS_SHA, "context": {"host": {"nvcc": False, "wsl": False}}, "env": {}},
    ],
)
def test_load_search_refuses_a_run_measured_with_other_offline_code_or_environment(tmp_path, meta):
    (tmp_path / "run.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="offline measurement code or environment changed"):
        main(["load", str(tmp_path)])
