import json

import pytest

from vllm_dyno.cli import main


def test_version_prints_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("dyno 0.")


def test_no_command_prints_usage(capsys):
    main([])
    assert "usage: dyno" in capsys.readouterr().out


def test_load_search_refuses_a_run_measured_with_other_offline_code(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"harness_sha": "000000000000"}))
    with pytest.raises(SystemExit, match="offline measurement code changed"):
        main(["load", str(tmp_path)])
