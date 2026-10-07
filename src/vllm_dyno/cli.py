from __future__ import annotations

import argparse
from importlib.metadata import version


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dyno", description="Measure vLLM serving configs on this GPU and recommend one."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('vllm-dyno')}")
    parser.parse_args(argv)
    parser.print_help()
