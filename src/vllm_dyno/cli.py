from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

log = logging.getLogger("vllm_dyno")

IGNORE_FILES = ["*.bin", "*.pth", "*.pt", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "original/*"]


def _setup_logging(logfile: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S", handlers=handlers)
    for noisy in ("httpx", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_context(args: argparse.Namespace):
    from .probe import AttentionShape, host_facts, memory_utilization, model_files, query_gpu
    from .space import QUANT_METHODS, Context, checkpoint_from, discover_checkpoints

    gpu = query_gpu()
    host = host_facts()
    cfg, base_bytes, revision = model_files(args.model)
    methods = QUANT_METHODS if host["bitsandbytes"] else QUANT_METHODS - {"bitsandbytes"}
    checkpoints = (
        []
        if args.no_discover
        else discover_checkpoints(args.model, args.max_checkpoints, methods, any_owner=args.any_owner)
    )
    for repo in args.checkpoint:
        ckpt = checkpoint_from(repo)
        if ckpt is None:
            raise SystemExit(f"{repo}: config.json declares no quantization method vLLM loads")
        checkpoints.append(ckpt)
    util = memory_utilization(gpu)
    if util < 0.2:
        raise SystemExit(f"only {gpu.free_mib:,} of {gpu.total_mib:,} MiB GPU memory is free")
    ctx = Context(
        model=args.model,
        gpu=gpu,
        host=host,
        shape=AttentionShape.from_hf_config(cfg),
        base_bytes=base_bytes,
        checkpoints=tuple(checkpoints),
        util=util,
        max_model_len=args.max_model_len,
        revision=revision,
    )
    return ctx


def _log_plan(ctx) -> None:
    from .probe import GIB, kv_bytes_per_token
    from .space import infeasible_reason, screen_candidates

    gpu, host = ctx.gpu, ctx.host
    log.info(
        "GPU      %s, compute capability %s, %s of %s MiB free",
        gpu.name,
        gpu.compute_cap,
        f"{gpu.free_mib:,}",
        f"{gpu.total_mib:,}",
    )
    log.info(
        "host     vLLM %s, nvcc %s, WSL %s",
        host["vllm"],
        "yes" if host["nvcc"] else "no",
        "yes" if host["wsl"] else "no",
    )
    log.info(
        "model    %s, %.2f GiB weights, %d KiB KV cache per token; gpu_memory_utilization %.2f",
        ctx.model,
        ctx.base_bytes / GIB,
        kv_bytes_per_token(ctx.shape) // 1024,
        ctx.util,
    )
    for c in ctx.checkpoints:
        log.info(
            "found    %s@%s (%s, %s-bit, %.2f GiB)",
            c.repo,
            (c.revision or "")[:8],
            c.method,
            c.bits,
            c.weight_bytes / GIB,
        )
    for cand in screen_candidates(ctx):
        reason = infeasible_reason(cand, ctx)
        log.info("plan     %-55s %s", cand.label(), f"skip: {reason}" if reason else "run")


def _prepare(run, ctx, args) -> None:
    from huggingface_hub import snapshot_download

    from . import data
    from .search import CONFIRM, POOL_SIZE
    from .space import infeasible_reason, screen_candidates

    runnable = {c.weights for c in screen_candidates(ctx) if not infeasible_reason(c, ctx)}
    repos = {ctx.model: ctx.revision} | {c.repo: c.revision for c in ctx.checkpoints if c.repo in runnable}
    for repo, revision in sorted(repos.items()):
        log.info("download %s@%s", repo, (revision or "")[:8])
        snapshot_download(repo, revision=revision, ignore_patterns=IGNORE_FILES)
    log.info("data     tokenizing %s", args.text or "WikiText-2")
    built = data.build(ctx.model, ctx.revision, args.text, CONFIRM["eval_tokens"], POOL_SIZE)
    (run.path / "data.json").write_text(json.dumps(built))


def cmd_run(args: argparse.Namespace) -> None:
    from .probe import child_env
    from .report import recommendation, render
    from .runner import HARNESS_SHA, Run
    from .search import Goal, Searcher
    from .space import Context

    if args.dry_run:
        _setup_logging(None)
        _log_plan(_build_context(args))
        return

    slug = args.model.replace("/", "--")
    out = args.out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    run = Run(out)
    _setup_logging(out / "dyno.log")
    if (out / "run.json").exists():
        meta = run.meta
        if meta["context"]["model"] != args.model:
            raise SystemExit(f"{out} holds a run for {meta['context']['model']}")
        if meta["harness_sha"] != HARNESS_SHA:
            raise SystemExit(f"the measurement code changed since {out} started; start a new run")
        ctx, goal = Context.from_json(meta["context"]), Goal(**meta["goal"])
        log.info("resume   %s, %d trials recorded", out, len(run.records))
    else:
        from .probe import package_version

        if package_version("vllm") is None:
            raise SystemExit("vLLM is not installed; install vllm-dyno[vllm] or run `uv sync --extra vllm`")
        ctx = _build_context(args)
        goal = Goal(args.goal, args.max_kl, args.max_ttft_ms, args.max_tpot_ms)
        _log_plan(ctx)
        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "argv": sys.argv[1:],
            "context": ctx.to_json(),
            "goal": asdict(goal),
            "gpu_price": args.gpu_price,
            "harness_sha": HARNESS_SHA,
            "free_mib_at_start": ctx.gpu.free_mib,
            "text": str(args.text) if args.text else "WikiText-2",
        }
        run.write_meta(meta)
        _prepare(run, ctx, args)

    searcher = Searcher(run, ctx, goal, env=child_env(ctx.host["nvcc"]), timeout_s=args.trial_timeout)
    searcher.search(confirm_top=args.confirm_top, max_combos=args.max_combos)
    (out / "report.md").write_text(render(run.meta, run.records, out))
    best = recommendation(run.records, goal)
    if best:
        log.info("best     %s", best["label"])
        log.info("serve    dyno serve %s", out)
    log.info("report   %s", out / "report.md")


def cmd_report(args: argparse.Namespace) -> None:
    from .report import render
    from .runner import Run

    run = Run(args.run_dir)
    text = render(run.meta, run.records, args.run_dir)
    (args.run_dir / "report.md").write_text(text)
    print(text)


def cmd_serve(args: argparse.Namespace) -> None:
    import os
    import shlex

    from .probe import child_env, host_facts
    from .report import recommendation
    from .runner import Run
    from .search import Goal, candidate_of
    from .space import Context, serve_command

    run = Run(args.run_dir)
    best = recommendation(run.records, Goal(**run.meta["goal"]))
    if best is None:
        raise SystemExit(f"{args.run_dir} has no recommendation; run `dyno run` to completion first")
    argv = shlex.split(serve_command(candidate_of(best), Context.from_json(run.meta["context"]))) + args.vllm_args
    print(shlex.join(argv), file=sys.stderr)
    os.execvpe(argv[0], argv, child_env(host_facts()["nvcc"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dyno", description="Measure vLLM serving configs on this GPU and recommend one."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('vllm-dyno')}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="search configs for a model and write a report")
    run.add_argument("model", help="Hugging Face repo id of the base model")
    run.add_argument(
        "--goal",
        choices=["throughput", "latency"],
        default="throughput",
        help="throughput: most batch tokens/s; latency: least time per output token for one user",
    )
    run.add_argument("--max-kl", type=float, default=0.05, help="largest KL divergence from the baseline to accept")
    run.add_argument("--max-ttft-ms", type=float, help="reject configs whose time to first token is higher")
    run.add_argument("--max-tpot-ms", type=float, help="reject configs whose time per output token is higher")
    run.add_argument("--gpu-price", type=float, help="USD per GPU hour; adds cost per 1M output tokens to the report")
    run.add_argument(
        "--checkpoint", action="append", default=[], help="quantized checkpoint repo to include (repeatable)"
    )
    run.add_argument("--no-discover", action="store_true", help="don't look for quantized checkpoints on the Hub")
    run.add_argument(
        "--any-owner",
        action="store_true",
        help="also discover checkpoints published by accounts other than the base model's owner",
    )
    run.add_argument(
        "--max-checkpoints", type=int, default=3, help="discovered checkpoints to try, one per method and bit width"
    )
    run.add_argument("--text", type=Path, help="text file for quality and speed prompts instead of WikiText-2")
    run.add_argument("--max-model-len", type=int, default=2048)
    run.add_argument("--confirm-top", type=int, default=3, help="configs re-measured in the confirm stage")
    run.add_argument("--max-combos", type=int, default=6, help="combinations tried in the combine stage")
    run.add_argument("--trial-timeout", type=float, default=900, help="seconds before a trial is killed")
    run.add_argument("--out", type=Path, help="run directory; pass an existing one to resume it")
    run.add_argument("--dry-run", action="store_true", help="print the plan without loading any model on the GPU")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="rewrite and print report.md for a run directory")
    report.add_argument("run_dir", type=Path)
    report.set_defaults(func=cmd_report)

    serve = sub.add_parser("serve", help="start vllm serve with a run's recommended config and its environment")
    serve.add_argument("run_dir", type=Path)
    serve.add_argument(
        "vllm_args", nargs=argparse.REMAINDER, help="extra arguments passed to vllm serve, e.g. --port 8001"
    )
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    args.func(args)
