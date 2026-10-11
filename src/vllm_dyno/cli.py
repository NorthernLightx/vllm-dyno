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
    util = memory_utilization(gpu, wsl=host["wsl"])
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
    from .load import load_goal, load_records
    from .probe import child_env, measured_env
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
        env = child_env(ctx.host, ctx.util)
        if meta.get("env") != measured_env(env, ctx.host, ctx.util):
            raise SystemExit(
                f"{out} was measured with {meta.get('env')} and vLLM would now run with "
                f"{measured_env(env, ctx.host, ctx.util)}; start a new run"
            )
        log.info("resume   %s, %d trials recorded", out, len(run.records))
    else:
        from .probe import package_version

        if package_version("vllm") is None:
            raise SystemExit("vLLM is not installed; install vllm-dyno[vllm] or run `uv sync --extra vllm`")
        ctx = _build_context(args)
        env = child_env(ctx.host, ctx.util)
        goal = Goal(args.goal, args.max_kl, args.max_ttft_ms, args.max_tpot_ms)
        _log_plan(ctx)
        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "argv": sys.argv[1:],
            "context": ctx.to_json(),
            "goal": asdict(goal),
            "gpu_price": args.gpu_price,
            "harness_sha": HARNESS_SHA,
            "env": measured_env(env, ctx.host, ctx.util),
            "free_mib_at_start": ctx.gpu.free_mib,
            "text": str(args.text) if args.text else "WikiText-2",
        }
        run.write_meta(meta)
        _prepare(run, ctx, args)

    searcher = Searcher(run, ctx, goal, env=env, timeout_s=args.trial_timeout)
    searcher.search(confirm_top=args.confirm_top, max_combos=args.max_combos)
    (out / "report.md").write_text(render(run.meta, run.records, out, load_records(out), load_goal(out)))
    best = recommendation(run.records, goal)
    if best:
        log.info("best     %s", best["label"])
        log.info("serve    dyno serve %s", out)
    log.info("report   %s", out / "report.md")


def _trial_env(meta: dict) -> dict[str, str | None]:
    from .probe import child_env, measured_env

    host, util = meta["context"]["host"], meta["context"]["util"]
    return measured_env(child_env(host, util), host, util)


def cmd_load(args: argparse.Namespace) -> None:
    from .load import LOAD_SHA, LoadSession, Slo, load_goal, load_records, prompt_pool, targets
    from .probe import child_env, measured_env, query_gpu
    from .report import render
    from .runner import HARNESS_SHA, Run
    from .search import Goal
    from .space import Context

    run = Run(args.run_dir)
    if not (args.run_dir / "run.json").exists():
        raise SystemExit(f"{args.run_dir} is not a run directory")
    if not args.config and (run.meta["harness_sha"] != HARNESS_SHA or run.meta.get("env") != _trial_env(run.meta)):
        raise SystemExit(
            f"the offline measurement code or environment changed since {args.run_dir} started, and combinations "
            "need quality trials comparable with its others; start a new run, or name configs with --config"
        )
    _setup_logging(args.run_dir / "dyno.log")
    ctx = Context.from_json(run.meta["context"])
    env = child_env(ctx.host, ctx.util)
    slo = Slo(args.slo_ttft_ms, args.slo_tpot_ms, args.slo_share)
    settings = {
        "slo": asdict(slo),
        "probe_s": args.probe_seconds,
        "load_sha": LOAD_SHA,
        "env": measured_env(env, ctx.host, ctx.util),
    }
    meta_path = args.run_dir / "load.json"
    gpu = query_gpu()
    free_mib = gpu.free_mib
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if {k: meta.get(k) for k in settings} != settings:
            raise SystemExit(
                f"{meta_path} was measured with other SLO settings, environment or load code; "
                "delete load.json and load.jsonl to measure again"
            )
        free_mib = meta["free_mib_at_start"]
    else:
        meta = {**settings, "created": time.strftime("%Y-%m-%d %H:%M"), "free_mib_at_start": free_mib}
    meta_path.write_text(json.dumps(meta | {"goal": args.goal}, indent=2))
    if gpu.free_mib < ctx.util * gpu.total_mib:
        raise SystemExit(
            f"{gpu.free_mib:,} MiB GPU memory is free; this run's gpu_memory_utilization {ctx.util} "
            f"needs {ctx.util * gpu.total_mib:,.0f}"
        )

    pool = prompt_pool(json.loads((args.run_dir / "data.json").read_text()))
    session = LoadSession(run, ctx, pool, slo, args.probe_seconds, env, args.server_timeout, free_mib)
    if args.config:
        session.baseline()
        for r in targets(run.records, args.config):
            if r["label"] != "baseline":
                session.measure(r)
    else:
        session.search(Goal(**run.meta["goal"]), args.max_combos, args.trial_timeout)
    text = render(run.meta, run.records, args.run_dir, load_records(args.run_dir), load_goal(args.run_dir))
    (args.run_dir / "report.md").write_text(text)
    log.info("report   %s", args.run_dir / "report.md")


def cmd_report(args: argparse.Namespace) -> None:
    from .load import load_goal, load_records
    from .report import render
    from .runner import Run

    run = Run(args.run_dir)
    text = render(run.meta, run.records, args.run_dir, load_records(args.run_dir), load_goal(args.run_dir))
    (args.run_dir / "report.md").write_text(text)
    print(text)


def cmd_serve(args: argparse.Namespace) -> None:
    import os
    import shlex

    from .load import load_goal, load_records
    from .probe import child_env, host_facts
    from .report import load_recommendation, recommendation
    from .runner import Run
    from .search import Goal, candidate_of
    from .space import Context, serve_command

    run = Run(args.run_dir)
    goal = Goal(**run.meta["goal"])
    under_load = load_recommendation(load_records(args.run_dir), run.records, goal, load_goal(args.run_dir))
    best = under_load[0] if under_load else recommendation(run.records, goal)
    if best is None:
        raise SystemExit(f"{args.run_dir} has no recommendation; run `dyno run` to completion first")
    ctx = Context.from_json(run.meta["context"])
    argv = shlex.split(serve_command(candidate_of(best), ctx)) + args.vllm_args
    print(shlex.join(argv), file=sys.stderr)
    os.execvpe(argv[0], argv, child_env(host_facts(), ctx.util))


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

    load = sub.add_parser(
        "load", help="measure a run's settings and their combinations under online load through vllm serve"
    )
    load.add_argument("run_dir", type=Path)
    load.add_argument(
        "--config",
        action="append",
        default=[],
        help="trial label to load instead of searching (repeatable); the baseline is always loaded",
    )
    load.add_argument(
        "--goal", choices=["chat", "batch"], default="chat", help="traffic profile to recommend a config for"
    )
    load.add_argument("--max-combos", type=int, default=8, help="combinations of gaining settings to load")
    load.add_argument(
        "--trial-timeout", type=float, default=900, help="seconds before a combination's quality trial is killed"
    )
    load.add_argument("--slo-ttft-ms", type=float, default=1000, help="time to first token a request may take")
    load.add_argument("--slo-tpot-ms", type=float, default=50, help="time per output token a request may take")
    load.add_argument("--slo-share", type=float, default=0.9, help="share of requests that must meet both limits")
    load.add_argument("--probe-seconds", type=float, default=30, help="arrival window of each chat probe")
    load.add_argument("--server-timeout", type=float, default=600, help="seconds to wait for vllm serve to answer")
    load.set_defaults(func=cmd_load)

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
