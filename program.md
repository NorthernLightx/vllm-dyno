# Running vllm-dyno from a coding agent

This file is for an agent such as Claude Code or Codex that runs vllm-dyno for someone.

The person wants a `vllm serve` command for one model on the GPU in front of you. vllm-dyno measures the configs it can run there and recommends one. Your job is to start the measurements, keep them clean, and report what they found.

## Before anything runs

1. vLLM runs on Linux. On Windows, work inside WSL2.
2. Install with `uv sync --extra vllm` in the repository.
3. Run `nvidia-smi`. If another process holds the GPU, ask the person before going on: a second job on the GPU slows every server and the comparison is lost. Never kill a process you did not start.
4. Run `uv run dyno run <model> --dry-run`. It loads nothing on the GPU and prints the configs it would try, the ones it skips and why, and the memory budget it picked. Show the person the skips. A skip they did not expect, such as FlashInfer without nvcc or a checkpoint from another account, is cheaper to discuss now.

## Offline run

```
uv run dyno run <model> --out runs/<name>
```

Each config runs in a fresh vLLM process: its quality against the unmodified model (KL divergence on WikiText-2), then its speed. On an RTX 3070 this took 15 minutes for Qwen2.5-1.5B-Instruct and 40 for Qwen3-1.7B. If the run stops, the same command with the same `--out` resumes it.

Each line of `runs/<name>/trials.jsonl` is one trial. A status other than `ok` comes with the error vLLM gave, and the log is under `runs/<name>/trials/`.

## Under load

```
uv run dyno load runs/<name>
```

This starts `vllm serve` once per config and measures chat capacity, the highest arrival rate that meets the SLO, and batch throughput. Each server took 6 to 9 minutes on the RTX 3070 with these models. Every session measures the baseline first and compares each config against that measurement, because the same config measures differently from one session to the next.

`--planner` decides which configs get a server:

- Leave the default, `fit`. It serves one config at a time, each the likeliest to be recommended under a fitted model of what every setting does under load, and stops once no config left is likely to change the recommendation. With Qwen2.5-1.5B-Instruct, Qwen3-1.7B and SmolLM2-1.7B-Instruct it served 7, 8 and 4 configs and recommended what serving every config did.
- `--planner rules` serves every setting on its own, then combinations of those that gained. It needs no model and serves more configs: 13 against 8 for Qwen3-1.7B.
- `--planner llm` asks a language model. It needs an API key and sends the GPU, the model and the measurements to the provider, so ask the person first.
- `--planner all` serves every allowed config. For Qwen3-1.7B that was 27 servers in four and a half hours. Use it only when the person wants the whole space measured.

While `dyno load` runs:

- Keep the machine quiet. Batch throughput is sensitive to CPU load on the host: on the RTX 3070 with Qwen3-1.7B, other heavy jobs, agents included, cost configs 5 to 15% of their batch throughput. Reading and writing files is fine; building, testing or training anything is not, until the session ends.
- Do not change the code under `src/vllm_dyno/` during a session or between sessions you mean to compare. Every load record carries a hash of the load code (`load_sha`), and records with different hashes are not comparable.
- Watch for `degraded` in the log: a server that ran slower than its own config should. vllm-dyno starts it once more by itself.

## Reading the result

`runs/<name>/report.md` holds the recommendation, the `vllm serve` command, and the chat capacity, batch throughput and KL of every config served. Before you pass it on, check:

- Configs within 10% of the best count as tied, and among them the one with the lowest KL wins. The report lists the tied configs. If the person cares more about speed than about drift from the original model, tell them which tied config was fastest.
- Gains are against the baseline of the same session. Do not compare numbers from two sessions or two run directories.
- A config missing from the table failed offline, went over the KL limit, failed under load, or was not picked. `trials.jsonl`, `load.jsonl` and `plan.jsonl` say which.

Then start the recommendation and send one request through it:

```
uv run dyno serve runs/<name>
```

Whenever you report a speed or quality figure, give the GPU and the model with it.

## When something fails

| status | what happened | what to do |
|---|---|---|
| `oom` | the config does not fit in the memory budget | nothing; it is recorded and skipped |
| `unsupported` | vLLM refused the config on this machine | nothing; it is recorded and skipped |
| `crash` | the process died; the record has the last error line of its log | read the log; rerun only if the cause was outside the config, such as a network failure |
| `timeout` | no result within the time limit | read the log |
| `degraded` | the server ran slower than its config should | vllm-dyno restarts it once; if it degrades again, check the GPU for other processes |

If the whole command stops, run it again: it reuses the offline trials and load records already written.
