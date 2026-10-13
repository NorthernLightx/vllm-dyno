# vllm-dyno

vllm-dyno finds a vLLM serving config for your GPU and model. It measures how far each candidate config's output drifts from the unmodified model, then serves candidates within your quality limit through `vllm serve` under concurrent load and recommends the one that handled the most traffic. Serving every candidate takes hours, so by default vllm-dyno fits a model of what each setting does under load and serves only the configs that model says could still win.

Status: v0.3, tested on one machine (RTX 3070 8 GB, WSL2, vLLM 0.31) with Qwen2.5-1.5B-Instruct and Qwen3-1.7B.

```
uv run dyno run Qwen/Qwen2.5-1.5B-Instruct --out runs/qwen2.5-1.5b
uv run dyno load runs/qwen2.5-1.5b
```

```
10:04:10  GPU      NVIDIA GeForce RTX 3070, compute capability 8.6, 7,626 of 8,192 MiB free
10:04:10  model    Qwen/Qwen2.5-1.5B-Instruct, 2.88 GiB weights, 28 KiB KV cache per token; gpu_memory_utilization 0.74
10:04:10  found    Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8@a4d4d713 (gptq, 8-bit, 1.69 GiB)
10:04:10  plan     kv_cache_dtype=fp8                                      skip: fp8 KV cache below compute capability 8.9 needs FlashInfer, which compiles with nvcc (not on PATH)
...
10:16:44  confirm 2/4  weights=fp8                                             ok    3132 tok/s  TPOT   8.6 ms  TTFT    53 ms  KL 0.0076  top-1 0.952  (76 s)
10:19:15  confirm 4/4  attention_backend=TRITON_ATTN                           ok    2857 tok/s  TPOT  11.3 ms  TTFT    53 ms  KL 0.0007  top-1 0.982  (74 s)
10:19:15  best     weights=fp8
...
11:04:37  load     baseline                                                batch 512 requests: 2356 tok/s, TPOT p50 82.3 ms, KV cache peak 100%, 72 preemptions
11:05:15  plan     fit picks attention_backend=TRITON_ATTN, expecting chat 1.00x, batch 1.00x: Settles attention_backend against the baseline and wins 21% of 256 draws of the fitted gains: chat 1.00x (90% interval 0.66x to 1.51x), ...
11:12:09  plan     attention_backend=TRITON_ATTN: chat 21.67 req/s (1.24x), batch 2623 tok/s (1.11x), KL 0.0006
11:12:09  plan     fit picks weights=Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8 attention_backend=TRITON_ATTN, expecting chat 1.27x, batch 1.14x: ...
...
11:56:06  plan     fit stops: No config left, loaded next, would be recommended in 10% of 256 draws of the fitted gains; the likeliest, kv_cache_dtype=int8_per_token_head attention_backend=TRITON_ATTN, would in 2%.
```

`dyno run` took 15 minutes to measure quality and speed in vLLM's offline engine, where fp8 weights were the fastest config within the default KL limit of 0.05. `dyno load` then measured the baseline under load and served seven of the 29 configs the offline run allowed, one at a time, in 58 minutes:

| config | attention | expected chat | chat req/s | batch tok/s | KL |
|---|---|---|---|---|---|
| baseline (bf16) | FLASH_ATTN | | 17.45 | 2,356 | 0 |
| TRITON_ATTN attention backend | TRITON_ATTN | 1.00x | 21.67 (1.24x) | 2,623 | 0.0007 |
| GPTQ-Int8 + TRITON_ATTN (recommended) | TRITON_ATTN | 1.27x | 20.75 (1.19x) | 2,883 | 0.0008 |
| GPTQ-Int8 + int8 KV cache + TRITON_ATTN | TRITON_ATTN | 1.18x | 20.75 (1.19x) | 2,889 | 0.0130 |
| GPTQ-Int8 + TRITON_ATTN + ngram speculative decoding | TRITON_ATTN | 0.99x | 14.67 (0.84x) | 2,098 | 0.0008 |
| fp8 weights + TRITON_ATTN | TRITON_ATTN | 1.29x | 21.67 (1.24x) | 2,919 | 0.0069 |
| fp8 weights + int8 KV cache + TRITON_ATTN | TRITON_ATTN | 1.25x | 20.75 (1.19x) | 2,946 | 0.0200 |
| turboquant KV cache (`turboquant_k8v4`) | FLASH_ATTN+TURBOQUANT | 0.95x | 17.45 (1.00x) | 2,451 | 0.0006 |

Chat capacity is the highest arrival rate at which 90% of requests got their first token within 1 s and every later token within 50 ms on average, with 256-token prompts and 128-token answers arriving at random. Batch is 512 such requests sent at once. Requests sample with the defaults in the model's `generation_config.json`, which for Qwen2.5 are temperature 0.7, top-k 20, top-p 0.8 and a repetition penalty of 1.1. "Expected chat" is the planner's prediction before it served the config.

Offline, TRITON_ATTN ran at 0.99x the baseline (2,857 against 2,872 tok/s); under load it served 24% more chat traffic. On this GPU, FlashAttention 2 ran a batch of decodes three to six times slower once a prefill joined it. A server adds each new request's prefill to the batch of running decodes, while the offline throughput test prefills every request first and then only decodes. The planner served TRITON_ATTN first anyway: it serves each setting at least once before it may stop, and its prediction for TRITON_ATTN, taken from the offline result, was uncertain enough that TRITON_ATTN still won a fifth of the draws.

Five configs reached 20.75 or 21.67 req/s, one bisection step apart, so chat capacity does not separate them. Among those also within 10% of the best batch throughput, vllm-dyno recommends the lowest KL: the GPTQ-Int8 checkpoint with TRITON_ATTN, at 1.19x the baseline's chat capacity and 1.22x its batch throughput. Serving every allowed config gave the same recommendation.

`dyno serve <run directory>` starts `vllm serve` with the recommended config and the environment the trials ran in.

## Install

vLLM runs on Linux; on Windows, use WSL2. With [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/NorthernLightx/vllm-dyno
cd vllm-dyno
uv sync --extra vllm
uv run dyno run <model> --dry-run    # the plan, without loading anything on the GPU
```

[program.md](program.md) walks a coding agent such as Claude Code or Codex through a whole run.

## How it works

Numbers in this section were measured on the RTX 3070 with Qwen2.5-1.5B-Instruct unless they name another model.

Before loading anything, vllm-dyno reads the GPU's free memory and compute capability, checks for nvcc, works out the model's weight size and KV cache bytes per token, and looks up the most downloaded quantized checkpoints of the model on the Hugging Face Hub, one per method and bit width. Only checkpoints published by the base model's owner count unless you pass `--any-owner`, since the recommended weights end up in production. Every model is pinned to the commit measured, and the recommended `vllm serve` command carries that `--revision`. vllm-dyno never runs code from a model repo (`trust_remote_code` stays off). It skips configs known to fail on this machine and logs the reason.

### Offline search

The search space has four settings: weights (as published, fp8 at load time, or a quantized checkpoint), KV cache dtype, attention backend, and ngram speculative decoding. `dyno run` searches it offline in three stages:

1. Screen: the baseline, then one changed setting at a time.
2. Combine: settings that each beat the baseline by 5% or more, tried together.
3. Confirm: the best three and the baseline again, on more tokens and with repeated speed runs.

Each trial starts a fresh vLLM engine in its own process and measures:

- quality on 4,096 tokens of WikiText-2 (16,384 when confirming): KL divergence from the baseline's next-token distribution over its top 20 tokens, how often both rank the same token first, and perplexity;
- time to first token and time per output token for one request with a 512-token prompt;
- throughput of 64 requests with 200-token prompts and 128 output tokens each;
- KV cache capacity in tokens.

Every trial is a line in `runs/<run>/trials.jsonl`, with its vLLM log in `runs/<run>/trials/`. To resume an interrupted run, pass the same `--out`.

### Under load

`dyno load <run directory>` starts `vllm serve` once per config, with vLLM's serving defaults, and runs both traffic profiles against it. It finds chat capacity by doubling the rate from 2 req/s until a probe fails, then bisecting to within 5%. A probe stops as soon as so many requests have missed the SLO that it can no longer pass. Every request carries its own `cache_salt`, so prompts that repeat do not hit the prefix cache.

Each `dyno load` measures the baseline first and compares every config against that measurement, since results drift between sessions: the baseline's chat capacity has differed by 9%, more than fp8 or GPTQ-Int8 weights changed it. The configs it may serve are every combination of settings that each stayed within the quality limit offline. A combination gets an offline trial for its KL before its server starts. A config that sets the attention backend is skipped when the same config without that setting was loaded and the server picked that backend anyway: the int8 KV cache runs on TRITON_ATTN either way. Results go to `runs/<run>/load.jsonl`, with each server's log in `runs/<run>/load/`.

### Stalls and slowdowns

Before measuring a server, `dyno load` sends bursts of short requests at every batch size up to 64 and a few above it. vLLM compiles its Triton top-k/top-p sampling kernels the first time each range of batch sizes needs them, and its own warmup leaves them out. Each round of compiles holds the server up for 2.7 to 4.2 seconds, long enough to fail a chat probe. A server whose start compiled the model's graph is started a second time and measured from vLLM's compile cache: vLLM sizes the KV cache from the memory peak of its start, and a start that compiles reports a higher peak. With Qwen3-1.7B, a compiling start left 7,888 tokens of KV cache and a cached start 12,528.

Two checks catch a server that runs slower than its config should. Before the search, two requests sent one at a time and decoded greedily must not run more than 1.5 times slower per token than the offline trial of the same config. Speculative configs skip this check, since how fast they decode depends on the text. After the batch profile, a probe at 2 req/s must not run more than 1.2 times slower than the first probe did. A server that fails either check is marked degraded and started once more, and a config degraded twice is left out of the comparison. The failing rate at which the search stops is probed a second time, so a single stall does not set the capacity.

Under WSL, the Windows graphics driver moves GPU memory into system memory instead of refusing an allocation, and PyTorch only frees its cache of unused blocks after an allocation fails. A server whose cache grows past the GPU then slows down without an error: with 2 GiB spilled, TPOT p90 at 16 req/s was 72 ms against 25 ms with the cache capped. vllm-dyno caps PyTorch's allocator at `gpu_memory_utilization` (`PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction`) for trials and servers, and leaves 1.5 GiB of GPU memory for the Windows desktop to grow into. The recommended command carries the same setting.

### Choosing which configs to serve

`--planner` picks how `dyno load` chooses the next config to serve:

- `fit`, the default, fits a model of each setting's gain under load and serves the config likeliest to be recommended.
- `rules` serves every setting on its own, then up to eight combinations of the settings that gained 10% or more on chat capacity or batch throughput.
- `llm` asks a language model through any OpenAI-compatible endpoint.
- `all` serves every allowed config, which measures the whole space.

Whichever planner chose them, the recommendation follows the same rule over the configs served. Each pick goes into `runs/<run>/plan.jsonl` and the report, with the planner's reasoning and its expected and measured gains.

The fit planner models a config's log chat gain, and its log batch gain, as a sum of one weight per value the config changes from the baseline. Each weight starts from half the log offline throughput gain of its value, since offline speed misjudges some settings under load, and every server's measurement updates the weights by Bayesian linear regression. vLLM picks the attention backend by the KV cache format, so a config that leaves the backend to vLLM counts as running the backend that servers with its KV cache format picked. At each step the planner draws 256 sets of gains from the fit, applies the recommendation rule to each, and serves the config that the most draws recommend.

No planner may stop before the effect of every setting it can change is known: two configs that served, or one and the baseline, must differ in that setting alone, or the one with the setting must have failed to serve. Planners that judge by offline speed otherwise stop before serving the attention backend. Once every effect is known, the fit planner stops when no config left would be recommended in 10% of the draws if it were served next.

#### How the planners compare

To compare planners on the same footing, `--planner all` served every allowed config of both models; that set of measurements is the ground truth. `dyno replay` then runs a planner many times against those records instead of GPU servers: each server it asks for is answered with a recorded measurement of that config, scaled to a recorded baseline. A replayed session recommends by the same rule as a real one, and is scored by whether its pick matches the recommendation over every config's mean gains, or falls among the configs tied with it (within 10% on both profiles). GPU minutes add the servers' and the offline trials' wall time.

Over 100 replayed sessions per planner, with chat as the goal:

| Qwen2.5-1.5B-Instruct (the full sweep) | servers | GPU minutes | same pick as the truth | tied with the truth |
|---|---|---|---|---|
| rules | 10.0 | 81 | 1.00 | 1.00 |
| fit | 6.1 | 53 | 1.00 | 1.00 |
| fit, 5 servers | 5.0 | 45 | 0.98 | 1.00 |
| Optuna TPE, 10 servers | 10.0 | 88 | 0.98 | 1.00 |
| Optuna TPE, 5 servers | 5.0 | 45 | 0.59 | 1.00 |
| random, 10 servers | 10.0 | 89 | 0.37 | 1.00 |
| random, 5 servers | 5.0 | 48 | 0.16 | 0.92 |

| Qwen3-1.7B (8 sessions, 1 a full sweep) | servers | GPU minutes | same pick as the truth | tied with the truth |
|---|---|---|---|---|
| rules | 13.2 | 106 | 0.03 | 0.94 |
| fit | 8.7 | 81 | 0.34 | 0.95 |
| fit, 5 servers | 5.0 | 49 | 0.13 | 0.80 |
| Optuna TPE, 10 servers | 10.0 | 98 | 0.15 | 0.99 |
| Optuna TPE, 5 servers | 5.0 | 52 | 0.09 | 0.99 |
| random, 10 servers | 10.0 | 98 | 0.12 | 0.90 |
| random, 5 servers | 5.0 | 52 | 0.06 | 0.74 |

Servers do not count the baseline. The Optuna planner serves the three configs with the best offline estimate first, then lets TPE choose. On Qwen3-1.7B, whose servers run out of KV cache before anything else, the configs fastest offline are also the fastest under load, so TPE's offline-first start is enough. Eight other configs are tied with the truth there, and which of them a session ends on decides the "same pick" column. On Qwen2.5-1.5B, the gain comes from TRITON_ATTN, which offline speed does not show, and the fit planner found the truth's config in every session with fewer servers than the rules.

Real sessions, each on a copy of the same offline run, measured by the same load code:

| model | planner | servers | minutes | recommended | chat | batch | KL |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | fit | 7 | 58 | GPTQ-Int8 + TRITON_ATTN | 1.19x | 1.22x | 0.0008 |
| Qwen2.5-1.5B | all | 20 | 207 | GPTQ-Int8 + TRITON_ATTN | 1.19x | 1.23x | 0.0008 |
| Qwen3-1.7B | fit | 8 | 68 | GPTQ-Int8 + int8 KV cache | 1.68x | 1.49x | 0.0292 |
| Qwen3-1.7B | rules | 13 | 110 | GPTQ-Int8 + int8 KV cache | 1.61x | 1.50x | 0.0292 |
| Qwen3-1.7B | rules | 13 | 97 | GPTQ-Int8 + int8 KV cache | 1.68x | 1.50x | 0.0292 |
| Qwen3-1.7B | Nemotron 3 Super | 4 | 45 | fp8 + int8 KV cache | 1.68x | 1.56x | 0.0378 |
| Qwen3-1.7B | Nemotron 3 Super | 6 | 55 | fp8 + int8 KV cache | 1.68x | 1.66x | 0.0378 |
| Qwen3-1.7B | Claude Opus 5.5 | 6 | 63 | fp8 + TRITON_ATTN | 1.48x | 1.37x | 0.0111 |
| Qwen3-1.7B | Claude Opus 5.5 | 6 | 62 | GPTQ-Int8 + int8 KV cache + TRITON_ATTN | 1.68x | 1.48x | 0.0270 |
| Qwen3-1.7B | all | 27 | 270 | GPTQ-Int8 + int8 KV cache | 1.61x | 1.51x | 0.0292 |

Servers do not count the baseline; minutes are wall time. Chat and batch are multiples of each session's own baseline, so one bisection step of the baseline moves a ratio by 4 to 9%. The Qwen2.5 sweep ran in two sessions; its minutes leave out the pause between them.

The LLM planner sends the model the GPU and model, the SLO, every offline trial, every config served so far with its gains, its own earlier picks next to what they measured, and the configs it may pick. It answers with a config, the gains it expects and why, or with a stop. If the endpoint fails, or the model twice names a config outside the list, the rules finish the session and reuse every config already served. By default the endpoint is Nemotron 3 Super on OpenRouter's free tier, with the API key in `OPENROUTER_API_KEY`:

```
uv run dyno load runs/<run> --planner llm
uv run dyno load runs/<run> --planner llm --planner-url https://api.anthropic.com/v1 --planner-model claude-opus-5-5 --planner-key-env ANTHROPIC_API_KEY
```

vllm-dyno takes the key out of its own environment before it starts anything, so neither vLLM nor the trials inherit it, and it writes the key nowhere.

`dyno replay` replays any planner over finished runs measured the same way: one model, GPU, memory budget, vLLM version, offline reference and load code. The Optuna planner needs the `bench` extra (`uv sync --extra bench`):

```
uv run dyno replay runs/<sweep> runs/<other session> --planner fit --repeats 100
```

## Options

`dyno run --help`, `dyno load --help` and `dyno replay --help` list all options. The ones most runs need:

- `--max-kl` sets the quality limit (default 0.05).
- `dyno run --goal latency` ranks configs by time per output token instead of throughput.
- `dyno run --max-ttft-ms` and `--max-tpot-ms` reject configs that are slower than a latency target.
- `dyno load --goal batch` recommends the config with the highest batch throughput instead of chat capacity.
- `dyno load --slo-ttft-ms`, `--slo-tpot-ms` and `--slo-share` set the SLO (defaults 1000 ms, 50 ms and 90% of requests).
- `dyno load --config <label>` loads the named configs instead of searching.
- `dyno load --planner` picks the planner (`fit`, `rules`, `llm` or `all`); `--max-servers` caps the servers the fit and LLM planners start (default 10). `--planner-url`, `--planner-model` and `--planner-key-env` pick the LLM planner's endpoint, model and the variable that holds its API key.
- `--gpu-price` adds cost per million output tokens, given USD per GPU hour.
- `--text` measures quality and speed on your own text instead of WikiText-2.
- `--checkpoint` adds a quantized checkpoint that discovery did not pick.
- `--any-owner` lets discovery consider checkpoints from other accounts.

## Limitations

- Load traffic is synthetic: 256-token prompts and 128-token answers, with no shared prefixes.
- Requests sample with the model's own defaults. Clients that send other sampling parameters get different capacity: on the RTX 3070 with GPTQ-Int8 and TRITON_ATTN, greedy requests reached 21.67 req/s and Qwen2.5's defaults 19.87.
- Without nvcc on PATH, as under WSL, FlashInfer is unavailable: its attention backend, fp8 KV cache below compute capability 8.9, and its sampler, so vLLM samples with Triton kernels instead.
- Chat capacity comes in bisection steps of 4 to 9%, and each config is served once per session, so configs within 10% of each other count as a tie.
- `max_num_seqs` and `max_num_batched_tokens` stay at vLLM's serving defaults, 256 and 2,048 on GPUs under 70 GiB. Raising them did not help on the RTX 3070. With the GPTQ-Int8 checkpoint and int8 KV cache, the batch profile used 38% of the cache, and `--max-num-seqs 512` lowered batch throughput from 2,850 to 2,513 tok/s. With GPTQ-Int8 and TRITON_ATTN, `--max-num-batched-tokens 4096` left batch throughput level and raised TPOT p90 at 21.67 req/s from 51 to 54 ms.
- KL is computed over the baseline's top 20 tokens and reads slightly low.
- The planners were compared on two models on one GPU. The fit planner's starting weights and noise were not tuned on other models, and the LLM planner's endpoint samples with its own defaults, so two of its sessions can pick differently.
- The LLM planner sends the GPU and model names and the measurements to its endpoint; check the provider's data policy. Free models on OpenRouter are rate limited.
- One GPU, no tensor parallelism.
- Tested only on an RTX 3070 with vLLM 0.31.

## License

Apache-2.0
