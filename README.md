# vllm-dyno

vllm-dyno finds a vLLM serving config for your GPU and model. It measures how far each candidate config's output drifts from the unmodified model, serves the candidates within your quality limit through `vllm serve` under concurrent load, and recommends the one that handled the most traffic. A language model can choose which configs to serve, through any OpenAI-compatible endpoint.

Status: v0.2 in progress, tested on one machine (RTX 3070 8 GB, WSL2, vLLM 0.31).

```
uv run dyno run Qwen/Qwen2.5-1.5B-Instruct --out runs/qwen2.5-1.5b-1011c
uv run dyno load runs/qwen2.5-1.5b-1011c
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
13:06:12  load     baseline                                                lone request: TPOT 11.8 ms, 11.7 ms offline
13:10:16  load     baseline                                                chat  16.71 req/s: 100% within SLO, TTFT p90 216 ms, TPOT p90 47.4 ms, KV cache peak 42%, 0 preemptions
13:11:06  load     baseline                                                batch 512 requests: 2308 tok/s, TPOT p50 83.1 ms, KV cache peak 100%, 72 preemptions
...
13:43:06  load     attention_backend=TRITON_ATTN                           chat  20.75 req/s: 100% within SLO, TTFT p90 204 ms, TPOT p90 42.2 ms, KV cache peak 48%, 0 preemptions
...
14:17:58  report   runs/qwen2.5-1.5b-1011c/report.md
```

`dyno run` took 15 minutes to measure quality and speed in vLLM's offline engine, where fp8 weights were the fastest config within the default KL limit of 0.05. `dyno load` then served the configs through `vllm serve`, which took 73 minutes:

| config | attention | chat req/s | batch tok/s | KL |
|---|---|---|---|---|
| baseline (bf16) | FLASH_ATTN | 16.71 | 2,308 | 0 |
| fp8 weights, quantized at load | FLASH_ATTN | 16.71 | 2,676 | 0.0076 |
| Qwen2.5-1.5B-Instruct-GPTQ-Int8 | FLASH_ATTN | 16.71 | 2,623 | 0.0006 |
| TRITON_ATTN attention backend | TRITON_ATTN | 20.75 | 2,561 | 0.0007 |
| int8 KV cache (`int8_per_token_head`) | TRITON_ATTN | 20.75 | 2,836 | 0.0132 |
| turboquant KV cache (`turboquant_k8v4`) | FLASH_ATTN+TURBOQUANT | 16.71 | 2,389 | 0.0006 |
| ngram speculative decoding | FLASH_ATTN | 11.31 | 1,506 | 0 |
| fp8 weights + TRITON_ATTN | TRITON_ATTN | 20.75 | 2,885 | 0.0069 |
| fp8 weights + int8 KV cache | TRITON_ATTN | 20.75 | 2,876 | 0.0200 |
| GPTQ-Int8 + TRITON_ATTN (recommended) | TRITON_ATTN | 20.75 | 2,817 | 0.0008 |
| GPTQ-Int8 + int8 KV cache | TRITON_ATTN | 20.75 | 2,834 | 0.0130 |

Chat capacity is the highest arrival rate at which 90% of requests got their first token within 1 s and every later token within 50 ms on average, with 256-token prompts and 128-token answers arriving at random. Batch is 512 such requests sent at once. Requests sample with the defaults in the model's `generation_config.json`, which for Qwen2.5 are temperature 0.7, top-k 20, top-p 0.8 and a repetition penalty of 1.1.

Offline, TRITON_ATTN ran at 0.99x the baseline (2,857 against 2,872 tok/s); under load it served 24% more chat traffic. On this GPU, FlashAttention 2 ran a batch of decodes three to six times slower once a prefill joined it. A server adds each new request's prefill to the batch of running decodes, while the offline throughput test prefills every request first and then only decodes. The int8 KV cache runs on TRITON_ATTN as well and reached the same chat capacity.

Six configs reached 20.75 req/s. At the next rate tried, 21.67 req/s, all six missed the 50 ms limit twice, with TPOT p90 between 51 and 55 ms, so chat capacity does not separate them. Among those also within 10% of the best batch throughput, vllm-dyno recommends the lowest KL: the GPTQ-Int8 checkpoint with TRITON_ATTN, at 1.24x the baseline's chat capacity and 1.22x its batch throughput.

`dyno serve <run directory>` starts `vllm serve` with the recommended config and the environment the trials ran in.

## Install

vLLM runs on Linux; on Windows, use WSL2. With [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/NorthernLightx/vllm-dyno
cd vllm-dyno
uv sync --extra vllm
uv run dyno run <model> --dry-run    # the plan, without loading anything on the GPU
```

## How it works

Numbers in this section were measured on the RTX 3070 with Qwen2.5-1.5B-Instruct.

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

Each `dyno load` measures the baseline first and compares every config against that measurement, since results drift between sessions: the baseline's chat capacity has differed by 9%, more than fp8 or GPTQ-Int8 weights changed it. By default, `dyno load` serves every setting the offline run kept within the quality limit, including those that gained less than 5% offline, like TRITON_ATTN at 1.00x in the screen. Then it serves up to eight combinations of the settings that gained 10% or more on chat capacity or batch throughput, each after an offline trial for its KL. A combination that sets the attention backend is skipped when the same config without that setting was loaded and the server picked that backend anyway: int8 KV cache runs on TRITON_ATTN either way. Results go to `runs/<run>/load.jsonl`, with each server's log in `runs/<run>/load/`.

### Stalls and slowdowns

Before measuring a server, `dyno load` sends bursts of short requests at every batch size up to 64 and a few above it. vLLM compiles its Triton top-k/top-p sampling kernels the first time each range of batch sizes needs them, and its own warmup leaves them out. Each round of compiles holds the server up for 2.7 to 4.2 seconds, long enough to fail a chat probe.

Two checks catch a server that runs slower than its config should. Before the search, two requests sent one at a time and decoded greedily must not run more than 1.5 times slower per token than the offline trial of the same config. Speculative configs skip this check, since how fast they decode depends on the text. After the batch profile, a probe at 2 req/s must not run more than 1.2 times slower than the first probe did. A server that fails either check is marked degraded and started once more, and a config degraded twice is left out of the comparison. The failing rate at which the search stops is probed a second time, so a single stall does not set the capacity.

Under WSL, the Windows graphics driver moves GPU memory into system memory instead of refusing an allocation, and PyTorch only frees its cache of unused blocks after an allocation fails. A server whose cache grows past the GPU then slows down without an error: with 2 GiB spilled, TPOT p90 at 16 req/s was 72 ms against 25 ms with the cache capped. vllm-dyno caps PyTorch's allocator at `gpu_memory_utilization` (`PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction`) for trials and servers, and leaves 1.5 GiB of GPU memory for the Windows desktop to grow into. The recommended command carries the same setting.

### Choosing configs with an LLM

`dyno load --planner llm` replaces the fixed order with a language model that picks one config at a time. After the baseline, vllm-dyno sends it the GPU and model, the SLO, every offline trial, every config loaded so far with its gains, its own earlier picks next to what they measured, and the configs it may pick: every combination of settings that each stayed within the quality limit offline. The model answers with a config, the gains it expects and why, or with a stop. A combination still gets its offline KL trial before a server starts, and `--max-servers` caps the servers. If the endpoint fails, or the model twice names a config outside the list, the fixed order finishes the session and reuses every config already loaded. Whichever finishes the session, the recommendation follows the same rule. Each pick goes into `runs/<run>/plan.jsonl` and the report with its reasoning, its expected and measured gains, and the tokens the endpoint counted.

A stop counts only once the effect of every setting the model may change is known: two configs that served, or one and the baseline, must differ in that setting alone. The models reason from offline speed, which misjudges the attention backend. Without any stop rule, two of four earlier sessions stopped after fp8 weights at 1.00x the baseline's chat capacity. A rule that only asked for each setting to be loaded once did not help: Nemotron put the int8 KV cache, TRITON_ATTN and ngram into a single config, expected 0.70x, measured 0.88x, stopped, and recommended fp8 weights at 1.04x.

Under the current rule, in sessions run back to back:

| planner | servers | minutes | recommended | chat | batch | KL | tokens in / out |
|---|---|---|---|---|---|---|---|
| fixed order | 11 | 75 | GPTQ-Int8 + TRITON_ATTN | 1.19x | 1.22x | 0.0008 | none |
| fixed order | 11 | 72 | GPTQ-Int8 + TRITON_ATTN | 1.19x | 1.22x | 0.0008 | none |
| Nemotron 3 Super | 8 | 57 | fp8 + TRITON_ATTN | 1.19x | 1.20x | 0.0069 | 23,226 / 11,792 |
| Nemotron 3 Super | 6 | 46 | GPTQ-Int8 + TRITON_ATTN | 1.19x | 1.23x | 0.0008 | 16,823 / 9,069 |
| Claude Opus 5.5 | 6 | 44 | GPTQ-Int8 + TRITON_ATTN | 1.19x | 1.24x | 0.0008 | 23,364 / 3,736 |
| Claude Opus 5.5 | 6 | 45 | GPTQ-Int8 + TRITON_ATTN | 1.24x | 1.23x | 0.0008 | 23,364 / 3,426 |

Servers include the baseline. Chat and batch are multiples of each session's own baseline, whose chat capacity came out at 16.00, 16.71 or 17.45 req/s, one bisection step apart. Both Opus sessions took the same path: fp8 weights, then TRITON_ATTN on top, then the int8 KV cache and ngram one at a time, then GPTQ-Int8 with TRITON_ATTN to win the tie on KL. Nemotron opened with fp8 weights in one session and with ngram in the other. In the first it never served GPTQ-Int8 with TRITON_ATTN, and recommended fp8 weights with TRITON_ATTN, as fast but at a KL of 0.0069 against 0.0008.

The planner talks to any OpenAI-compatible endpoint. By default that is Nemotron 3 Super on OpenRouter's free tier, with the API key in `OPENROUTER_API_KEY`:

```
uv run dyno load runs/<run> --planner llm
uv run dyno load runs/<run> --planner llm --planner-url https://api.anthropic.com/v1 --planner-model claude-opus-5-5 --planner-key-env ANTHROPIC_API_KEY
```

vllm-dyno takes the key out of its own environment before it starts anything, so neither vLLM nor the trials inherit it, and it writes the key nowhere.

## Options

`dyno run --help` and `dyno load --help` list all options. The ones most runs need:

- `--max-kl` sets the quality limit (default 0.05).
- `dyno run --goal latency` ranks configs by time per output token instead of throughput.
- `dyno run --max-ttft-ms` and `--max-tpot-ms` reject configs that are slower than a latency target.
- `dyno load --goal batch` recommends the config with the highest batch throughput instead of chat capacity.
- `dyno load --slo-ttft-ms`, `--slo-tpot-ms` and `--slo-share` set the SLO (defaults 1000 ms, 50 ms and 90% of requests).
- `dyno load --config <label>` loads the named configs instead of searching.
- `dyno load --planner llm` lets a language model choose the configs to load; `--planner-url`, `--planner-model` and `--planner-key-env` pick the endpoint, the model and the variable that holds its API key, and `--max-servers` caps the servers it starts (default 10).
- `--gpu-price` adds cost per million output tokens, given USD per GPU hour.
- `--text` measures quality and speed on your own text instead of WikiText-2.
- `--checkpoint` adds a quantized checkpoint that discovery did not pick.
- `--any-owner` lets discovery consider checkpoints from other accounts.

## Limitations

- Load traffic is synthetic: 256-token prompts and 128-token answers, with no shared prefixes.
- Requests sample with the model's own defaults. Clients that send other sampling parameters get different capacity: on the RTX 3070 with GPTQ-Int8 and TRITON_ATTN, greedy requests reached 21.67 req/s and Qwen2.5's defaults 19.87.
- Without nvcc on PATH, as under WSL, FlashInfer is unavailable: its attention backend, fp8 KV cache below compute capability 8.9, and its sampler, so vLLM samples with Triton kernels instead.
- Chat capacity comes in bisection steps of 4 to 9%, and each config is loaded once, so configs within 10% of each other count as a tie.
- `max_num_seqs` and `max_num_batched_tokens` stay at vLLM's serving defaults, 256 and 2,048 on GPUs under 70 GiB. Raising them did not help on the RTX 3070. With the GPTQ-Int8 checkpoint and int8 KV cache, the batch profile used 38% of the cache, and `--max-num-seqs 512` lowered batch throughput from 2,850 to 2,513 tok/s. With GPTQ-Int8 and TRITON_ATTN, `--max-num-batched-tokens 4096` left batch throughput level and raised TPOT p90 at 21.67 req/s from 51 to 54 ms.
- KL is computed over the baseline's top 20 tokens and reads slightly low.
- The planner has run four sessions under its current stop rule, all with one model on one GPU. There one setting, TRITON_ATTN, explains most of the gain, and the planner's prompt names attention backends as what offline speed misjudges. Whether it beats the fixed order where nobody knows the answer in advance is untested. The endpoint samples with its own defaults, so two sessions can pick differently.
- The planner sends the GPU and model names and the measurements to its endpoint; check the provider's data policy. Free models on OpenRouter are rate limited.
- One GPU, no tensor parallelism.
- Tested only on an RTX 3070 with vLLM 0.31.

## License

Apache-2.0
