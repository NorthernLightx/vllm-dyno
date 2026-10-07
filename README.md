# vllm-dyno

vllm-dyno runs candidate vLLM serving configs on your GPU, measures throughput, latency and how far each config's output drifts from the unmodified model, and recommends the fastest config that stays within your quality limit.

Status: v0.1, tested on one machine (RTX 3070 8 GB, WSL2, vLLM 0.31).

```
uv run dyno run Qwen/Qwen2.5-1.5B-Instruct
```

```
15:39:34  GPU      NVIDIA GeForce RTX 3070, compute capability 8.6, 5,934 of 8,192 MiB free
15:39:34  found    Qwen/Qwen2.5-1.5B-Instruct-AWQ (awq, 4-bit, 1.50 GiB)
15:39:34  found    Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 (gptq, 4-bit, 1.07 GiB)
15:39:34  found    Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8 (gptq, 8-bit, 1.69 GiB)
15:39:34  plan     kv_cache_dtype=fp8                                      skip: fp8 KV cache below compute capability 8.9 needs FlashInfer, which compiles with nvcc (not on PATH)
...
15:41:12  screen  1/11  baseline                                                ok    2645 tok/s  TPOT  12.6 ms  TTFT    57 ms  KL 0.0000  top-1 1.000  (94 s)
15:42:18  screen  2/11  weights=fp8                                             ok    2938 tok/s  TPOT   8.9 ms  TTFT    56 ms  KL 0.0071  top-1 0.949  (70 s)
...
16:07:00  confirm 2/4  weights=fp8                                             ok    2844 tok/s  TPOT   9.6 ms  TTFT    59 ms  KL 0.0076  top-1 0.952  (90 s)
16:09:43  best     weights=fp8
```

The trials took about 20 minutes; the gap in the timestamps is where the run was interrupted and resumed. The report confirmed these four configs with longer, repeated measurements:

| config | tok/s | TPOT ms | KV cache tokens | KL | top-1 agreement | perplexity |
|---|---|---|---|---|---|---|
| fp8 weights, quantized at load | 2,844 | 9.6 | 91,440 | 0.0076 | 0.952 | 11.553 |
| Qwen2.5-1.5B-Instruct-GPTQ-Int8 | 2,720 | 9.1 | 90,560 | 0.0006 | 0.976 | 11.483 |
| baseline (bf16) | 2,607 | 12.9 | 44,624 | 0 | 1 | 11.486 |
| int8 KV cache (`int8_per_token_head`) | 2,560 | 12.9 | 86,688 | 0.0172 | 0.938 | 11.659 |

The 4-bit AWQ and GPTQ checkpoints were faster per token (7.3 and 7.1 ms) but exceeded the default KL limit of 0.05, so they did not reach this stage.

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

Before loading anything, vllm-dyno reads the GPU's free memory and compute capability, checks for nvcc, works out the model's weight size and KV cache bytes per token, and looks up the most downloaded quantized checkpoints of the model on the Hugging Face Hub (one per method and bit width). It skips configs known to fail on this machine and logs the reason.

The search space has four settings: weights (as published, fp8 at load time, or a quantized checkpoint), KV cache dtype, attention backend, and ngram speculative decoding. The search runs in three stages:

1. Screen: the baseline, then one changed setting at a time.
2. Combine: settings that each beat the baseline by 5% or more, tried together.
3. Confirm: the best three and the baseline again, on more tokens and with repeated speed runs.

Each trial starts a fresh vLLM engine in its own process and measures:

- quality on 4,096 tokens of WikiText-2 (16,384 when confirming): KL divergence from the baseline's next-token distribution over its top 20 tokens, how often both rank the same token first, and perplexity;
- time to first token and time per output token for one request with a 512-token prompt;
- throughput of 64 requests with 200-token prompts and 128 output tokens each;
- KV cache capacity in tokens.

Every trial is a line in `runs/<run>/trials.jsonl`, with its vLLM log in `runs/<run>/trials/`. To resume an interrupted run, pass the same `--out`.

## Options

`dyno run --help` lists all options. The ones most runs need:

- `--goal latency` ranks configs by time per output token instead of throughput.
- `--max-kl` sets the quality limit (default 0.05).
- `--max-ttft-ms` and `--max-tpot-ms` reject configs that are slower than a latency target.
- `--gpu-price` adds cost per million output tokens, given USD per GPU hour.
- `--text` measures quality and speed on your own text instead of WikiText-2.
- `--checkpoint` adds a quantized checkpoint that discovery did not pick.

## Limitations

- Speed is measured with vLLM's offline engine; nothing yet measures a server under concurrent load. The throughput test fills about 21,000 tokens of KV cache, so a config that enlarges the cache gains nothing in it.
- KL is computed over the baseline's top 20 tokens and reads slightly low.
- One GPU, no tensor parallelism.
- Tested only on an RTX 3070 with vLLM 0.31.

## License

Apache-2.0
