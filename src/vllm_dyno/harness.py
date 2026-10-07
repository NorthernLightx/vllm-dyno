"""Start one vLLM engine and measure it. Runs as its own process, one per trial:

    python -m vllm_dyno.harness <trial>/spec.json

The run stores a hash of this file with every trial, and a resumed run refuses
to mix in results measured by a different version of it.

Quality is measured against the reference written by the baseline trial, on the
same tokens: KL divergence over the reference's top-k next-token distribution,
how often both pick the same top token, and perplexity.

Speed uses prompts no earlier request has seen, so prefix caching cannot shortcut prefill.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

TOP_K = 20  # vLLM's default max_logprobs
CHUNK = 512
TPUT_PROMPTS = 64
TPUT_PROMPT_LEN = 200
TPUT_TOKENS = 128


def position_stats(ref_ids: list[int], ref_lps: list[float], cand: dict[int, float]) -> tuple[float, bool]:
    """KL(reference || candidate) over the reference's top-k tokens, and whether both rank the same token first.

    `cand` holds the candidate's top-k logprobs. A reference token missing from it gets the
    lowest of them, an upper bound on its true logprob, so the KL here errs low.
    """
    floor = min(cand.values())
    kl = sum(math.exp(lp) * (lp - cand.get(t, floor)) for t, lp in zip(ref_ids, ref_lps))
    return kl, max(cand, key=cand.__getitem__) == ref_ids[0]


def measure_quality(llm, eval_ids: list[int], n_tokens: int, ref_path: Path, write_ref: bool) -> dict:
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(max_tokens=1, prompt_logprobs=TOP_K, temperature=0)
    ref_ids: list[list[int]] = []
    ref_lps: list[list[float]] = []
    if not write_ref:
        with np.load(ref_path) as ref:  # NpzFile re-reads an array on every key access
            ref_ids, ref_lps = ref["ids"].tolist(), ref["lps"].tolist()
    span = len(eval_ids) if write_ref else n_tokens  # the reference covers every eval token
    rows_ids, rows_lps = [], []
    nll, kl_sum, same, pos = 0.0, 0.0, 0, 0
    # one chunk per call: prompt logprobs materialise vocab-sized logits for every prompt token
    for start in range(0, span - CHUNK + 1, CHUNK):
        piece = eval_ids[start : start + CHUNK]
        out = llm.generate([TokensPrompt(prompt_token_ids=piece)], params, use_tqdm=False)[0]
        for tid, entry in zip(piece[1:], out.prompt_logprobs[1:]):
            topk = {t: lp.logprob for t, lp in entry.items() if lp.rank is not None and lp.rank <= TOP_K}
            if write_ref:
                order = sorted(topk, key=topk.__getitem__, reverse=True)[:TOP_K]
                order += [order[-1]] * (TOP_K - len(order))
                rows_ids.append(order)
                rows_lps.append([topk[t] for t in order])
            elif pos < n_tokens:
                kl, top1 = position_stats(ref_ids[pos], ref_lps[pos], topk)
                kl_sum += kl
                same += top1
            if pos < n_tokens:
                nll -= entry[tid].logprob
            pos += 1
    counted = min(pos, n_tokens)
    if write_ref:
        np.savez(ref_path, ids=np.array(rows_ids, dtype=np.int32), lps=np.array(rows_lps, dtype=np.float32))
        return {"kl": 0.0, "top1_agree": 1.0, "ppl": math.exp(nll / counted), "eval_tokens": counted}
    return {
        "kl": max(0.0, kl_sum / counted),  # truncating to the top-k can push the estimate slightly below zero
        "top1_agree": same / counted,
        "ppl": math.exp(nll / counted),
        "eval_tokens": counted,
    }


def measure_speed(llm, pool, tier: dict) -> dict:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    def timed(prompts: list[list[int]], max_tokens: int) -> tuple[float, int]:
        params = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        return time.perf_counter() - t0, sum(len(o.outputs[0].token_ids) for o in outs)

    ttft = statistics.median(timed([next(pool)], 1)[0] for _ in range(tier["ttft_n"]))
    n = tier["tpot_tokens"]
    tpot = statistics.median((timed([next(pool)], n)[0] - ttft) / (n - 1) for _ in range(tier["tpot_n"]))
    rates = []
    for _ in range(tier["tput_repeats"]):
        seconds, tokens = timed([next(pool)[:TPUT_PROMPT_LEN] for _ in range(TPUT_PROMPTS)], TPUT_TOKENS)
        rates.append(tokens / seconds)
    tput = statistics.mean(rates)
    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "tput_tok_s": tput,
        "tput_cv": statistics.stdev(rates) / tput if len(rates) > 1 else None,
    }


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    data = json.loads(Path(spec["data"]).read_text())
    tier = spec["tier"]

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t0 = time.perf_counter()
    # checkpoints can come from the Hub; never run code shipped inside a model repo
    llm = LLM(**spec["engine"], seed=0, trust_remote_code=False)
    load_s = time.perf_counter() - t0
    cache = llm.llm_engine.vllm_config.cache_config
    pool = iter(data["speed_prompts"])
    warmup = SamplingParams(max_tokens=8, temperature=0)
    llm.generate([TokensPrompt(prompt_token_ids=next(pool)[:64])], warmup, use_tqdm=False)

    metrics = {"load_s": load_s, "kv_tokens": (cache.num_gpu_blocks or 0) * cache.block_size}
    metrics |= measure_quality(
        llm, data["eval_ids"], tier["eval_tokens"], Path(spec["reference"]), spec["write_reference"]
    )
    metrics |= measure_speed(llm, pool, tier)
    Path(spec["out"]).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
