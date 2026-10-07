"""Token ids every trial reads: an eval span for quality and a pool of fresh prompts for speed."""

from __future__ import annotations

from pathlib import Path

SEQ_LEN = 512
CHARS_PER_TOKEN = 6  # generous; text is cut to this many characters per needed token before tokenizing


def wikitext(split: str) -> str:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return "".join(pq.read_table(path).column("text").to_pylist())


def build(model: str, revision: str | None, text_file: Path | None, eval_tokens: int, pool_size: int) -> dict:
    """Quality reads WikiText-2 test and speed prompts come from train; a user file supplies both, in that order."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    if tok is None:
        raise SystemExit(f"{model} has no tokenizer")
    pool_tokens = pool_size * SEQ_LEN

    def ids(text: str, n: int) -> list[int]:
        return tok(text[: n * CHARS_PER_TOKEN], verbose=False).input_ids[:n]

    if text_file:
        all_ids = ids(text_file.read_text(), eval_tokens + pool_tokens)
        eval_ids, speed_ids = all_ids[:eval_tokens], all_ids[eval_tokens:]
    else:
        eval_ids, speed_ids = ids(wikitext("test"), eval_tokens), ids(wikitext("train"), pool_tokens)
    if len(eval_ids) < eval_tokens or len(speed_ids) < pool_tokens:
        raise SystemExit(
            f"text too short: need {eval_tokens:,} tokens for quality and {pool_tokens:,} for speed prompts"
        )
    return {"eval_ids": eval_ids, "speed_prompts": [speed_ids[i : i + SEQ_LEN] for i in range(0, pool_tokens, SEQ_LEN)]}
