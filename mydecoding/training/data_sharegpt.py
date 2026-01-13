"""
ShareGPT dataset helpers (robust local raw parser).

Fixes:
- HF datasets JSON schema casting issues (we only keep a single `text` column).
- JSONDecodeError: Extra data (file is JSONL or concatenated JSON objects but named .json)

Local files supported:
- JSONL (one JSON object per line)
- JSON list: [ {...}, {...}, ... ]
- JSON dict wrapper: { "data": [ ... ] }
- Concatenated JSON objects: {}{}{} (rare but seen)

Each record is rendered to one training string using:
- conversations: [{"from": "human"/"gpt", "value": ...}]
- messages: [{"role": "user"/"assistant", "content": ...}]
- prompt/response fallback

Env:
- SHAREGPT_DATASET=local  -> uses SHAREGPT_TRAIN_FILE / SHAREGPT_VAL_FILE
- SHAREGPT_TRAIN_FILE defaults to mydecoding/data/sharegpt/raw/train.json
- SHAREGPT_VAL_FILE   defaults to mydecoding/data/sharegpt/raw/val.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset


def _normalize_role(role: str) -> str:
    r = (role or "").lower()
    if r in ("human", "user", "prompt"):
        return "user"
    if r in ("gpt", "assistant", "bot", "model"):
        return "assistant"
    return "user"


def _iter_jsonl_lines(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _iter_concatenated_json(text: str) -> Iterable[Dict[str, Any]]:
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except Exception:
            break
        i = end
        if isinstance(obj, dict):
            yield obj
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    yield it
        elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            for it in obj["data"]:
                if isinstance(it, dict):
                    yield it


def _iter_json_records(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ShareGPT file not found: {path}")

    if p.suffix.lower() == ".jsonl":
        yield from _iter_jsonl_lines(p)
        return

    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    yield it
            return

        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                for it in obj["data"]:
                    if isinstance(it, dict):
                        yield it
                return
            yield obj
            return

    except json.JSONDecodeError:
        # Many "train.json" are actually JSONL.
        yielded = False
        for ex in _iter_jsonl_lines(p):
            yielded = True
            yield ex
        if yielded:
            return

        text = p.read_text(encoding="utf-8", errors="ignore")
        yield from _iter_concatenated_json(text)
        return


def _render_messages(messages: List[Dict[str, str]], tokenizer) -> str:
    cleaned = []
    for m in messages:
        cleaned.append({"role": _normalize_role(m.get("role", "user")), "content": str(m.get("content", ""))})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass

    parts = []
    for m in cleaned:
        if m["role"] == "user":
            parts.append(f"User: {m['content']}\n")
        else:
            parts.append(f"Assistant: {m['content']}\n")
    return "".join(parts).strip()


def _render_sharegpt_example(example: Dict[str, Any], tokenizer) -> str:
    conv = example.get("conversations")
    if isinstance(conv, list):
        msgs = []
        for t in conv:
            if not isinstance(t, dict):
                continue
            msgs.append({"role": _normalize_role(t.get("from", "user")), "content": str(t.get("value", ""))})
        if msgs:
            return _render_messages(msgs, tokenizer)

    msgs_raw = example.get("messages")
    if isinstance(msgs_raw, list):
        msgs = []
        for t in msgs_raw:
            if not isinstance(t, dict):
                continue
            msgs.append({"role": _normalize_role(t.get("role", "user")), "content": str(t.get("content", ""))})
        if msgs:
            return _render_messages(msgs, tokenizer)

    if "prompt" in example and "response" in example:
        msgs = [{"role": "user", "content": str(example.get("prompt", ""))},
                {"role": "assistant", "content": str(example.get("response", ""))}]
        return _render_messages(msgs, tokenizer)

    return json.dumps(example, ensure_ascii=False)


def _default_local_paths() -> Tuple[str, str]:
    # relative to repo root: mydecoding/data/sharegpt/raw/{train,val}.json
    train_file = os.environ.get('SHAREGPT_TRAIN_FILE', '/home/lwb/Decoding/mydecoding/mydecoding/data/sharegpt/raw/train.json')
    val_file = os.environ.get('SHAREGPT_VAL_FILE', '/home/lwb/Decoding/mydecoding/mydecoding/data/sharegpt/raw/val.json')
    return train_file, val_file

def _load_sharegpt_text_dataset(dataset_name: str, split: str, tokenizer, streaming: bool = False) -> Dataset:
    ds_name = (dataset_name or "").strip()

    if ds_name.lower() in ("local", ""):
        train_path, val_path = _default_local_paths()
        path = train_path if split in ("train", "training") else val_path
        records = [{"text": _render_sharegpt_example(ex, tokenizer)} for ex in _iter_json_records(path)]
        return Dataset.from_list(records)

    p = Path(ds_name)
    if p.exists() and p.suffix.lower() in (".json", ".jsonl"):
        records = [{"text": _render_sharegpt_example(ex, tokenizer)} for ex in _iter_json_records(str(p))]
        return Dataset.from_list(records)

    ds = load_dataset(ds_name, split=split, streaming=streaming)
    if streaming:
        texts = [{"text": _render_sharegpt_example(ex, tokenizer)} for ex in ds]
        return Dataset.from_list(texts)

    ds = ds.map(lambda ex: {"text": _render_sharegpt_example(ex, tokenizer)}, remove_columns=ds.column_names)
    return ds


def build_sharegpt_tokenized_dataset(
    tokenizer,
    seq_len: int,
    dataset_name: str,
    split: str = "train",
    streaming: bool = False,
    max_samples: int = 0,
    seed: int = 42,
) -> Dataset:
    ds_text = _load_sharegpt_text_dataset(dataset_name, split=split, tokenizer=tokenizer, streaming=streaming)

    if max_samples and max_samples > 0:
        ds_text = ds_text.shuffle(seed=seed).select(range(min(max_samples, len(ds_text))))

    def _tok(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=seq_len)
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    tokenized = ds_text.map(_tok, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized
