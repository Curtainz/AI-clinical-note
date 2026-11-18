#!/usr/bin/env python
"""Build a chat-format SFT dataset from input.json + reference.json.

The competition package never ships weights, and the competition itself does
not permit swapping the deployed model, so this script is the *offline* data
preparation step for local research: it converts the task's own case format
into the JSONL files consumed by ``train/train_lora.py``.

Input format::

    input.json      list of {"case_id", "record_cutoff", "sources": [...]}
    reference.json  {case_id: {"case_features", "diagnostic_discussion",
                               "plan", "evidence", "review_items"}}

Every sample is a chat conversation::

    system: prompts/active.txt
    user:   the case JSON (exactly what the runtime sends to the model)
    assistant: the reference medical_record JSON

Usage (PowerShell)::

    python tools/build_sft_dataset.py --input data/synth/input.json \
        --reference data/synth/reference.json --out-dir data/sft

    # add judge-checklists so the checker can score generated records later
    python tools/build_sft_dataset.py ... --checklists examples

Outputs: ``train.jsonl``, ``val.jsonl``, ``dataset_stats.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]

RECORD_FIELDS = ("case_features", "diagnostic_discussion", "plan", "evidence", "review_items")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array of cases")
    return [case for case in data if isinstance(case, dict)]


def _load_references(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object keyed by case_id")
    references: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(value.get("medical_record"), dict):
            value = value["medical_record"]
        if isinstance(value, dict):
            references[str(key)] = value
    return references


def _is_complete_record(record: dict[str, Any]) -> bool:
    if not all(isinstance(record.get(field), str) and record[field].strip() for field in RECORD_FIELDS[:3]):
        return False
    return isinstance(record.get("evidence"), list) and isinstance(record.get("review_items"), list)


def _split_bucket(case_id: str, val_ratio: float, salt: str) -> str:
    """Deterministic split so re-running produces identical files."""
    digest = hashlib.sha256(f"{salt}:{case_id}".encode("utf-8")).hexdigest()
    return "val" if (int(digest[:8], 16) / 0xFFFFFFFF) < val_ratio else "train"


def build_samples(
    cases: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    system_prompt: str,
    val_ratio: float,
    salt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str):
            continue
        reference = references.get(case_id)
        if not isinstance(reference, dict):
            skipped.append({"case_id": case_id, "reason": "no reference record"})
            continue
        if not _is_complete_record(reference):
            skipped.append({"case_id": case_id, "reason": "incomplete reference record"})
            continue
        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(reference, ensure_ascii=False)},
            ],
            "case_id": case_id,
        }
        if _split_bucket(case_id, val_ratio, salt) == "val":
            val.append(sample)
        else:
            train.append(sample)

    stats = {
        "cases_in_input": len(cases),
        "references_available": len(references),
        "train_samples": len(train),
        "val_samples": len(val),
        "skipped": skipped,
        "val_ratio": val_ratio,
        "split_salt": salt,
        "avg_assistant_chars": (
            round(sum(len(sample["messages"][2]["content"]) for sample in train + val) / max(len(train + val), 1), 1)
        ),
    }
    return train, val, stats


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build chat-format SFT data from input.json + reference.json")
    parser.add_argument("--input", required=True, type=Path, help="cases input.json (same format as the runtime)")
    parser.add_argument("--reference", required=True, type=Path, help="reference records keyed by case_id")
    parser.add_argument("--out-dir", required=True, type=Path, help="directory for train.jsonl / val.jsonl")
    parser.add_argument("--prompt", type=Path, default=BASE_DIR / "prompts" / "active.txt", help="system prompt file")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction of cases used for validation")
    parser.add_argument("--split-salt", default="clinical-note-v1", help="salt for the deterministic split")
    parser.add_argument("--max-cases", type=int, default=0, help="limit the number of cases used (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="random seed used only for shuffling")
    args = parser.parse_args(argv)

    if not args.prompt.is_file():
        print(f"ERROR: prompt file not found: {args.prompt}", file=sys.stderr)
        return 2
    system_prompt = args.prompt.read_text(encoding="utf-8-sig").strip()
    if not system_prompt:
        print(f"ERROR: prompt file is empty: {args.prompt}", file=sys.stderr)
        return 2
    if not 0.0 <= args.val_ratio < 1.0:
        print("ERROR: --val-ratio must be in [0, 1)", file=sys.stderr)
        return 2

    cases = _load_cases(args.input)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    references = _load_references(args.reference)

    train, val, stats = build_samples(cases, references, system_prompt, args.val_ratio, args.split_salt)
    if not train and not val:
        print("ERROR: no usable samples (check that reference.json covers the input case_ids)", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(train)
    _write_jsonl(args.out_dir / "train.jsonl", train)
    _write_jsonl(args.out_dir / "val.jsonl", val)
    (args.out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"train samples: {len(train)} -> {args.out_dir / 'train.jsonl'}")
    print(f"val samples:   {len(val)} -> {args.out_dir / 'val.jsonl'}")
    if stats["skipped"]:
        print(f"skipped cases: {len(stats['skipped'])} (no/incomplete reference)")
    print(f"average assistant length: {stats['avg_assistant_chars']} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
