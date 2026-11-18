#!/usr/bin/env python
"""Collect validated run outputs into a reusable SFT dataset ("data flywheel").

This closes the loop between the runtime and the offline training track:

    main.py (generate + validate [+ self-review])
        -> output.json / validation_report.json
        -> this tool (filter by schema + content score)
        -> reference.json + train.jsonl / val.jsonl
        -> train/train_lora.py -> train/evaluate.py

Only records that still pass schema/source validation and reach ``--min-score``
are exported.  ``provenance.json`` keeps the score, checklist path and source
hashes so a training set can always be traced back to the run that produced it.

Usage (PowerShell)::

    python tools/collect_training_data.py `
        --input input.json --output output.json `
        --report validation_report.json --out-dir data/collected

    # default threshold is 80; lower it for smoke data
    python tools/collect_training_data.py --input input.json --output output.json --min-score 0 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.checklist import ChecklistError, load_checklist_for_case
from src.content_check import CHECKER_VERSION, check_record
from src.validate import validate_medical_record
from tools.build_sft_dataset import _write_jsonl, build_samples

RECORD_FIELDS = ("case_features", "diagnostic_discussion", "plan", "evidence", "review_items")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array of cases")
    return [case for case in data if isinstance(case, dict) and isinstance(case.get("case_id"), str)]


def load_outputs(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array of case results")
    outputs: dict[str, dict[str, Any]] = {}
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str):
            outputs[entry["case_id"]] = entry
    return outputs


def load_report(path: Path) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Return ``(run_id, case_id -> report entry)`` from validation_report.json."""
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a validation report object")
    run_id = data.get("run_id") if isinstance(data.get("run_id"), str) else None
    cases: dict[str, dict[str, Any]] = {}
    for entry in data.get("cases") or []:
        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str):
            cases[entry["case_id"]] = entry
    return run_id, cases


def collect(
    cases: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    report_cases: dict[str, dict[str, Any]],
    project_root: Path | str,
    min_score: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter run outputs into training references.

    Returns ``(accepted_records, provenance_rows, rejected_rows)``.
    """
    accepted: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["case_id"]
        entry = outputs.get(case_id)
        if not isinstance(entry, dict):
            rejected.append({"case_id": case_id, "reason": "no output.json entry"})
            continue
        record = entry.get("medical_record")
        if entry.get("status") != "success" or not isinstance(record, dict):
            rejected.append({"case_id": case_id, "reason": f"status={entry.get('status')!r}"})
            continue

        schema_errors = [
            issue
            for issue in validate_medical_record(record, case)
            if issue.get("severity") == "error"
        ]
        if schema_errors:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "schema/source error: " + str(schema_errors[0].get("code")),
                }
            )
            continue

        checklist = None
        try:
            checklist = load_checklist_for_case(case_id, project_root)
        except ChecklistError as exc:
            rejected.append({"case_id": case_id, "reason": f"checklist error: {exc}"})
            continue

        result = check_record(case, record, checklist)
        if result.score < min_score:
            rejected.append(
                {"case_id": case_id, "reason": f"content score {result.score} < min_score {min_score}"}
            )
            continue

        reported = (report_cases.get(case_id) or {}).get("content_check") or {}
        accepted[case_id] = record
        provenance.append(
            {
                "case_id": case_id,
                "content_score": result.score,
                "content_passed": result.passed,
                "reported_score": reported.get("score"),
                "content_issue_codes": sorted({issue.code for issue in result.issues}),
                "checklist_path": checklist.get("_path") if checklist else None,
                "text_chars": sum(len(str(record.get(name) or "")) for name in RECORD_FIELDS[:3]),
            }
        )

    return accepted, provenance, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect validated output.json records into an SFT dataset for train/train_lora.py"
    )
    parser.add_argument("--input", required=True, type=Path, help="cases input.json used for the run")
    parser.add_argument("--output", required=True, type=Path, help="output.json produced by main.py")
    parser.add_argument("--report", type=Path, default=None, help="validation_report.json (optional, for run_id)")
    parser.add_argument("--out-dir", required=True, type=Path, help="destination directory")
    parser.add_argument("--prompt", type=Path, default=BASE_DIR / "prompts" / "active.txt", help="system prompt file")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=BASE_DIR,
        help="root used to look up checklists/ and examples/ (default: project root)",
    )
    parser.add_argument("--min-score", type=int, default=80, help="minimum content score to accept a record")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction of cases used for validation")
    parser.add_argument("--split-salt", default="collected-v1", help="salt for the deterministic split")
    parser.add_argument("--max-cases", type=int, default=0, help="limit the number of cases (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="random seed used only for shuffling")
    args = parser.parse_args(argv)

    for label, path in (("input", args.input), ("output", args.output)):
        if not path.is_file():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 2
    if args.report is not None and not args.report.is_file():
        print(f"ERROR: report file not found: {args.report}", file=sys.stderr)
        return 2
    if not args.prompt.is_file():
        print(f"ERROR: prompt file not found: {args.prompt}", file=sys.stderr)
        return 2
    if not 0 <= args.min_score <= 100:
        print("ERROR: --min-score must be within [0, 100]", file=sys.stderr)
        return 2
    if not 0.0 <= args.val_ratio < 1.0:
        print("ERROR: --val-ratio must be in [0, 1)", file=sys.stderr)
        return 2

    system_prompt = args.prompt.read_text(encoding="utf-8-sig").strip()
    if not system_prompt:
        print(f"ERROR: prompt file is empty: {args.prompt}", file=sys.stderr)
        return 2

    cases = load_cases(args.input)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    outputs = load_outputs(args.output)
    run_id, report_cases = load_report(args.report) if args.report is not None else (None, {})

    accepted, provenance, rejected = collect(cases, outputs, report_cases, args.project_root, args.min_score)
    if not accepted:
        print("ERROR: no record passed the filters (schema/source + content score)", file=sys.stderr)
        for row in rejected:
            print(f"  - {row['case_id']}: {row['reason']}", file=sys.stderr)
        return 1

    selected_cases = [case for case in cases if case["case_id"] in accepted]
    train, val, stats = build_samples(
        selected_cases, accepted, system_prompt, args.val_ratio, args.split_salt
    )
    if not train and not val:
        print("ERROR: accepted records produced no samples", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "train.jsonl", train)
    _write_jsonl(args.out_dir / "val.jsonl", val)
    (args.out_dir / "reference.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    provenance_doc = {
        "generated_at_utc": generated_at,
        "tool": "tools/collect_training_data.py",
        "run_id": run_id,
        "input_sha256": _sha256(args.input),
        "output_sha256": _sha256(args.output),
        "checker_version": CHECKER_VERSION,
        "min_score": args.min_score,
        "cases": provenance,
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats_out = {
        **stats,
        "min_score": args.min_score,
        "run_id": run_id,
        "accepted_cases": len(accepted),
        "rejected": rejected,
        "generated_at_utc": generated_at,
    }
    (args.out_dir / "dataset_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"accepted cases: {len(accepted)} / {len(cases)} (min_score={args.min_score})")
    print(f"train samples:  {len(train)} -> {args.out_dir / 'train.jsonl'}")
    print(f"val samples:    {len(val)} -> {args.out_dir / 'val.jsonl'}")
    if rejected:
        print(f"rejected cases: {len(rejected)} (see dataset_stats.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
