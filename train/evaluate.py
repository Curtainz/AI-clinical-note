"""Evaluate the base model or a LoRA adapter on the held-out validation split.

For every validation sample the script regenerates a medical record from the
user message, then scores it with the same code the runtime uses:

* ``src.validate.parse_json_content`` / ``validate_medical_record`` — schema
  compliance (the contract the competition grades against);
* ``src.content_check.check_record`` — content fidelity (numeric traceability,
  negation preservation, plan/diagnosis wording, evidence grounding, ...),
  optionally with judge checklists from ``--checklists``.

Usage (PowerShell)::

    # baseline: base model only
    python train/evaluate.py --val-file data/sft/val.jsonl --output reports/base_eval.json

    # improved: with the trained adapter, on the same validation set
    python train/evaluate.py --val-file data/sft/val.jsonl \
        --adapter checkpoints/qwen3-4b-lora --output reports/lora_eval.json

Then compare the two reports (and, for a run-level comparison, use
``tools/check_output.py --baseline-output``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.checklist import ChecklistError, load_checklist_for_case  # noqa: E402
from src.content_check import check_record  # noqa: E402
from src.validate import parse_json_content, validate_medical_record  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def _extract_case(sample: dict[str, Any]) -> dict[str, Any]:
    user_content = sample["messages"][1]["content"] if len(sample["messages"]) > 1 else "{}"
    try:
        case = json.loads(user_content)
    except json.JSONDecodeError:
        return {}
    return case if isinstance(case, dict) else {}


def score_sample(sample: dict[str, Any], content: str, checklists_dir: Path | None) -> dict[str, Any]:
    case = _extract_case(sample)
    case_id = str(case.get("case_id") or sample.get("case_id") or "")
    result: dict[str, Any] = {
        "case_id": case_id,
        "has_content": bool(content and content.strip()),
        "json_parse_ok": False,
        "schema_valid": False,
        "schema_error_codes": [],
        "content_score": None,
        "content_passed": None,
        "content_error_codes": [],
        "numeric_untraceable": None,
        "evidence_count": None,
        "char_total": len(content or ""),
    }
    if not result["has_content"]:
        return result

    try:
        record, _fence = parse_json_content(content)
        result["json_parse_ok"] = True
    except Exception:  # noqa: BLE001 - any parse failure is a failed case
        return result

    issues = validate_medical_record(record, case)
    errors = [issue for issue in issues if issue.get("severity") == "error"]
    result["schema_valid"] = not errors
    result["schema_error_codes"] = sorted({issue.get("code", "") for issue in errors})

    checklist = None
    if checklists_dir is not None and case_id:
        try:
            checklist = load_checklist_for_case(case_id, checklists_dir)
        except ChecklistError:
            checklist = None
    content_result = check_record(case, record, checklist)
    result["content_score"] = content_result.score
    result["content_passed"] = content_result.passed
    result["content_error_codes"] = sorted({issue.code for issue in content_result.errors})
    metrics = content_result.metrics
    result["numeric_untraceable"] = len(metrics.get("untraceable_numbers") or [])
    result["evidence_count"] = (metrics.get("evidence") or {}).get("count")
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    scored = [item for item in results if item["content_score"] is not None]
    return {
        "evaluated_cases": total,
        "parse_rate": round(sum(item["json_parse_ok"] for item in results) / max(total, 1), 4),
        "schema_pass_rate": round(sum(item["schema_valid"] for item in results) / max(total, 1), 4),
        "content_checked_cases": len(scored),
        "content_pass_rate": round(sum(bool(item["content_passed"]) for item in scored) / max(len(scored), 1), 4),
        "average_content_score": round(sum(item["content_score"] for item in scored) / max(len(scored), 1), 2),
        "average_untraceable_numbers": round(
            sum(item["numeric_untraceable"] or 0 for item in scored) / max(len(scored), 1), 2
        ),
        "average_evidence_items": round(
            sum(item["evidence_count"] or 0 for item in scored) / max(len(scored), 1), 2
        ),
        "average_chars": round(sum(item["char_total"] for item in results) / max(total, 1), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate base model / LoRA adapter on the validation split")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory (omit for baseline)")
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--checklists", type=Path, default=None, help="optional checklist directory")
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report to this path")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N samples (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--allow-cpu", action="store_true", help="allow CPU generation (slow, smoke tests only)")
    parser.add_argument("--dry-check", action="store_true", help="validate inputs without importing torch")
    args = parser.parse_args(argv)

    if not args.val_file.is_file():
        print(f"ERROR: validation file not found: {args.val_file}", file=sys.stderr)
        return 2
    samples = _load_jsonl(args.val_file)
    if not samples:
        print(f"ERROR: validation file is empty: {args.val_file}", file=sys.stderr)
        return 2
    if args.limit > 0:
        samples = samples[: args.limit]

    config = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "val_file": str(args.val_file),
        "samples": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy (do_sample=False) for reproducibility",
    }
    if args.dry_check:
        print(json.dumps({"config": config}, ensure_ascii=False, indent=2))
        print("DRY CHECK OK (heavy dependencies are not imported in this mode)")
        return 0

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the local environment
        print(
            "ERROR: evaluation dependencies are missing "
            f"({exc}).\nInstall them with: pip install -r requirements-train.txt",
            file=sys.stderr,
        )
        return 3

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("ERROR: no CUDA device found; pass --allow-cpu only for smoke tests.", file=sys.stderr)
        return 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if args.adapter:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        except ImportError as exc:  # pragma: no cover
            print(f"ERROR: peft is required to load the adapter ({exc})", file=sys.stderr)
            return 3
    model.eval()

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            prompt = _apply_chat_template(tokenizer, sample["messages"][:2], add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            scored = score_sample(sample, generated, args.checklists)
            scored["generated_preview"] = generated[:400]
            results.append(scored)
            print(f"[{index}/{len(samples)}] {scored['case_id']}: score={scored['content_score']}")

    summary = summarize(results)
    payload = {"config": config, "summary": summary, "cases": results}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
