#!/usr/bin/env python
"""Offline content-quality report for an existing output.json.

This tool re-runs the content-level checks (src/content_check.py) against a
finished ``output.json`` without calling the model, so a run can be audited
after the fact and two prompts/versions can be compared on the same cases
(the "改进验证：baseline vs improved" evidence the task asks for).

Usage (PowerShell)::

    # single evaluation
    python tools/check_output.py --input input.json --output output.json

    # write a machine-readable report
    python tools/check_output.py --input input.json --output output.json --report content_report.json

    # baseline vs improved comparison on the same cases
    python tools/check_output.py --input input.json --output output_grounded.json \
        --baseline-output output_baseline.json --report comparison.json

Exit code is 0 even when content problems are found (this is a reporting
tool); pass ``--fail-on-error`` to get exit code 1 when any error-severity
content issue exists.
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
from src.content_check import check_record, summarize  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_cases(input_path: Path) -> list[dict[str, Any]]:
    data = _load_json(input_path)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(f"{input_path} is not a JSON array of cases")
    return [case for case in data if isinstance(case, dict)]


def _load_output(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    by_case: dict[str, dict[str, Any]] = {}
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str):
            by_case[entry["case_id"]] = entry
    return by_case


def evaluate_output(
    input_path: Path,
    output_path: Path,
    checklists_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the content checks over one output.json and return a JSON report."""
    cases = _load_cases(Path(input_path))
    output_by_case = _load_output(Path(output_path))
    base_dir = Path(checklists_dir) if checklists_dir else BASE_DIR

    results = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str):
            continue
        entry = output_by_case.get(case_id)
        record = entry.get("medical_record") if entry else None
        if not isinstance(record, dict):
            skipped.append({"case_id": case_id, "reason": "no medical_record in output.json (case failed)"})
            continue
        checklist = None
        try:
            checklist = load_checklist_for_case(case_id, base_dir)
        except ChecklistError as exc:
            skipped.append({"case_id": case_id, "reason": f"invalid checklist: {exc}"})
        results.append(check_record(case, record, checklist))

    summary = summarize(results)
    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    summary["skipped_cases"] = skipped
    summary["cases"] = [result.to_dict() for result in results]
    return summary


def compare_reports(baseline: dict[str, Any], improved: dict[str, Any]) -> dict[str, Any]:
    """Compare two evaluation reports case by case (baseline vs improved)."""
    baseline_cases = {item["case_id"]: item for item in baseline.get("cases", [])}
    improved_cases = {item["case_id"]: item for item in improved.get("cases", [])}
    shared = sorted(set(baseline_cases) & set(improved_cases))

    per_case = []
    improved_count = regressed_count = unchanged_count = 0
    for case_id in shared:
        before = baseline_cases[case_id]
        after = improved_cases[case_id]
        delta = int(after.get("score") or 0) - int(before.get("score") or 0)
        if delta > 0:
            improved_count += 1
        elif delta < 0:
            regressed_count += 1
        else:
            unchanged_count += 1
        before_errors = {issue["code"] for issue in before.get("issues", []) if issue.get("severity") == "error"}
        after_errors = {issue["code"] for issue in after.get("issues", []) if issue.get("severity") == "error"}
        per_case.append(
            {
                "case_id": case_id,
                "baseline_score": before.get("score"),
                "improved_score": after.get("score"),
                "delta": delta,
                "baseline_errors": sorted(before_errors),
                "improved_errors": sorted(after_errors),
                "fixed_error_codes": sorted(before_errors - after_errors),
                "new_error_codes": sorted(after_errors - before_errors),
            }
        )

    baseline_avg = baseline.get("average_score")
    improved_avg = improved.get("average_score")
    gains: list[str] = []
    limitations: list[str] = []
    if len(shared) < max(len(baseline.get("cases", [])), len(improved.get("cases", []))):
        limitations.append(
            f"仅有 {len(shared)} 例在两次运行中都成功产出记录，比较样本小于输入例数。"
        )
    delta_avg = None
    if not shared:
        limitations.append("两次运行没有可比较的共同病例，无法给出前后对比结论。")
    elif isinstance(baseline_avg, (int, float)) and isinstance(improved_avg, (int, float)):
        delta_avg = round(float(improved_avg) - float(baseline_avg), 2)
        if delta_avg > 0:
            gains.append(f"内容校验平均分由 {baseline_avg} 提升至 {improved_avg}（+{delta_avg}）。")
        elif delta_avg < 0:
            limitations.append(f"内容校验平均分由 {baseline_avg} 下降至 {improved_avg}（{delta_avg}）。")
        else:
            limitations.append(f"内容校验平均分持平（{baseline_avg}），本次改动无量化收益。")
    for item in per_case:
        if item["fixed_error_codes"]:
            gains.append(
                f"{item['case_id']}: 消除错误 {', '.join(item['fixed_error_codes'])}（分数 {item['baseline_score']}→{item['improved_score']}）。"
            )
        if item["new_error_codes"]:
            limitations.append(
                f"{item['case_id']}: 新增错误 {', '.join(item['new_error_codes'])}（分数 {item['baseline_score']}→{item['improved_score']}）。"
            )

    return {
        "shared_cases": len(shared),
        "baseline_average_score": baseline_avg,
        "improved_average_score": improved_avg,
        "average_score_delta": delta_avg,
        "improved_cases": improved_count,
        "regressed_cases": regressed_count,
        "unchanged_cases": unchanged_count,
        "per_case": per_case,
        "gains": gains,
        "limitations": limitations,
    }


def _print_evaluation(report: dict[str, Any]) -> None:
    print(f"== 内容校验报告: {report['output']} ==")
    print(
        f"检查 {report['checked_cases']} 例，通过 {report['passed_cases']} 例，"
        f"平均分 {report['average_score']}，最低分 {report['min_score']}"
    )
    if report["issues_by_code"]:
        print("问题分布: " + ", ".join(f"{code}×{count}" for code, count in sorted(report["issues_by_code"].items())))
    for item in report.get("skipped_cases", []):
        print(f"跳过 {item['case_id']}: {item['reason']}")
    for case in report["cases"]:
        errors = [issue for issue in case["issues"] if issue["severity"] == "error"]
        status = "PASS" if case["passed"] else "FAIL"
        print(f"  [{status}] {case['case_id']}: {case['score']} 分，{len(errors)} 个错误")
        for issue in errors[:5]:
            print(f"      - {issue['code']}: {issue['message']}")
    if report.get("skipped_cases"):
        print()


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("== baseline vs improved 对比 ==")
    print(
        f"共同病例 {comparison['shared_cases']} 例；平均分 "
        f"{comparison['baseline_average_score']} → {comparison['improved_average_score']}"
        f"（差值 {comparison['average_score_delta']}）"
    )
    print(
        f"提升 {comparison['improved_cases']} 例，退步 {comparison['regressed_cases']} 例，"
        f"持平 {comparison['unchanged_cases']} 例"
    )
    print("收益:")
    for line in comparison["gains"] or ["无（未观察到分数提升或错误消除）"]:
        print(f"  - {line}")
    print("局限:")
    for line in comparison["limitations"] or ["无特定限制"]:
        print(f"  - {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline content-quality report for output.json")
    parser.add_argument("--input", required=True, type=Path, help="input.json used for the run")
    parser.add_argument("--output", required=True, type=Path, help="output.json produced by main.py")
    parser.add_argument("--checklists", type=Path, default=None, help="directory containing <case_id>.checklist.json")
    parser.add_argument("--baseline-output", type=Path, default=None, help="baseline output.json to compare against")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report to this path")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="exit with code 1 when any error-severity content issue is found",
    )
    args = parser.parse_args(argv)

    report = evaluate_output(args.input, args.output, args.checklists)
    _print_evaluation(report)
    payload: dict[str, Any] = {"evaluation": report}

    if args.baseline_output:
        baseline_report = evaluate_output(args.input, args.baseline_output, args.checklists)
        comparison = compare_reports(baseline_report, report)
        _print_comparison(comparison)
        payload["baseline_evaluation"] = baseline_report
        payload["comparison"] = comparison

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入 {args.report}")

    has_errors = any(issue["severity"] == "error" for case in report["cases"] for issue in case["issues"])
    if args.fail_on_error and has_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
