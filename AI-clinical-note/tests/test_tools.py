"""Unit tests for the offline tooling (dataset builder + output comparison)."""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.build_sft_dataset import _is_complete_record, build_samples
from tools.check_output import compare_reports


def make_case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "record_cutoff": "T0",
        "sources": [{"source_id": "S01", "content": "示例来源"}],
    }


def make_reference(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "medical_record": {
            "case_features": "病例特点",
            "diagnostic_discussion": "拟诊讨论",
            "plan": "诊疗计划",
            "evidence": [{"statement": "示例来源", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": [],
        },
    }


class TestSFTDatasetBuilder(unittest.TestCase):
    def test_build_samples_chat_format(self):
        cases = [make_case("S001"), make_case("S002")]
        references = {case_id: make_reference(case_id)["medical_record"] for case_id in ("S001", "S002")}
        train, val, stats = build_samples(cases, references, "system prompt", val_ratio=0.0, salt="t")
        self.assertEqual(len(train) + len(val), 2)
        self.assertEqual(val, [])
        sample = train[0]
        self.assertEqual([message["role"] for message in sample["messages"]], ["system", "user", "assistant"])
        self.assertEqual(sample["messages"][0]["content"], "system prompt")
        self.assertIn('"case_id"', sample["messages"][1]["content"])
        self.assertIn('"case_features"', sample["messages"][2]["content"])
        self.assertEqual(stats["train_samples"], 2)
        self.assertEqual(stats["skipped"], [])

    def test_missing_reference_is_skipped(self):
        cases = [make_case("S001"), make_case("S002")]
        references = {"S001": make_reference("S001")["medical_record"]}
        train, val, stats = build_samples(cases, references, "p", val_ratio=0.0, salt="t")
        self.assertEqual(len(train), 1)
        self.assertEqual(stats["skipped"][0]["case_id"], "S002")

    def test_incomplete_reference_is_skipped(self):
        broken = {"case_features": "", "diagnostic_discussion": "x", "plan": "y", "evidence": [], "review_items": []}
        self.assertFalse(_is_complete_record(broken))
        self.assertTrue(_is_complete_record(make_reference("S001")["medical_record"]))

    def test_split_is_deterministic(self):
        cases = [make_case(f"S{index:03d}") for index in range(1, 21)]
        references = {case["case_id"]: make_reference(case["case_id"])["medical_record"] for case in cases}
        first_train, first_val, _ = build_samples(cases, references, "p", val_ratio=0.5, salt="salt")
        second_train, second_val, _ = build_samples(cases, references, "p", val_ratio=0.5, salt="salt")
        self.assertEqual(
            [sample["case_id"] for sample in first_train],
            [sample["case_id"] for sample in second_train],
        )
        self.assertEqual(
            [sample["case_id"] for sample in first_val],
            [sample["case_id"] for sample in second_val],
        )


def make_report(case_ids, scores, codes_by_case=None) -> dict:
    cases = []
    for index, case_id in enumerate(case_ids):
        codes = (codes_by_case or {}).get(case_id, [])
        cases.append(
            {
                "case_id": case_id,
                "score": scores[index],
                "issues": [{"severity": "warning" if code.startswith("W") else "error", "code": code} for code in codes],
            }
        )
    return {"cases": cases, "average_score": round(sum(scores) / len(scores), 2)}


class TestCompareReports(unittest.TestCase):
    def test_improvement_is_summarized(self):
        baseline = make_report(["C1", "C2"], [40, 60], {"C1": ["PLAN_AS_DONE", "WNIT"], "C2": []})
        improved = make_report(["C1", "C2"], [90, 60], {"C1": ["WNIT"], "C2": []})
        comparison = compare_reports(baseline, improved)
        self.assertEqual(comparison["shared_cases"], 2)
        self.assertEqual(comparison["average_score_delta"], 25.0)
        self.assertEqual(comparison["improved_cases"], 1)
        self.assertEqual(comparison["regressed_cases"], 0)
        self.assertEqual(comparison["unchanged_cases"], 1)
        first = comparison["per_case"][0]
        self.assertEqual(first["fixed_error_codes"], ["PLAN_AS_DONE"])
        self.assertEqual(first["new_error_codes"], [])
        self.assertTrue(any("平均分" in line for line in comparison["gains"]))

    def test_regression_is_reported_as_limitation(self):
        baseline = make_report(["C1"], [80], {"C1": []})
        improved = make_report(["C1"], [50], {"C1": ["NEGATION_REWRITTEN"]})
        comparison = compare_reports(baseline, improved)
        self.assertEqual(comparison["regressed_cases"], 1)
        self.assertEqual(comparison["per_case"][0]["new_error_codes"], ["NEGATION_REWRITTEN"])
        self.assertTrue(any("新增错误" in line for line in comparison["limitations"]))

    def test_no_shared_cases_is_reported(self):
        baseline = make_report(["C1"], [80])
        improved = make_report(["C2"], [90])
        comparison = compare_reports(baseline, improved)
        self.assertEqual(comparison["shared_cases"], 0)
        self.assertIsNone(comparison["average_score_delta"])
        self.assertTrue(comparison["limitations"])


if __name__ == "__main__":
    unittest.main()
