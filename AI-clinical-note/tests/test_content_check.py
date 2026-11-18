"""Unit tests for src/content_check.py (content-level validation engine)."""

import json
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.checklist import load_checklist_for_case
from src.content_check import (
    CATEGORY_WEIGHTS,
    ContentCheckResult,
    check_record,
    summarize,
)


def load_demo_case(case_id: str) -> dict:
    data = json.loads((BASE_DIR / "examples" / f"{case_id}.input.json").read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return data[0]
    return data["cases"][0]


GOOD_RECORD = {
    "case_features": (
        "患者成人，乏力2周，活动后心悸，未见明确黑便及肉眼血尿；查体见结膜苍白、双下肢轻度水肿，"
        "血压及尿量未提供。实验室检查示血红蛋白82g/L（实验室标低），肌酐185μmol/L（实验室标高），"
        "既往肌酐、既往贫血检查及用药史未提供。"
    ),
    "diagnostic_discussion": (
        "结合乏力、活动后心悸与结膜苍白，考虑贫血；血红蛋白82g/L与肌酐185μmol/L同时异常，"
        "提示贫血合并肾功能异常。贫血原因需鉴别缺铁性贫血、肾性贫血及其他慢性病贫血；"
        "肾功能异常急慢性未明，现有资料不足以确定病因，入院初步判断为贫血及肾功能异常原因待查。"
    ),
    "plan": (
        "拟完善血常规、网织红细胞、铁代谢、叶酸维生素B12、肾功能及尿常规复查，拟行泌尿系超声评估肾脏形态；"
        "拟监测血压与尿量并记录出入量；建议肾内科会诊协助判断肾功能异常急慢性并评估肾性贫血可能；"
        "待检查结果回报后进一步明确贫血病因。"
    ),
    "evidence": [
        {"statement": "乏力2周，活动后心悸，未见明确黑便及肉眼血尿", "source_ids": ["S01"], "kind": "fact"},
        {"statement": "结膜苍白、双下肢轻度水肿，未提供血压尿量", "source_ids": ["S02"], "kind": "fact"},
        {"statement": "血红蛋白82g/L实验室标低，肌酐185μmol/L实验室标高", "source_ids": ["S03"], "kind": "fact"},
    ],
    "review_items": [
        "血压与尿量未提供，需补测并记录",
        "既往肾功能基线未提供，需追问既往肌酐水平",
        "贫血病程与出血风险待核，需追问黑便、月经量及铁剂使用史",
        "既往病史和药物史未提供，需补充",
        "用药史等缺失病史需核实",
    ],
}


def bad_record() -> dict:
    return {
        "case_features": (
            "患者成人，乏力2周，活动后心悸；结膜苍白、双下肢轻度水肿。血压正常，尿量正常。"
            "实验室检查示血红蛋白90g/L、肌酐185umol/L。"
        ),
        "diagnostic_discussion": "确诊肾性贫血，肾功能异常为慢性。",
        "plan": "已行输血治疗，安排肾内科会诊。",
        "evidence": [{"statement": "血红蛋白110g/L", "source_ids": ["S03"], "kind": "fact"}],
        "review_items": [],
    }


class TestContentCheckWithChecklist(unittest.TestCase):
    def setUp(self):
        self.case = load_demo_case("DEMO001")
        self.checklist = load_checklist_for_case("DEMO001", BASE_DIR)

    def test_good_record_passes(self):
        result = check_record(self.case, GOOD_RECORD, self.checklist)
        self.assertIsInstance(result, ContentCheckResult)
        self.assertTrue(result.passed, [i.to_dict() for i in result.issues])
        self.assertGreaterEqual(result.score, 80)
        self.assertEqual(result.metrics["required_fact_coverage"], 1.0)
        self.assertGreaterEqual(result.metrics["evidence"]["count"], 3)
        review_coverage = result.metrics["negation"]["review_coverage"]
        self.assertIsInstance(review_coverage, float)
        self.assertGreaterEqual(review_coverage, 0.0)
        self.assertLessEqual(review_coverage, 1.0)

    def test_bad_record_is_flagged(self):
        result = check_record(self.case, bad_record(), self.checklist)
        self.assertFalse(result.passed)
        codes = {issue.code for issue in result.issues}
        self.assertIn("PLAN_AS_DONE", codes)
        self.assertIn("DIAGNOSIS_CONFIRMED", codes)
        self.assertIn("NEGATION_REWRITTEN", codes)
        self.assertIn("REVIEW_ITEMS_EMPTY", codes)
        self.assertIn("CHECKLIST_NUMERIC_MISMATCH", codes)
        self.assertIn("CHECKLIST_FORBIDDEN_CLAIM", codes)
        self.assertLess(result.score, 40)

    def test_score_breakdown_is_category_weighted(self):
        result = check_record(self.case, GOOD_RECORD, self.checklist)
        self.assertEqual(set(result.score_breakdown), set(CATEGORY_WEIGHTS))
        for key, value in result.score_breakdown.items():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, CATEGORY_WEIGHTS[key])

    def test_checklist_case_mismatch(self):
        other = dict(self.checklist)
        other["case_id"] = "SOMETHING_ELSE"
        result = check_record(self.case, GOOD_RECORD, other)
        self.assertIn("CHECKLIST_CASE_MISMATCH", {issue.code for issue in result.issues})

    def test_record_not_object(self):
        result = check_record(self.case, "not-a-record", self.checklist)
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0)
        self.assertEqual({issue.code for issue in result.issues}, {"RECORD_NOT_OBJECT"})

    def test_ungrounded_evidence_number(self):
        record = dict(GOOD_RECORD)
        record["evidence"] = list(GOOD_RECORD["evidence"]) + [
            {"statement": "血红蛋白95g/L", "source_ids": ["S02"], "kind": "fact"}
        ]
        result = check_record(self.case, record, self.checklist)
        self.assertIn("EVIDENCE_NUMBER_UNGROUNDED", {issue.code for issue in result.issues})


class TestContentCheckWithoutChecklist(unittest.TestCase):
    def setUp(self):
        self.case = load_demo_case("DEMO001")

    def test_generic_checks_apply_without_checklist(self):
        result = check_record(self.case, GOOD_RECORD, None)
        self.assertTrue(result.passed)
        self.assertIsNone(result.metrics["required_fact_coverage"])
        codes = {issue.code for issue in result.issues}
        self.assertFalse({code for code in codes if code.startswith("CHECKLIST_")})

    def test_missing_info_rewritten_is_caught_without_checklist(self):
        result = check_record(self.case, bad_record(), None)
        codes = {issue.code for issue in result.issues}
        self.assertIn("NEGATION_REWRITTEN", codes)
        self.assertIn("DIAGNOSIS_CONFIRMED", codes)

    def test_preserved_missing_item_is_not_flagged_as_rewrite(self):
        case = {
            "case_id": "PRESERVE1",
            "record_cutoff": "T0",
            "sources": [
                {"source_id": "S01", "content": "腹部查体未见异常，未提供血压及血氧饱和度。"},
                {"source_id": "S02", "content": "血常规尚未回报。"},
            ],
        }
        record = {
            "case_features": "腹部查体未见异常；未提供血压及血氧饱和度，血氧饱和度未测。",
            "diagnostic_discussion": "腹部体征未见异常，血压与血氧饱和度资料缺失，原因待查。",
            "plan": "拟完善血常规，拟监测血压与血氧饱和度，待结果回报后评估。",
            "evidence": [{"statement": "腹部查体未见异常", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": ["血压及血氧饱和度未提供，需补测", "血常规结果需追踪"],
        }
        result = check_record(case, record, None)
        self.assertNotIn("NEGATION_REWRITTEN", {issue.code for issue in result.issues})
        self.assertIn("血压", result.metrics["negation"]["preserved_items"])

    def test_plan_written_as_done_matches_generic_regex(self):
        record = dict(GOOD_RECORD)
        record["plan"] = GOOD_RECORD["plan"] + "已给予输血治疗。"
        result = check_record(self.case, record, None)
        self.assertIn("PLAN_AS_DONE", {issue.code for issue in result.issues})

    def test_plan_as_done_allowed_when_source_reports_it(self):
        case = {
            "case_id": "SRCSUPPORT1",
            "record_cutoff": "T0",
            "sources": [
                {"source_id": "S01", "content": "急诊已给予退热治疗，体温降至37.2℃。"},
            ],
        }
        record = {
            "case_features": "患者发热，急诊已给予退热治疗，体温降至37.2℃。",
            "diagnostic_discussion": "发热原因待查，考虑感染可能。",
            "plan": "已给予退热治疗，拟完善血常规与CRP。",
            "evidence": [{"statement": "急诊已给予退热治疗", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": [],
        }
        result = check_record(case, record, None)
        self.assertNotIn("PLAN_AS_DONE", {issue.code for issue in result.issues})

    def test_untraceable_number_is_warning(self):
        record = dict(GOOD_RECORD)
        record["case_features"] = GOOD_RECORD["case_features"] + "血小板56×10^9/L。"
        result = check_record(self.case, record, None)
        self.assertIn("NUMBER_NOT_TRACEABLE", {issue.code for issue in result.issues})


class TestSummarize(unittest.TestCase):
    def test_summarize_aggregates(self):
        case = load_demo_case("DEMO001")
        checklist = load_checklist_for_case("DEMO001", BASE_DIR)
        good = check_record(case, GOOD_RECORD, checklist)
        bad = check_record(case, bad_record(), checklist)
        summary = summarize([good, bad])
        self.assertEqual(summary["checked_cases"], 2)
        self.assertEqual(summary["passed_cases"], 1)
        self.assertEqual(summary["failed_cases"], 1)
        self.assertIsNotNone(summary["average_score"])
        self.assertIn("PLAN_AS_DONE", summary["issues_by_code"])
        self.assertEqual(summary["checker_version"], "1.0")

    def test_summarize_empty(self):
        summary = summarize([])
        self.assertEqual(summary["checked_cases"], 0)
        self.assertIsNone(summary["average_score"])


if __name__ == "__main__":
    unittest.main()
