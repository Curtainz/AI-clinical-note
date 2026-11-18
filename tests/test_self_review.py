"""Unit tests for the optional self-review pass (pure logic only, no HTTP)."""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.self_review import (
    SelfReviewOutcome,
    build_review_user_content,
    evaluate_revision,
    parse_review_response,
)

SOURCES = [
    {"source_id": "S01", "content": "乏力2周，活动后心悸；现有记录未见黑便、肉眼血尿"},
    {"source_id": "S02", "content": "血红蛋白82g/L，实验室标低；肌酐185μmol/L，实验室标高"},
]


def make_case() -> dict:
    return {"case_id": "C1", "record_cutoff": "T0", "sources": SOURCES}


def make_record() -> dict:
    return {
        "case_features": "患者乏力2周，活动后心悸，现有记录未见黑便及肉眼血尿。" * 2,
        "diagnostic_discussion": "血红蛋白82g/L提示贫血，肌酐185μmol/L提示肾功能异常，原因均待查。" * 2,
        "plan": "拟完善血常规、网织红细胞、尿检及肾功能复查，待结果回报后评估。" * 2,
        "evidence": [
            {"statement": "血红蛋白82g/L", "source_ids": ["S02"], "kind": "fact"},
            {"statement": "肌酐185μmol/L", "source_ids": ["S02"], "kind": "fact"},
            {"statement": "贫血原因待查", "source_ids": ["S01", "S02"], "kind": "inference"},
        ],
        "review_items": ["待核既往肌酐基线"],
    }


class TestReviewPayload(unittest.TestCase):
    def test_user_content_contains_case_draft_and_findings(self):
        case = make_case()
        record = make_record()
        text = build_review_user_content(case, record)
        self.assertIn("C1", text)
        self.assertIn("待审查的初稿", text)
        self.assertIn("case_features", text)

    def test_parse_pass_response(self):
        parsed = parse_review_response('{"verdict": "pass", "issues": [], "notes": ["ok"], "revised_record": null}')
        self.assertEqual(parsed["verdict"], "pass")
        self.assertIsNone(parsed["revised_record"])
        self.assertEqual(parsed["notes"], ["ok"])

    def test_parse_fenced_revise_response(self):
        payload = '```json\n{"verdict": "revise", "issues": [{"code": "X", "severity": "ERROR", "field": "plan", "message": "m"}], "revised_record": {"case_features": "a"}, "notes": []}\n```'
        parsed = parse_review_response(payload)
        self.assertEqual(parsed["verdict"], "revise")
        self.assertEqual(parsed["issues"][0]["severity"], "error")
        self.assertIsInstance(parsed["revised_record"], dict)

    def test_parse_rejects_bad_verdict(self):
        with self.assertRaises(ValueError):
            parse_review_response('{"verdict": "maybe"}')

    def test_parse_rejects_revise_without_record(self):
        with self.assertRaises(ValueError):
            parse_review_response('{"verdict": "revise", "revised_record": null}')

    def test_parse_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_review_response("not json")


class TestRevisionDecision(unittest.TestCase):
    def test_identical_revision_is_accepted(self):
        case = make_case()
        record = make_record()
        accepted, reason, score, _result = evaluate_revision(case, record, record)
        self.assertTrue(accepted)
        self.assertIsNone(reason)
        self.assertIsNotNone(score)

    def test_invalid_source_reference_is_rejected(self):
        case = make_case()
        record = make_record()
        broken = make_record()
        broken["evidence"][0]["source_ids"] = ["S99"]
        accepted, reason, _score, _result = evaluate_revision(case, record, broken)
        self.assertFalse(accepted)
        self.assertIn("schema", reason)

    def test_plan_written_as_done_is_rejected(self):
        case = make_case()
        record = make_record()
        regressed = make_record()
        regressed["plan"] = regressed["plan"] + "已给予红细胞生成素治疗。"
        accepted, reason, _score, _result = evaluate_revision(case, record, regressed)
        self.assertFalse(accepted)
        self.assertIn("score", reason)

    def test_outcome_serialisation(self):
        outcome = SelfReviewOutcome(called=True, verdict="pass", score_before=88, accepted=False)
        data = outcome.to_dict()
        self.assertTrue(data["called"])
        self.assertEqual(data["verdict"], "pass")
        self.assertFalse(data["revised"])


if __name__ == "__main__":
    unittest.main()
