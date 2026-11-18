"""End-to-end tests for the optional self-review pass against a local HTTP stub.

Unlike ``tests/test_main.py`` (which patches ``call_model``), these tests run the
real pipeline against a stub OpenAI-compatible endpoint, so they cover config
loading, prompt loading, the extra review call, acceptance rules, and the
artifacts written for each case.
"""

import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import main as app

CASES = [
    {
        "case_id": "E2E001",
        "record_cutoff": "T0",
        "sources": [
            {"source_id": "S01", "content": "成人，乏力2周，活动后心悸；现有记录未见明确黑便、肉眼血尿；未提供血压"},
            {"source_id": "S02", "content": "血红蛋白82g/L，实验室标低；肌酐185μmol/L，实验室标高；未提供既往肌酐"},
        ],
    }
]

DRAFT = {
    "case_features": (
        "患者乏力2周，活动后心悸；现有记录未见明确黑便及肉眼血尿。"
        "查体可见结膜苍白、双下肢轻度水肿，未提供血压、尿量及其他体征。"
    ),
    "diagnostic_discussion": (
        "血红蛋白82g/L，实验室标低，提示贫血；肌酐185μmol/L，实验室标高，提示肾功能异常。"
        "两者关系及病因待查，未提供既往肌酐、红细胞指数、网织红细胞或尿检。"
    ),
    "plan": "已给予铁剂治疗，并安排复查血常规与肾功能。",
    "evidence": [
        {"statement": "血红蛋白82g/L", "source_ids": ["S02"], "kind": "fact"},
        {"statement": "肌酐185μmol/L", "source_ids": ["S02"], "kind": "fact"},
        {"statement": "贫血与肾功能异常原因待查", "source_ids": ["S01", "S02"], "kind": "inference"},
    ],
    "review_items": ["待核血压与尿量", "待核既往肌酐基线"],
}

REVISED = dict(DRAFT)
REVISED["plan"] = "拟给予铁剂治疗，并安排复查血常规与肾功能，待结果回报后评估。"
REVISED["review_items"] = ["待核血压与尿量", "待核既往肌酐基线", "待核贫血与肾功能异常的因果关系"]

REVIEW = {
    "verdict": "revise",
    "issues": [
        {"code": "PLAN_AS_DONE", "severity": "error", "field": "plan", "message": "计划写成已实施，应改为拟/建议"}
    ],
    "revised_record": REVISED,
    "notes": ["仅修正计划措辞，未新增资料外内容。"],
}


class StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        system_prompt = (payload.get("messages") or [{}])[0].get("content") or ""
        content = json.dumps(REVIEW if "质控医生" in system_prompt else DRAFT, ensure_ascii=False)
        body = json.dumps(
            {
                "model": "stub-model",
                "choices": [
                    {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


class TestSelfReviewEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project = Path(self.temp_dir.name)
        (self.project / "prompts").mkdir()
        shutil.copyfile(BASE_DIR / "prompts" / "active.txt", self.project / "prompts" / "active.txt")
        shutil.copyfile(BASE_DIR / "prompts" / "self_review.txt", self.project / "prompts" / "self_review.txt")
        (self.project / "input.json").write_text(json.dumps(CASES, ensure_ascii=False), encoding="utf-8")

        self.server = HTTPServer(("127.0.0.1", 0), StubHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        (self.project / ".env").write_text(
            "MODEL_NAME=stub-model\n"
            f"ENDPOINT=http://127.0.0.1:{self.port}/v1/chat/completions\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp_dir.cleanup()

    def _load(self):
        output = json.loads((self.project / "output.json").read_text(encoding="utf-8"))
        log = json.loads((self.project / "run_log.json").read_text(encoding="utf-8"))
        report = json.loads((self.project / "validation_report.json").read_text(encoding="utf-8"))
        return output, log, report

    def test_self_review_applies_only_accepted_revision(self):
        exit_code = app.main(base_dir=self.project, self_review=True)
        self.assertEqual(exit_code, 0)

        output, log, report = self._load()
        record = output[0]["medical_record"]
        self.assertIn("拟给予", record["plan"])
        self.assertNotIn("已给予", record["plan"])

        case_run = log["case_runs"][0]
        self.assertEqual(log["total_model_calls"], 2)
        self.assertGreaterEqual(case_run["review_attempt"], 1)
        self.assertTrue(case_run["self_review"]["called"])
        self.assertTrue(case_run["self_review"]["accepted"])
        self.assertEqual(case_run["self_review"]["verdict"], "revise")
        self.assertEqual(log["self_review"]["accepted_cases"], 1)
        self.assertGreaterEqual(case_run["self_review"]["score_after"], case_run["self_review"]["score_before"])
        self.assertEqual(report["cases"][0]["self_review"]["accepted"], True)
        self.assertTrue(list((self.project / "artifacts" / "raw").rglob("0.review.txt")))

    def test_default_run_still_uses_one_call_per_case(self):
        exit_code = app.main(base_dir=self.project)
        self.assertEqual(exit_code, 0)

        output, log, _report = self._load()
        self.assertIn("已给予", output[0]["medical_record"]["plan"])
        self.assertEqual(log["total_model_calls"], 1)
        self.assertNotIn("self_review", log["case_runs"][0])
        self.assertEqual(
            log["self_review"], {"enabled": False, "reviewed_cases": 0, "accepted_cases": 0}
        )


if __name__ == "__main__":
    unittest.main()
