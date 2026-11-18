"""Acceptance and regression tests for main.py.

All tests execute in isolated temporary directories to guarantee zero pollution
of the real project's artifacts, logs, or workspace files.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_client import ModelHttpError, ModelTimeoutError
import main


class IsolatedProjectTestCase(unittest.TestCase):
    """Base test case providing an isolated temporary project workspace."""

    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir_obj.name)

        # Create basic directory structure in temp dir
        (self.project_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        # Default valid .env
        (self.project_dir / ".env").write_text(
            "MODEL_NAME=test-model\nENDPOINT=http://127.0.0.1:9999/v1/chat/completions\n",
            encoding="utf-8",
        )
        # Default prompt
        (self.project_dir / "prompts" / "active.txt").write_text(
            "You are a medical record generator.", encoding="utf-8"
        )

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def create_input(self, data: object) -> Path:
        input_path = self.project_dir / "input.json"
        input_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return input_path


class TestMainIsolatedScenarios(IsolatedProjectTestCase):
    def test_empty_array_produces_empty_output_and_exits_zero(self):
        self.create_input([])
        exit_code = main.main(base_dir=self.project_dir)
        self.assertEqual(exit_code, 0)

        output_file = self.project_dir / "output.json"
        self.assertTrue(output_file.is_file())
        output_data = json.loads(output_file.read_text(encoding="utf-8"))
        self.assertEqual(output_data, [])

        # Check logs written to both root and artifacts
        self.assertTrue((self.project_dir / "run_log.json").is_file())
        self.assertTrue((self.project_dir / "validation_report.json").is_file())
        self.assertTrue((self.project_dir / "artifacts" / "run_log.json").is_file())

    def test_duplicate_case_id_exits_two_no_model_calls(self):
        dup_cases = [
            {"case_id": "DUP01", "record_cutoff": "T0", "sources": []},
            {"case_id": "DUP01", "record_cutoff": "T0", "sources": []},
        ]
        self.create_input(dup_cases)

        with patch("main.call_model") as mock_call:
            exit_code = main.main(base_dir=self.project_dir)
            self.assertEqual(exit_code, 2)
            mock_call.assert_not_called()

    def test_startup_failure_records_uncompleted_log(self):
        """Regression test for [P2] R88-R92: startup failure must record completed=False log."""
        dup_cases = [
            {"case_id": "DUP01", "record_cutoff": "T0", "sources": []},
            {"case_id": "DUP01", "record_cutoff": "T0", "sources": []},
        ]
        self.create_input(dup_cases)

        exit_code = main.main(base_dir=self.project_dir)
        self.assertEqual(exit_code, 2)

        # Verify that run_log.json exists and has completed=False and error info
        run_log_path = self.project_dir / "run_log.json"
        self.assertTrue(run_log_path.is_file())
        log_data = json.loads(run_log_path.read_text(encoding="utf-8"))
        self.assertFalse(log_data["completed"])
        self.assertEqual(log_data["error"]["code"], "TASK_INPUT_ERROR")
        self.assertIn("Duplicate case_id", log_data["error"]["message"])

        # Also check validation_report has fatal_error
        val_path = self.project_dir / "validation_report.json"
        self.assertTrue(val_path.is_file())
        val_data = json.loads(val_path.read_text(encoding="utf-8"))
        self.assertIn("fatal_error", val_data)

    def test_attempt_is_zero_when_input_invalid(self):
        """Regression test for [P2] R121-R130: attempt must be 0 if case input validation fails."""
        invalid_cases = [
            {
                "case_id": "BAD01",
                # missing record_cutoff and invalid sources
                "sources": "not-a-list",
            }
        ]
        self.create_input(invalid_cases)

        with patch("main.call_model") as mock_call:
            exit_code = main.main(base_dir=self.project_dir)
            self.assertEqual(exit_code, 0)
            mock_call.assert_not_called()

        run_log_path = self.project_dir / "run_log.json"
        log_data = json.loads(run_log_path.read_text(encoding="utf-8"))
        self.assertEqual(log_data["total_model_calls"], 0)
        self.assertEqual(len(log_data["case_runs"]), 1)
        case_run = log_data["case_runs"][0]
        self.assertEqual(case_run["attempt"], 0)
        self.assertFalse(case_run["model_called"])
        self.assertEqual(case_run["status"], "failed")

    def test_http_error_retains_diagnostic_evidence(self):
        """Regression test for [P2] R258-R267: interface errors must preserve raw response and elapsed seconds."""
        cases = [
            {
                "case_id": "HTTP_FAIL_01",
                "record_cutoff": "T0",
                "sources": [{"source_id": "S01", "content": "Text"}],
            }
        ]
        self.create_input(cases)

        def mock_http_fail(*args, **kwargs):
            raise ModelHttpError(
                "Model endpoint returned HTTP 401: Unauthorized",
                details={
                    "http_status": 401,
                    "elapsed_seconds": 1.456,
                    "raw_response": '{"error": {"message": "Invalid API key provided", "type": "auth_error"}}',
                },
            )

        with patch("main.call_model", side_effect=mock_http_fail):
            exit_code = main.main(base_dir=self.project_dir)
            self.assertEqual(exit_code, 0)

        # Check output.json
        output_data = json.loads((self.project_dir / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(output_data[0]["status"], "failed")
        self.assertEqual(output_data[0]["error"]["code"], "MODEL_HTTP_ERROR")

        # Check run_log.json
        run_log = json.loads((self.project_dir / "run_log.json").read_text(encoding="utf-8"))
        case_run = run_log["case_runs"][0]
        self.assertEqual(case_run["attempt"], 1)
        self.assertTrue(case_run["model_called"])
        self.assertEqual(case_run["elapsed_seconds"], 1.456)
        self.assertIsNotNone(case_run["raw_response_path"])
        self.assertEqual(case_run["details"]["http_status"], 401)

        # Check that raw response was written to file
        raw_file = self.project_dir / case_run["raw_response_path"]
        self.assertTrue(raw_file.is_file())
        raw_content = raw_file.read_text(encoding="utf-8")
        self.assertIn("Invalid API key provided", raw_content)

    def test_middle_case_simulated_failure_continues_batch(self):
        three_cases = [
            {
                "case_id": "C01",
                "record_cutoff": "T0",
                "sources": [{"source_id": "S01", "content": "Text 1"}],
            },
            {
                "case_id": "C02",
                "record_cutoff": "T0",
                "sources": [{"source_id": "S01", "content": "Text 2"}],
            },
            {
                "case_id": "C03",
                "record_cutoff": "T0",
                "sources": [{"source_id": "S01", "content": "Text 3"}],
            },
        ]
        self.create_input(three_cases)

        valid_record = {
            "case_features": "Features",
            "diagnostic_discussion": "Discussion",
            "plan": "Plan",
            "evidence": [{"statement": "Text 1", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": [],
        }

        def mock_call(case, system_prompt, **kwargs):
            if case["case_id"] == "C02":
                raise ModelTimeoutError(
                    "Simulated timeout on case C02",
                    details={"elapsed_seconds": 10.0, "raw_response": None},
                )
            return {
                "content": json.dumps(valid_record, ensure_ascii=False),
                "finish_reason": "stop",
                "usage": None,
                "elapsed_seconds": 0.5,
                "model_name": "mock-model",
                "http_status": 200,
                "raw_response": json.dumps(valid_record),
            }

        with patch("main.call_model", side_effect=mock_call):
            exit_code = main.main(base_dir=self.project_dir)
            self.assertEqual(exit_code, 0)

        output_file = self.project_dir / "output.json"
        self.assertTrue(output_file.is_file())
        output_data = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(len(output_data), 3)
        self.assertEqual(output_data[0]["case_id"], "C01")
        self.assertEqual(output_data[0]["status"], "success")

        self.assertEqual(output_data[1]["case_id"], "C02")
        self.assertEqual(output_data[1]["status"], "failed")
        self.assertEqual(output_data[1]["error"]["code"], "MODEL_TIMEOUT")

        self.assertEqual(output_data[2]["case_id"], "C03")
        self.assertEqual(output_data[2]["status"], "success")

    def test_no_project_pollution(self):
        """Regression test for [P1] R24-R38: running main tests in temp dir never touches BASE_DIR artifacts."""
        real_artifacts = BASE_DIR / "artifacts"
        real_files_before = set(real_artifacts.rglob("*")) if real_artifacts.exists() else set()

        # Run test inside temp directory
        self.create_input([])
        main.main(base_dir=self.project_dir)

        real_files_after = set(real_artifacts.rglob("*")) if real_artifacts.exists() else set()
        self.assertEqual(real_files_before, real_files_after)


class TestContentCheckIntegration(IsolatedProjectTestCase):
    """Integration tests for the content-level validation stage in main.py."""

    def _valid_record(self, **overrides) -> dict:
        record = {
            "case_features": "患者乏力2周，活动后心悸；查体见结膜苍白，血压及尿量未提供。血红蛋白82g/L（标低）。",
            "diagnostic_discussion": "考虑贫血，原因待查；肾功能情况需结合未提供的既往肌酐判断。",
            "plan": "拟完善血常规、铁代谢及肾功能复查，拟监测血压与尿量，建议血液科会诊。",
            "evidence": [{"statement": "血红蛋白82g/L（标低）", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": ["血压与尿量未提供，需补测", "既往肌酐未提供，需追问"],
        }
        record.update(overrides)
        return record

    def _write_checklist(self, case_id: str, **overrides) -> None:
        checklist = {
            "case_id": case_id,
            "required_facts": [{"fact_id": "F01", "text": "乏力2周活动后心悸", "source_ids": ["S01"]}],
            "numeric_facts": [{"concept": "血红蛋白", "value": 82, "unit": "g/L", "source_ids": ["S01"]}],
            "negation_scope": [],
            "missing_items": ["血压", "尿量"],
            "forbidden_claims": ["血压正常或尿量正常"],
            "review_targets": ["血压与尿量"],
            "split": "dev",
            "origin": "unit-test",
            "clinical_review_status": "not_expert_reviewed",
        }
        checklist.update(overrides)
        (self.project_dir / "checklists").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "checklists" / f"{case_id}.checklist.json").write_text(
            json.dumps(checklist, ensure_ascii=False), encoding="utf-8"
        )

    def _run_with_record(self, record: dict, case_id: str = "CASE01") -> int:
        case = {
            "case_id": case_id,
            "record_cutoff": "T0",
            "sources": [{"source_id": "S01", "content": "乏力2周，活动后心悸；血红蛋白82g/L（标低）；未提供血压、尿量。"}],
        }
        self.create_input([case])
        mock_response = {
            "content": json.dumps(record, ensure_ascii=False),
            "finish_reason": "stop",
            "usage": None,
            "elapsed_seconds": 0.2,
            "model_name": "mock-model",
            "http_status": 200,
            "raw_response": json.dumps(record, ensure_ascii=False),
        }
        with patch("main.call_model", return_value=mock_response):
            return main.main(base_dir=self.project_dir)

    def test_content_check_written_to_reports(self):
        self._write_checklist("CASE01")
        exit_code = self._run_with_record(self._valid_record())
        self.assertEqual(exit_code, 0)

        validation = json.loads((self.project_dir / "validation_report.json").read_text(encoding="utf-8"))
        case_report = validation["cases"][0]
        self.assertTrue(case_report["protocol_valid"])
        self.assertIsNotNone(case_report["content_check"])
        self.assertIsInstance(case_report["content_check"]["score"], int)

        run_log = json.loads((self.project_dir / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(run_log["content_summary"]["checked_cases"], 1)
        self.assertEqual(run_log["case_runs"][0]["status"], "success")
        self.assertIn("content_score", run_log["case_runs"][0])
        self.assertEqual(run_log["checklist_errors"], [])
        self.assertTrue(run_log["case_runs"][0]["checklist_path"].endswith("CASE01.checklist.json"))

    def test_content_problems_do_not_fail_the_run(self):
        self._write_checklist("CASE01")
        bad_record = self._valid_record(case_features="患者乏力2周；血压正常，尿量正常。")
        exit_code = self._run_with_record(bad_record)
        self.assertEqual(exit_code, 0)

        output = json.loads((self.project_dir / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["status"], "success")

        run_log = json.loads((self.project_dir / "run_log.json").read_text(encoding="utf-8"))
        case_run = run_log["case_runs"][0]
        self.assertFalse(case_run["content_passed"])
        self.assertIn("NEGATION_REWRITTEN", case_run["content_issues_by_code"])

    def test_checklist_facts_are_enforced(self):
        self._write_checklist(
            "CASE01",
            required_facts=[{"fact_id": "F01", "text": "患者有糖尿病病史10年", "source_ids": ["S01"]}],
        )
        exit_code = self._run_with_record(self._valid_record())
        self.assertEqual(exit_code, 0)

        validation = json.loads((self.project_dir / "validation_report.json").read_text(encoding="utf-8"))
        codes = {issue["code"] for issue in validation["cases"][0]["content_check"]["issues"]}
        self.assertIn("CHECKLIST_FACT_MISSING", codes)

    def test_broken_checklist_is_non_fatal(self):
        (self.project_dir / "checklists").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "checklists" / "CASE01.checklist.json").write_text(
            json.dumps({"case_id": "CASE01", "required_facts": "not-a-list"}, ensure_ascii=False),
            encoding="utf-8",
        )
        exit_code = self._run_with_record(self._valid_record())
        self.assertEqual(exit_code, 0)

        run_log = json.loads((self.project_dir / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(len(run_log["checklist_errors"]), 1)
        self.assertEqual(run_log["checklist_errors"][0]["case_id"], "CASE01")
        self.assertIn("checklist_error", run_log["case_runs"][0])
        # The case still runs and its content check still happens without the checklist.
        output = json.loads((self.project_dir / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["status"], "success")
        validation = json.loads((self.project_dir / "validation_report.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(validation["cases"][0]["content_check"])
        self.assertIsNone(validation["cases"][0]["content_check"]["checklist"])


if __name__ == "__main__":
    unittest.main()
