"""Unit tests for src.validate module."""

import json
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.validate import (
    TaskInputError,
    parse_json_content,
    validate_case,
    validate_input_document,
    validate_medical_record,
)


class TestValidateInputDocument(unittest.TestCase):
    def test_valid_input(self):
        doc = [
            {"case_id": "C001", "record_cutoff": "T0", "sources": []},
            {"case_id": "C002", "record_cutoff": "T0", "sources": []},
        ]
        # Should not raise
        validate_input_document(doc)

    def test_empty_list_is_valid(self):
        # Empty array [] must be accepted at task level
        validate_input_document([])

    def test_non_list_fails(self):
        with self.assertRaises(TaskInputError):
            validate_input_document({"case_id": "C001"})

    def test_non_dict_item_fails(self):
        with self.assertRaises(TaskInputError):
            validate_input_document(["not a dict"])

    def test_missing_case_id_fails(self):
        with self.assertRaises(TaskInputError):
            validate_input_document([{"record_cutoff": "T0"}])

    def test_empty_or_whitespace_case_id_fails(self):
        with self.assertRaises(TaskInputError):
            validate_input_document([{"case_id": "   "}])

    def test_duplicate_case_id_fails(self):
        with self.assertRaises(TaskInputError):
            validate_input_document([
                {"case_id": "C001", "record_cutoff": "T0"},
                {"case_id": "C001", "record_cutoff": "T0"},
            ])


class TestValidateCase(unittest.TestCase):
    def test_valid_case(self):
        case = {
            "case_id": "C001",
            "record_cutoff": "T0",
            "sources": [
                {"source_id": "S01", "content": "Content S01"},
                {"source_id": "S02", "content": "Content S02"},
            ],
        }
        issues = validate_case(case)
        self.assertEqual(issues, [])

    def test_missing_record_cutoff(self):
        case = {"case_id": "C001", "sources": []}
        issues = validate_case(case)
        self.assertTrue(any(i["field"] == "record_cutoff" for i in issues))

    def test_duplicate_source_id(self):
        case = {
            "case_id": "C001",
            "record_cutoff": "T0",
            "sources": [
                {"source_id": "S01", "content": "A"},
                {"source_id": "S01", "content": "B"},
            ],
        }
        issues = validate_case(case)
        self.assertTrue(any("Duplicate source_id" in i["message"] for i in issues))


class TestParseJsonContent(unittest.TestCase):
    def test_clean_json(self):
        content = '{"key": "value"}'
        data, stripped = parse_json_content(content)
        self.assertEqual(data, {"key": "value"})
        self.assertFalse(stripped)

    def test_markdown_fence_stripped(self):
        content = "```json\n{\"key\": \"value\"}\n```"
        data, stripped = parse_json_content(content)
        self.assertEqual(data, {"key": "value"})
        self.assertTrue(stripped)

    def test_malformed_json_raises(self):
        content = "not json at all"
        with self.assertRaises(Exception):
            parse_json_content(content)


class TestValidateMedicalRecord(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "C001",
            "record_cutoff": "T0",
            "sources": [
                {"source_id": "S01", "content": "Patient has mild anemia"},
                {"source_id": "S02", "content": "Blood pressure 120/80"},
            ],
        }
        self.valid_record = {
            "case_features": "Features text",
            "diagnostic_discussion": "Discussion text",
            "plan": "Plan text",
            "evidence": [
                {"statement": "Patient has mild anemia", "source_ids": ["S01"], "kind": "fact"},
                {"statement": "Blood pressure is normal", "source_ids": ["S02"], "kind": "inference"},
                {"statement": "Summary judgment", "source_ids": ["S01", "S02"], "kind": "inference"},
            ],
            "review_items": ["Check previous labs"],
        }

    def test_valid_record(self):
        issues = validate_medical_record(self.valid_record, self.case)
        error_issues = [i for i in issues if i["severity"] == "error"]
        self.assertEqual(error_issues, [])

    def test_invalid_source_id_detected(self):
        bad_record = dict(self.valid_record)
        bad_record["evidence"] = [
            {"statement": "Text", "source_ids": ["S99_UNKNOWN"], "kind": "fact"}
        ]
        issues = validate_medical_record(bad_record, self.case)
        errors = [i for i in issues if i["severity"] == "error"]
        self.assertTrue(any(e["code"] == "INVALID_SOURCE_REFERENCE" for e in errors))

    def test_empty_string_field_detected(self):
        bad_record = dict(self.valid_record)
        bad_record["plan"] = "   "
        issues = validate_medical_record(bad_record, self.case)
        errors = [i for i in issues if i["severity"] == "error"]
        self.assertTrue(any(e["field"] == "plan" for e in errors))

    def test_prohibited_extra_fields_detected(self):
        bad_record = dict(self.valid_record)
        bad_record["patient_id"] = "12345"
        issues = validate_medical_record(bad_record, self.case)
        errors = [i for i in issues if i["severity"] == "error"]
        self.assertTrue(any(e["field"] == "additionalProperties" for e in errors))

    def test_invalid_kind_detected(self):
        bad_record = dict(self.valid_record)
        bad_record["evidence"] = [
            {"statement": "Text", "source_ids": ["S01"], "kind": "guesswork"}
        ]
        issues = validate_medical_record(bad_record, self.case)
        errors = [i for i in issues if i["severity"] == "error"]
        self.assertTrue(any("kind must be 'fact' or 'inference'" in e["message"] for e in errors))


if __name__ == "__main__":
    unittest.main()
