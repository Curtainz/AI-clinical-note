"""Unit tests for src/checklist.py (loading + text-alignment primitives)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.checklist import (
    ChecklistError,
    compile_forbidden_claims,
    coverage,
    extract_measurements,
    extract_missing_items,
    extract_negative_statements,
    find_measurement,
    load_checklist_for_case,
    measurement_keys,
    normalize_unit,
    split_clauses,
    validate_checklist,
)


class TestCoverage(unittest.TestCase):
    def test_identical_text_is_fully_covered(self):
        fact = "血红蛋白82g/L"
        self.assertAlmostEqual(coverage(fact, fact), 1.0)

    def test_unrelated_text_has_low_coverage(self):
        self.assertLess(coverage("血红蛋白82g/L", "患者咳嗽咳痰伴发热三天"), 0.2)

    def test_short_fact_uses_substring(self):
        self.assertEqual(coverage("血压", "入院时未提供血压及尿量"), 1.0)
        self.assertEqual(coverage("尿量", "未提供血压"), 0.0)

    def test_paraphrase_is_partially_covered(self):
        cov = coverage("血红蛋白82g/L，实验室标低", "实验室检查示血红蛋白82g/L（标低）")
        self.assertGreater(cov, 0.6)


class TestMeasurements(unittest.TestCase):
    def test_value_and_unit_are_extracted(self):
        measurements = extract_measurements("实验室检查示血红蛋白82g/L，肌酐185μmol/L。")
        keys = {m.key for m in measurements}
        self.assertIn("82 g/l", keys)
        self.assertIn("185 μmol/l", keys)

    def test_blood_pressure_splits_into_two_values(self):
        measurements = extract_measurements("血压145/92mmHg")
        values = {m.value for m in measurements}
        self.assertEqual(values, {"145", "92"})
        self.assertTrue(all(m.unit == "mmhg" for m in measurements))

    def test_scientific_unit_normalised(self):
        keys = measurement_keys("白细胞13.5×10^9/L")
        self.assertIn("13.5 x10^9/l", keys)

    def test_unit_normalisation(self):
        self.assertEqual(normalize_unit("µmol/L"), "μmol/l")
        self.assertEqual(normalize_unit("°C"), "℃")
        self.assertEqual(normalize_unit("×10^9/L"), "x10^9/l")

    def test_find_measurement(self):
        measurements = extract_measurements("体温38.6℃，心率88次/分")
        self.assertIsNotNone(find_measurement(measurements, "38.6", "℃"))
        self.assertIsNotNone(find_measurement(measurements, "38.6", ""))
        self.assertIsNone(find_measurement(measurements, "38.6", "mmol/L"))
        self.assertIsNone(find_measurement(measurements, "39.0", "℃"))

    def test_microgram_per_litre_is_extracted(self):
        keys = measurement_keys("肌钙蛋白I阴性（<0.02μg/L）")
        self.assertIn("0.02 μg/l", keys)


class TestMissingAndNegation(unittest.TestCase):
    def test_missing_items_are_extracted(self):
        case = json.loads((BASE_DIR / "examples" / "DEMO001.input.json").read_text(encoding="utf-8-sig"))
        if isinstance(case, list):
            case = case[0]
        items = extract_missing_items(case["sources"])
        self.assertIn("血压", items)
        self.assertIn("尿量", items)

    def test_negative_statements_classified(self):
        case = json.loads((BASE_DIR / "examples" / "DEMO001.input.json").read_text(encoding="utf-8-sig"))
        if isinstance(case, list):
            case = case[0]
        statements = extract_negative_statements(case["sources"])
        kinds = {statement.kind for statement in statements}
        self.assertIn("not_provided", kinds)
        self.assertIn("negative_observation", kinds)

    def test_clause_splitting(self):
        self.assertEqual(split_clauses("未提供血压、尿量及既往病史"), ["未提供血压", "尿量及既往病史"])

    def test_items_with_he_inside_words_are_not_truncated(self):
        items = extract_missing_items(
            [{"source_id": "S01", "content": "未提供血压及血氧饱和度，胸部CT结果未回报。"}]
        )
        self.assertIn("血氧饱和度", items)
        self.assertNotIn("血氧饱", items)

    def test_item_suffix_variants_are_deduped(self):
        items = extract_missing_items(
            [
                {"source_id": "S01", "content": "未提供胸部CT结果。"},
                {"source_id": "S02", "content": "胸部CT尚未完成，痰培养未回报。"},
            ]
        )
        compact = [item.replace("结果", "") for item in items]
        self.assertEqual(len(compact), len(set(compact)))


class TestForbiddenClaims(unittest.TestCase):
    def test_concept_marker_claim_compiles(self):
        compiled = compile_forbidden_claims(["血压正常或尿量正常"])
        matched = [pattern for pattern in compiled.patterns if "血压" in pattern.claim]
        self.assertTrue(matched)
        self.assertTrue(matched[0].regex.search("查体血压正常，尿量正常"))

    def test_unparsable_claim_goes_to_manual_review(self):
        compiled = compile_forbidden_claims(["不得编造无依据的结论"])
        self.assertEqual(compiled.patterns, [])
        self.assertEqual(len(compiled.manual_review), 1)


class TestChecklistLoading(unittest.TestCase):
    def test_demo_checklists_are_valid(self):
        for case_id in ("DEMO001", "DEMO002", "DEMO003"):
            checklist = load_checklist_for_case(case_id, BASE_DIR)
            self.assertIsNotNone(checklist, case_id)
            self.assertEqual(checklist["case_id"], case_id)
            self.assertTrue(checklist["required_facts"])
            self.assertTrue(checklist["numeric_facts"])

    def test_unknown_case_has_no_checklist(self):
        self.assertIsNone(load_checklist_for_case("NO_SUCH_CASE", BASE_DIR))

    def test_malformed_checklist_raises(self):
        with self.assertRaises(ChecklistError):
            validate_checklist({"case_id": "X"})
        with self.assertRaises(ChecklistError):
            validate_checklist({"case_id": "X", "required_facts": "not-a-list"})

    def test_optional_keys_preserved(self):
        data = json.loads((BASE_DIR / "examples" / "DEMO002.checklist.json").read_text(encoding="utf-8"))
        checklist = validate_checklist(data)
        self.assertEqual(checklist["split"], "dev")
        self.assertEqual(checklist["clinical_review_status"], "not_expert_reviewed")

    def test_case_id_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checklists").mkdir()
            (root / "checklists" / "CASE01.checklist.json").write_text(
                json.dumps(
                    {
                        "case_id": "OTHER",
                        "required_facts": [],
                        "numeric_facts": [],
                        "negation_scope": [],
                        "missing_items": [],
                        "forbidden_claims": [],
                        "review_targets": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ChecklistError):
                load_checklist_for_case("CASE01", root)


if __name__ == "__main__":
    unittest.main()
