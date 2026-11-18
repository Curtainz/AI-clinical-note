"""Unit tests for tools/collect_training_data.py (run outputs -> SFT dataset)."""

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.collect_training_data import collect, main

TMP_ROOT = BASE_DIR / "tests" / "_tmp"
SOURCES = [
    {"source_id": "S01", "content": "乏力2周，活动后心悸；现有记录未见黑便、肉眼血尿"},
    {"source_id": "S02", "content": "血红蛋白82g/L，实验室标低；肌酐185μmol/L，实验室标高"},
]


def make_case(case_id: str) -> dict:
    return {"case_id": case_id, "record_cutoff": "T0", "sources": SOURCES}


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


def write_run(root: Path, case_ids: list[str], failed: set[str] | None = None) -> None:
    failed = failed or set()
    (root / "input.json").write_text(
        json.dumps([make_case(case_id) for case_id in case_ids], ensure_ascii=False), encoding="utf-8"
    )
    entries = []
    for case_id in case_ids:
        if case_id in failed:
            entries.append(
                {"case_id": case_id, "status": "failed", "medical_record": None, "error": {"code": "X"}}
            )
        else:
            entries.append(
                {"case_id": case_id, "status": "success", "medical_record": make_record(), "error": None}
            )
    (root / "output.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    (root / "validation_report.json").write_text(
        json.dumps({"run_id": "RUN1", "cases": []}, ensure_ascii=False), encoding="utf-8"
    )
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "active.txt").write_text("system prompt", encoding="utf-8")


def run_tool(root: Path, out: Path, *extra: str) -> int:
    return main(
        [
            "--input", str(root / "input.json"),
            "--output", str(root / "output.json"),
            "--report", str(root / "validation_report.json"),
            "--out-dir", str(out),
            "--prompt", str(root / "prompts" / "active.txt"),
            "--project-root", str(root),
            "--min-score", "0",
            *extra,
        ]
    )


class TestCollectTrainingData(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TMP_ROOT / f"collect-{uuid.uuid4().hex[:10]}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_filters_failed_cases_and_writes_dataset(self):
        root = self.root
        write_run(root, ["C1", "C2", "C3"], failed={"C2", "C3"})
        out = root / "out"
        self.assertEqual(run_tool(root, out), 0)

        reference = json.loads((out / "reference.json").read_text(encoding="utf-8"))
        self.assertEqual(list(reference), ["C1"])

        train = [
            json.loads(line)
            for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["case_id"], "C1")
        self.assertEqual([m["role"] for m in train[0]["messages"]], ["system", "user", "assistant"])

        stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
        rejected = {row["case_id"]: row["reason"] for row in stats["rejected"]}
        self.assertIn("C2", rejected)
        self.assertIn("C3", rejected)
        self.assertEqual(stats["accepted_cases"], 1)
        self.assertEqual(stats["run_id"], "RUN1")

        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["run_id"], "RUN1")
        self.assertEqual(provenance["cases"][0]["case_id"], "C1")
        self.assertIn("input_sha256", provenance)

    def test_split_is_deterministic(self):
        root = self.root
        case_ids = [f"C{index:02d}" for index in range(1, 13)]
        write_run(root, case_ids)
        self.assertEqual(run_tool(root, root / "out1", "--val-ratio", "0.5"), 0)
        self.assertEqual(run_tool(root, root / "out2", "--val-ratio", "0.5"), 0)

        def ids(path: Path) -> list[str]:
            return [
                json.loads(line)["case_id"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]

        self.assertEqual(ids(root / "out1" / "train.jsonl"), ids(root / "out2" / "train.jsonl"))
        self.assertEqual(ids(root / "out1" / "val.jsonl"), ids(root / "out2" / "val.jsonl"))
        self.assertTrue(ids(root / "out1" / "val.jsonl"))

    def test_rejects_out_of_range_min_score(self):
        root = self.root
        write_run(root, ["C1"])
        rc = main(
            [
                "--input", str(root / "input.json"),
                "--output", str(root / "output.json"),
                "--out-dir", str(root / "out"),
                "--prompt", str(root / "prompts" / "active.txt"),
                "--project-root", str(root),
                "--min-score", "101",
            ]
        )
        self.assertEqual(rc, 2)

    def test_content_below_threshold_is_rejected(self):
        case = make_case("C1")
        outputs = {"C1": {"status": "success", "medical_record": make_record()}}
        accepted, provenance, rejected = collect([case], outputs, {}, Path("."), 101)
        self.assertEqual(accepted, {})
        self.assertEqual(provenance, [])
        self.assertIn("min_score", rejected[0]["reason"])

    def test_collect_helper_reports_missing_entries(self):
        case = make_case("C1")
        accepted, provenance, rejected = collect([case], {}, {}, Path("."), 0)
        self.assertEqual(accepted, {})
        self.assertEqual(provenance, [])
        self.assertEqual(rejected[0]["reason"], "no output.json entry")


if __name__ == "__main__":
    unittest.main()
