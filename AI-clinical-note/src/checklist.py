"""Checklist loading and text-alignment helpers used by content-level validation.

The competition rules (rules/8-clinical.txt) require the generated first-course
record to be traceable to its sources: values must keep their original numbers
and units, "not provided" must not become "normal", preliminary impressions must
not be upgraded to confirmed diagnoses, and planned actions must not be written
as already-performed treatment.  ``templates/checklist.template.json`` and
``examples/DEMO001.checklist.json`` describe the per-case checklist that makes
those expectations machine-checkable:

  required_facts   facts that must survive into the record
  numeric_facts    concept + value + unit that must be reproduced faithfully
  negation_scope   statements that limit how a negative finding may be used
  missing_items    information the sources explicitly lack
  forbidden_claims claims the record must not make
  review_targets   items that should surface in review_items

This module contains only loading/normalisation/alignment primitives; the rule
engine lives in ``src/content_check.py``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

CHECKLIST_REQUIRED_KEYS = (
    "case_id",
    "required_facts",
    "numeric_facts",
    "negation_scope",
    "missing_items",
    "forbidden_claims",
    "review_targets",
)
CHECKLIST_OPTIONAL_KEYS = (
    "parent_case_id",
    "split",
    "origin",
    "reviewer",
    "clinical_review_status",
    "review_note",
)


class ChecklistError(ValueError):
    """Raised when a checklist file is missing, malformed or internally inconsistent."""


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

_PUNCT_CATEGORIES = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Sk", "Sm", "So", "Zs", "Cc", "Cf", "Co"}
_PUNCT_EXTRA = set("，。；：、！？（）()[]{}\"'“”‘’《》<>~!@#$%^&*_+=|\\/·…—-‐‑‒–—― \t\r\n")


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/whitespace so paraphrases can be aligned."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        if ch in _PUNCT_EXTRA:
            continue
        if unicodedata.category(ch) in _PUNCT_CATEGORIES:
            continue
        out.append(ch.lower())
    return "".join(out)


def bigrams(text: str) -> set[str]:
    """Return the set of character bigrams (after normalisation)."""
    norm = normalize_text(text)
    if len(norm) < 2:
        return {norm} if norm else set()
    return {norm[i : i + 2] for i in range(len(norm) - 1)}


def coverage(fact: str, text: str) -> float:
    """Fraction of ``fact`` bigrams that also occur in ``text`` (0..1).

    Short facts (< 4 normalised chars) fall back to a substring test, which is
    both cheap and exact for terms such as "血压" or "尿量".
    """
    fact_norm = normalize_text(fact)
    text_norm = normalize_text(text)
    if not fact_norm:
        return 0.0
    if not text_norm:
        return 0.0
    if len(fact_norm) < 4:
        return 1.0 if fact_norm in text_norm else 0.0
    fb = bigrams(fact)
    tb = bigrams(text)
    if not fb:
        return 0.0
    return len(fb & tb) / len(fb)


def best_field_coverage(fact: str, fields: dict[str, str]) -> tuple[str | None, float]:
    """Return (field_name, coverage) for the field that best contains ``fact``."""
    best_field: str | None = None
    best_cov = 0.0
    for name, text in fields.items():
        cov = coverage(fact, text)
        if cov > best_cov:
            best_field, best_cov = name, cov
    return best_field, round(best_cov, 4)


_SENTENCE_SPLIT_RE = re.compile(r"[。；;！？!?\n]+")
_CLAUSE_SPLIT_RE = re.compile(r"[，,、；;。\n]+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text or "") if part.strip()]


def split_clauses(text: str) -> list[str]:
    return [part.strip() for part in _CLAUSE_SPLIT_RE.split(text or "") if part.strip()]


def context_around(text: str, needle: str, span: int = 12) -> Iterable[str]:
    """Yield the ±span character windows around every occurrence of ``needle``."""
    if not needle:
        return
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            return
        yield text[max(0, pos - span) : pos + len(needle) + span]
        start = pos + len(needle)


# --------------------------------------------------------------------------- #
# Measurements (value + unit) extraction
# --------------------------------------------------------------------------- #

# Longest units first so "×10^9/L" wins over "/L" style prefixes.
_UNITS = [
    "×10^9/L",
    "×10^12/L",
    "×10^9/L",
    "*10^9/L",
    "*10^12/L",
    "pg/mL",
    "ng/mL",
    "μg/mL",
    "µg/mL",
    "ug/mL",
    "μg/L",
    "µg/L",
    "ug/L",
    "ng/L",
    "mIU/L",
    "IU/mL",
    "mmol/L",
    "μmol/L",
    "µmol/L",
    "umol/L",
    "mg/dL",
    "mg/L",
    "g/L",
    "IU/L",
    "U/L",
    "mU/L",
    "mmol/L",
    "mmHg",
    "cmH2O",
    "次/分",
    "bpm",
    "ml/min",
    "mL/min",
    "kg/m2",
    "kg/m²",
    "mV",
    "mm",
    "cm",
    "kg",
    "℃",
    "°C",
    "%",
    "分钟",
    "小时",
    "周",
    "月",
    "年",
    "岁",
    "天",
    "日",
    "次",
    "分",
    "秒",
    "ml",
    "mL",
    "g",
    "mg",
    "L",
]

_UNIT_ALT = "|".join(re.escape(u) for u in sorted(set(_UNITS), key=len, reverse=True))
_MEASUREMENT_RE = re.compile(
    rf"(?P<a>\d+(?:\.\d+)?)(?:\s*[/\-~–—~至到]\s*(?P<b>\d+(?:\.\d+)?))?\s*(?P<u>{_UNIT_ALT})"
)


def normalize_unit(unit: str) -> str:
    unit = unicodedata.normalize("NFKC", unit or "").strip().lower()
    unit = unit.replace("µ", "μ").replace("×", "x").replace("*", "x").replace("／", "/")
    unit = unit.replace("°c", "℃").replace(" ", "")
    return unit


def _number_token(raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value == int(value):
        return str(int(value))
    return repr(round(value, 6)).rstrip("0").rstrip(".") or "0"


@dataclass(frozen=True)
class Measurement:
    """A numeric value with its unit, as written in the text."""

    value: str  # canonical numeric token, e.g. "82", "38.6"
    unit: str  # normalised unit, e.g. "g/l"
    raw: str

    def matches(self, other: "Measurement") -> bool:
        if self.unit != other.unit:
            return False
        try:
            return abs(float(self.value) - float(other.value)) < 1e-9
        except ValueError:
            return self.value == other.value

    @property
    def key(self) -> str:
        return f"{self.value} {self.unit}".strip()


def extract_measurements(text: str) -> list[Measurement]:
    """Extract every ``value + unit`` pair from ``text`` (normalised units)."""
    text = unicodedata.normalize("NFKC", text or "")
    found: list[Measurement] = []
    seen_spans: set[tuple[int, int]] = set()
    for match in _MEASUREMENT_RE.finditer(text):
        span = match.span()
        if span in seen_spans:
            continue
        seen_spans.add(span)
        unit = normalize_unit(match.group("u"))
        raw = match.group(0)
        found.append(Measurement(_number_token(match.group("a")), unit, raw))
        if match.group("b"):
            found.append(Measurement(_number_token(match.group("b")), unit, raw))
    return found


def measurement_keys(text: str) -> set[str]:
    return {m.key for m in extract_measurements(text)}


def find_measurement(measurements: Iterable[Measurement], value: str, unit: str) -> Measurement | None:
    """Locate a measurement with the given value/unit (unit "" matches any unit)."""
    target_value = _number_token(value)
    unit_norm = normalize_unit(unit)
    for item in measurements:
        same_value = item.value == target_value
        if not same_value:
            try:
                same_value = abs(float(item.value) - float(target_value)) < 1e-9
            except ValueError:
                same_value = False
        if not same_value:
            continue
        if not unit_norm or item.unit == unit_norm:
            return item
    return None


# --------------------------------------------------------------------------- #
# Missing-information / negation extraction
# --------------------------------------------------------------------------- #

_MISSING_MARKERS = (
    "未提供",
    "未记录",
    "未查",
    "未测",
    "未做",
    "未完善",
    "尚未完成",
    "尚未进行",
    "尚未做",
    "尚未回报",
    "未进行",
    "未见记录",
    "不详",
    "未明确",
)
_MISSING_MARKER_RE = re.compile("|".join(_MISSING_MARKERS))
_TRIM_CHARS = "，,；;。.、 \t\n：:（）()"
_GENERIC_ITEM_WORDS = {
    "其他",
    "其余",
    "相关",
    "明显",
    "异常",
    "检查",
    "结果",
    "情况",
    "记录",
    "资料",
    "内容",
    "项目",
    "指标",
    "体征",
    "病史",
}
_ITEM_TAIL_RE = re.compile(r"(?:等|及其他.*|及.*|等.*)$")


_ITEM_SPLIT_RE = re.compile(r"[、，,]|以及|及|(?<!饱)和|或")
_ITEM_CUT_RE = re.compile(r"(?:等(?:其他)?.*|及其他.*|等.*)$")
_ITEM_SUFFIXES = ("结果", "报告", "情况", "记录")


def _normalize_item(item: str) -> str:
    """Strip generic suffixes so "胸部CT" and "胸部CT结果" collapse into one item."""
    for suffix in _ITEM_SUFFIXES:
        if len(item) > len(suffix) + 1 and item.endswith(suffix):
            return item[: -len(suffix)]
    return item


def _marker_regions(sentence: str) -> list[str]:
    """Return the text regions governed by a missing-information marker.

    A marker can precede its items ("未提供血压、尿量") or follow them
    ("既往病史、药物史未提供"); both forms are handled.
    """
    markers = list(_MISSING_MARKER_RE.finditer(sentence))
    regions: list[str] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(sentence)
        region = sentence[start:end].strip(_TRIM_CHARS)
        if not region:
            head_start = markers[index - 1].end() if index else 0
            region = sentence[head_start : marker.start()].strip(_TRIM_CHARS)
        if region:
            regions.append(region)
    return regions


def _split_items(region: str) -> list[str]:
    items: list[str] = []
    for piece in _ITEM_SPLIT_RE.split(region):
        piece = piece.strip(_TRIM_CHARS)
        piece = _ITEM_CUT_RE.sub("", piece).strip(_TRIM_CHARS)
        if not 2 <= len(piece) <= 14:
            continue
        if piece in _GENERIC_ITEM_WORDS:
            continue
        if any(word in piece for word in ("其他", "其余")):
            continue
        if piece not in items:
            items.append(piece)
    return items


def extract_missing_items(sources: Iterable[dict[str, Any]]) -> list[str]:
    """Extract item names that the sources explicitly mark as not provided."""
    items: list[str] = []
    seen: set[str] = set()
    for src in sources or []:
        content = src.get("content") if isinstance(src, dict) else None
        if not isinstance(content, str):
            continue
        for sentence in split_sentences(content):
            for region in _marker_regions(sentence):
                for piece in _split_items(region):
                    key = _normalize_item(piece)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(piece)
    return items


@dataclass
class NegativeStatement:
    """A source sentence that limits how a negative observation may be used."""

    source_id: str
    sentence: str
    kind: str  # "not_provided" or "negative_observation"
    item: str


_NEG_OBS_RE = re.compile(r"(?:未见明确|未见|无明显|无|否认|未闻及|未触及)([^，,；;。]{2,20})")


def extract_negative_statements(sources: Iterable[dict[str, Any]]) -> list[NegativeStatement]:
    statements: list[NegativeStatement] = []
    for src in sources or []:
        content = src.get("content") if isinstance(src, dict) else None
        if not isinstance(content, str):
            continue
        sid = str(src.get("source_id", ""))
        for sentence in split_sentences(content):
            for region in _marker_regions(sentence):
                region_items = _split_items(region)
                for item in region_items:
                    statements.append(NegativeStatement(sid, sentence, "not_provided", item))
                if not region_items and 2 <= len(region) <= 20:
                    statements.append(NegativeStatement(sid, sentence, "not_provided", region))
            marker_free = _MISSING_MARKER_RE.sub("", sentence)
            for clause in split_clauses(marker_free):
                match = _NEG_OBS_RE.search(clause)
                if not match:
                    continue
                item = match.group(1).strip(_TRIM_CHARS)
                item = _ITEM_TAIL_RE.sub("", item).strip(_TRIM_CHARS)
                if 2 <= len(item) <= 20:
                    statements.append(NegativeStatement(sid, sentence, "negative_observation", item))
    return statements


# --------------------------------------------------------------------------- #
# Forbidden-claim compilation
# --------------------------------------------------------------------------- #

_CLAIM_MARKERS = ("正常", "阴性", "无异常", "未见异常", "已排除", "排除", "确诊", "明确诊断", "已完成", "完成", "已实施", "已执行")
_LEADING_FILLERS = ("已经", "已", "确定", "明确", "提示", "考虑")


@dataclass
class ForbiddenPattern:
    claim: str
    regex: re.Pattern[str]


@dataclass
class CompiledForbiddenClaims:
    patterns: list[ForbiddenPattern] = field(default_factory=list)
    manual_review: list[tuple[str, str]] = field(default_factory=list)


def compile_forbidden_claims(claims: Iterable[str]) -> CompiledForbiddenClaims:
    """Turn natural-language forbidden claims into the patterns we can check.

    Claims that cannot be reduced to a checkable pattern are returned in
    ``manual_review`` instead of being silently dropped.
    """
    compiled = CompiledForbiddenClaims()
    for raw_claim in claims or []:
        claim = str(raw_claim).strip()
        if not claim:
            continue
        produced = 0
        for alternative in re.split(r"或|或者|以及", claim):
            alternative = alternative.strip()
            core = alternative
            for filler in _LEADING_FILLERS:
                if core.startswith(filler):
                    core = core[len(filler) :]
                    break
            if not core:
                continue
            concept = None
            marker = None
            for candidate in _CLAIM_MARKERS:
                pos = core.find(candidate)
                if pos >= 2:
                    concept, marker = core[:pos], candidate
                    break
            if concept and marker:
                prefix = re.escape(concept[-10:])
                compiled.patterns.append(
                    ForbiddenPattern(alternative, re.compile(prefix + r".{0,6}" + re.escape(marker)))
                )
                produced += 1
                continue
            if core.startswith(("确诊", "明确诊断")):
                compiled.patterns.append(
                    ForbiddenPattern(
                        alternative,
                        re.compile(r"(?:确诊|明确诊断|诊断明确).{0,8}" + re.escape(core[2:] or "")),
                    )
                )
                produced += 1
                continue
            if core.startswith("完成"):
                noun = core[2:]
                compiled.patterns.append(
                    ForbiddenPattern(
                        alternative,
                        re.compile(r"(?:已经?|已)?完成.{0,10}" + re.escape(noun or "")),
                    )
                )
                produced += 1
                continue
        if not produced and not any(p.claim == claim for p in compiled.patterns):
            compiled.manual_review.append((claim, "无法自动判定，需人工复核"))
    return compiled


# --------------------------------------------------------------------------- #
# Checklist loading
# --------------------------------------------------------------------------- #

CHECKLIST_FILENAMES = ("{case_id}.checklist.json", "{case_id}.json")
CHECKLIST_DIRS = ("checklists", "examples")


def resolve_checklist_path(case_id: str, base_dir: Path | str) -> Path | None:
    """Find the checklist for ``case_id`` under ``base_dir`` if any."""
    base = Path(base_dir)
    for dirname in CHECKLIST_DIRS:
        folder = base / dirname
        if not folder.is_dir():
            continue
        for template in CHECKLIST_FILENAMES:
            candidate = folder / template.format(case_id=case_id)
            if candidate.is_file():
                return candidate
    return None


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChecklistError(f"{where} must be a non-empty string")
    return value.strip()


def _require_str_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ChecklistError(f"{where} must be an array of strings")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(_require_str(item, f"{where}[{index}]"))
    return out


def validate_checklist(data: Any, path: Path | None = None) -> dict[str, Any]:
    """Validate checklist structure and return a normalised copy."""
    where = str(path) if path else "checklist"
    if not isinstance(data, dict):
        raise ChecklistError(f"{where}: top level must be a JSON object")

    missing = [key for key in CHECKLIST_REQUIRED_KEYS if key not in data]
    if missing:
        raise ChecklistError(f"{where}: missing required keys: {missing}")

    normalized: dict[str, Any] = {
        "case_id": _require_str(data["case_id"], f"{where}:case_id"),
    }

    facts = data["required_facts"]
    if not isinstance(facts, list):
        raise ChecklistError(f"{where}:required_facts must be an array")
    normalized["required_facts"] = []
    for index, item in enumerate(facts):
        if not isinstance(item, dict):
            raise ChecklistError(f"{where}:required_facts[{index}] must be an object")
        normalized["required_facts"].append(
            {
                "fact_id": item.get("fact_id") or f"F{index + 1:02d}",
                "text": _require_str(item.get("text"), f"{where}:required_facts[{index}].text"),
                "source_ids": _require_str_list(item.get("source_ids"), f"{where}:required_facts[{index}].source_ids"),
            }
        )

    numbers = data["numeric_facts"]
    if not isinstance(numbers, list):
        raise ChecklistError(f"{where}:numeric_facts must be an array")
    normalized["numeric_facts"] = []
    for index, item in enumerate(numbers):
        if not isinstance(item, dict):
            raise ChecklistError(f"{where}:numeric_facts[{index}] must be an object")
        value = item.get("value")
        if isinstance(value, (int, float)):
            value = str(value)
        unit = item.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise ChecklistError(f"{where}:numeric_facts[{index}].unit must be a string")
        normalized["numeric_facts"].append(
            {
                "concept": _require_str(item.get("concept"), f"{where}:numeric_facts[{index}].concept"),
                "value": _require_str(value, f"{where}:numeric_facts[{index}].value"),
                "unit": (unit or "").strip(),
                "source_ids": _require_str_list(item.get("source_ids"), f"{where}:numeric_facts[{index}].source_ids"),
            }
        )

    for key in ("negation_scope", "missing_items", "forbidden_claims", "review_targets"):
        normalized[key] = _require_str_list(data[key], f"{where}:{key}")

    for key in CHECKLIST_OPTIONAL_KEYS:
        if key in data:
            normalized[key] = data[key]

    normalized["_path"] = str(path) if path else None
    return normalized


def load_checklist(path: Path | str | None) -> dict[str, Any] | None:
    """Load and validate a checklist file; ``None`` means "no checklist"."""
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        raise ChecklistError(f"checklist file not found: {target}")
    try:
        raw = json.loads(target.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ChecklistError(f"{target}: invalid JSON: {exc}") from exc
    return validate_checklist(raw, target)


def load_checklist_for_case(case_id: str, base_dir: Path | str) -> dict[str, Any] | None:
    """Resolve and load the checklist for a case, or return None."""
    path = resolve_checklist_path(case_id, base_dir)
    if path is None:
        return None
    checklist = load_checklist(path)
    if checklist is not None and checklist.get("case_id") != case_id:
        raise ChecklistError(
            f"{path}: checklist case_id '{checklist.get('case_id')}' does not match input case '{case_id}'"
        )
    return checklist
