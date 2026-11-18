"""Content-level validation for generated first-course records.

Task 2 of the competition requires a run that is *reproducible* and that the
record stays faithful to the case sources.  ``src/validate.py`` enforces the
JSON schema and source-id references; this module checks the *content*:

  factual grounding   every value+unit in the record must exist in the sources
  fact preservation   checklist ``required_facts`` must survive into the record
  numeric fidelity    checklist ``numeric_facts`` must keep value and unit
  negation fidelity   "not provided" must not become "normal"/"denied"
  clinical boundary   plans must not be written as done; impressions must not
                      become confirmed diagnoses; nothing after the cutoff
  evidence quality    evidence numbers must be grounded in the cited sources
  review prompts      missing information must surface in ``review_items``

The checker is deliberately non-fatal: it never rejects a record and never
triggers extra model calls.  Results are attached to ``validation_report.json``
and ``run_log.json`` so graders/developers can audit them, and
``tools/check_output.py`` can produce the same report offline (including a
baseline-vs-improved comparison for the improvement-verification requirement).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.checklist import (
    ChecklistError,
    NegativeStatement,
    compile_forbidden_claims,
    coverage,
    extract_measurements,
    extract_missing_items,
    extract_negative_statements,
    find_measurement,
    measurement_keys,
    normalize_text,
    split_clauses,
)

CHECKER_VERSION = "1.0"

SECTIONS = ("case_features", "diagnostic_discussion", "plan")

# Category weights (sum to 100) used for the 0-100 content score.
CATEGORY_WEIGHTS = {
    "structure": 15,
    "factual": 25,
    "numeric": 20,
    "negation": 15,
    "clinical_boundary": 15,
    "evidence_review": 10,
}

# Which category an issue code belongs to, and how much one severity deducts.
_CODE_CATEGORY = {
    "RECORD_NOT_OBJECT": "structure",
    "FIELD_TOO_SHORT": "structure",
    "TOTAL_LENGTH_OUT_OF_RANGE": "structure",
    "PLACEHOLDER_TEXT": "structure",
    "SECTION_DUPLICATE": "structure",
    "CHECKLIST_FACT_MISSING": "factual",
    "CHECKLIST_FACT_PARTIAL": "factual",
    "MISSING_ITEM_UNMENTIONED": "factual",
    "NUMBER_NOT_TRACEABLE": "numeric",
    "UNIT_MISMATCH": "numeric",
    "CHECKLIST_NUMERIC_MISSING": "numeric",
    "CHECKLIST_NUMERIC_MISMATCH": "numeric",
    "NEGATION_REWRITTEN": "negation",
    "NEGATION_SCOPE_VIOLATION": "negation",
    "CHECKLIST_FORBIDDEN_CLAIM": "negation",
    "PLAN_AS_DONE": "clinical_boundary",
    "DIAGNOSIS_CONFIRMED": "clinical_boundary",
    "CUTOFF_VIOLATION": "clinical_boundary",
    "REVIEW_ITEMS_EMPTY": "evidence_review",
    "EVIDENCE_NUMBER_UNGROUNDED": "evidence_review",
    "EVIDENCE_FACT_NOT_GROUNDED": "evidence_review",
    "CHECKLIST_REVIEW_TARGET_MISSING": "evidence_review",
    "CHECKLIST_CASE_MISMATCH": "evidence_review",
}

_SEVERITY_WEIGHT = {"error": 1.0, "warning": 0.25, "info": 0.05}

_RECOMMENDED_MIN_TOTAL = 800
_RECOMMENDED_MAX_TOTAL = 1600

_PLACEHOLDER_RE = (
    r"[XxｘＸ]{2,}|待补充|待填写|TODO|TBD|填写|示例文本|占位|＜＞|<>"
)
_PLAN_AS_DONE_RE = (
    r"(?:已经?|已)(?:予|给予|给|行|使用|加用|安排|完成|实施|执行|开始|口服|静滴|静推|注射|输注|转入|收住|下达)"
    r"|(?:今日|入院后|急诊)(?:已|予|给予|行)"
)
_CONFIRMED_DIAGNOSIS_RE = r"确诊为|确诊|明确诊断|诊断明确|最终诊断|确定诊断为"
_PLAN_DONE_TERMS = ("已予", "已给予", "已行", "已完成", "已实施", "已执行", "已安排", "已开始", "已口服", "已静滴", "已注射")
_CUTOFF_TERMS = ("出院", "转归", "随访结果", "病理结果", "术后病理", "死亡")
_NORMAL_CLAIM_WORDS = ("正常", "阴性", "无异常", "未见异常", "已排除", "排除")
_PRESERVED_WORDS = ("未提供", "不详", "未见", "未记录", "待核", "待评估", "未做", "未查", "未完善", "需核", "必要时", "待明确", "未测")
_REVIEW_WORDS = ("待核", "复核", "确认", "补充", "核实", "待明确")


@dataclass
class ContentIssue:
    severity: str
    code: str
    field: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class ContentCheckResult:
    case_id: str
    issues: list[ContentIssue] = field(default_factory=list)
    manual_review: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    score: int = 100
    score_breakdown: dict[str, float] = field(default_factory=dict)
    checklist_path: str | None = None

    @property
    def errors(self) -> list[ContentIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ContentIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "checker_version": CHECKER_VERSION,
            "checklist": self.checklist_path,
            "passed": self.passed,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "issues": [issue.to_dict() for issue in self.issues],
            "manual_review": self.manual_review,
            "metrics": self.metrics,
        }


def _joined(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _evidence_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    entries = record.get("evidence")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _review_items(record: dict[str, Any]) -> list[str]:
    items = record.get("review_items")
    if not isinstance(items, list):
        return []
    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def _source_index(case: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_id: dict[str, str] = {}
    sources: list[dict[str, Any]] = []
    raw = case.get("sources")
    if isinstance(raw, list):
        for src in raw:
            if not isinstance(src, dict):
                continue
            sources.append(src)
            sid = src.get("source_id")
            content = src.get("content")
            if isinstance(sid, str) and isinstance(content, str):
                by_id[sid] = content
    return by_id, sources


def _check_structure(sections: dict[str, str], all_text: str, issues: list[ContentIssue]) -> dict[str, Any]:
    char_counts = {name: len(text) for name, text in sections.items()}
    total = sum(char_counts.values())
    char_counts["total"] = total

    for name, text in sections.items():
        if not text:
            continue
        if len(text) < 60:
            issues.append(
                ContentIssue("warning", "FIELD_TOO_SHORT", name, f"字段 '{name}' 仅 {len(text)} 字，可能不足以承载该部分内容")
            )

    if total and not _RECOMMENDED_MIN_TOTAL <= total <= _RECOMMENDED_MAX_TOTAL:
        issues.append(
            ContentIssue(
                "warning",
                "TOTAL_LENGTH_OUT_OF_RANGE",
                "root",
                f"三段正文合计 {total} 字，超出建议区间 {_RECOMMENDED_MIN_TOTAL}-{_RECOMMENDED_MAX_TOTAL} 字",
                {"total_chars": total},
            )
        )

    for name, text in sections.items():
        match = re.search(_PLACEHOLDER_RE, text)
        if match:
            issues.append(
                ContentIssue(
                    "error",
                    "PLACEHOLDER_TEXT",
                    name,
                    f"字段 '{name}' 含占位符/待填内容：'{match.group(0)}'",
                    {"matched": match.group(0)},
                )
            )

    # Near-duplicate sections suggest padding rather than analysis.
    names = list(sections)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left, right = sections[names[i]], sections[names[j]]
            if not left or not right:
                continue
            left_grams = {left[k : k + 2] for k in range(len(normalize_text(left)) - 1)}
            right_grams = {right[k : k + 2] for k in range(len(normalize_text(right)) - 1)}
            if not left_grams or not right_grams:
                continue
            jaccard = len(left_grams & right_grams) / len(left_grams | right_grams)
            if jaccard > 0.8:
                issues.append(
                    ContentIssue(
                        "warning",
                        "SECTION_DUPLICATE",
                        f"{names[i]}|{names[j]}",
                        f"'{names[i]}' 与 '{names[j]}' 内容高度重复（Jaccard={jaccard:.2f}），疑似凑长度",
                        {"jaccard": round(jaccard, 3)},
                    )
                )
    return char_counts


def _check_boundaries(
    sections: dict[str, str],
    full_text: str,
    source_text: str,
    issues: list[ContentIssue],
) -> None:
    """Plan-as-done, confirmed-diagnosis and post-cutoff wording checks.

    A phrase that is *already* reported as done in the sources (e.g. an
    emergency treatment) may legitimately be described as done, so every hit is
    cross-checked against the sources first.
    """
    plan = sections.get("plan", "")
    normalized_source = normalize_text(source_text)

    reported: list[str] = []

    def _already_reported(phrase: str) -> bool:
        return any(phrase in item or item in phrase for item in reported)

    for term in _PLAN_DONE_TERMS:
        contexts = list(_iter_contexts(plan, term))
        if not contexts or _already_reported(term):
            continue
        if normalize_text(term) and normalize_text(term) in normalized_source:
            continue
        reported.append(term)
        issues.append(
            ContentIssue(
                "error",
                "PLAN_AS_DONE",
                "plan",
                f"诊疗计划出现已实施表述 '{term}'，需改写为拟/建议/待评估",
                {"term": term, "context": contexts[0]},
            )
        )
        break

    for match in re.finditer(_PLAN_AS_DONE_RE, plan):
        phrase = match.group(0)
        if _already_reported(phrase):
            continue
        if normalize_text(phrase) in normalized_source:
            continue
        reported.append(phrase)
        issues.append(
            ContentIssue(
                "error",
                "PLAN_AS_DONE",
                "plan",
                f"诊疗计划出现已实施表述 '{phrase}'，需改写为拟/建议/待评估",
                {"term": phrase, "context": plan[max(0, match.start() - 12) : match.end() + 12]},
            )
        )
        if len(reported) >= 3:
            break

    for match in re.finditer(_CONFIRMED_DIAGNOSIS_RE, full_text):
        phrase = match.group(0)
        if normalize_text(phrase) in normalized_source:
            continue
        issues.append(
            ContentIssue(
                "error",
                "DIAGNOSIS_CONFIRMED",
                "root",
                f"出现确诊表述 '{phrase}'，入院初步判断不得写成确诊",
                {"term": phrase, "context": full_text[max(0, match.start() - 12) : match.end() + 12]},
            )
        )
        break

    for term in _CUTOFF_TERMS:
        if term in full_text and term not in source_text:
            issues.append(
                ContentIssue(
                    "warning",
                    "CUTOFF_VIOLATION",
                    "root",
                    f"出现截止点后才可能获得的表述 '{term}'，资料未支持",
                    {"term": term},
                )
            )
            break


def _iter_contexts(text: str, needle: str, span: int = 12) -> Iterable[str]:
    start = 0
    while needle:
        pos = text.find(needle, start)
        if pos < 0:
            return
        yield text[max(0, pos - span) : pos + len(needle) + span]
        start = pos + len(needle)


def _claim_right_after(
    text: str,
    needle: str,
    claim_words: tuple[str, ...],
    prefix: int = 10,
    tail: int = 8,
) -> str | None:
    """Return the item's context only when a claim word directly qualifies it.

    A claim such as "血压正常" puts the word right after the item, whereas
    "腹部查体未见异常，未提供血压" merely mentions an unrelated negative
    observation in the same sentence.  Occurrences preceded by an explicit
    "missing info" marker are treated as faithful preservation, never as a
    rewrite.
    """
    for match in re.finditer(re.escape(needle), text):
        before = text[max(0, match.start() - prefix) : match.start()]
        if any(word in before for word in _PRESERVED_WORDS):
            continue
        after = text[match.end() : match.end() + tail]
        if any(word in after for word in claim_words):
            return text[max(0, match.start() - prefix) : match.end() + tail]
    return None


def _check_numbers(
    sections: dict[str, str],
    evidence: list[dict[str, Any]],
    source_by_id: dict[str, str],
    source_text: str,
    checklist: dict[str, Any] | None,
    issues: list[ContentIssue],
) -> dict[str, Any]:
    all_text = "".join(sections.values())
    record_measurements = extract_measurements(all_text)
    source_measurements = extract_measurements(source_text)
    source_keys = {item.key for item in source_measurements}

    untraceable: list[dict[str, str]] = []
    for item in record_measurements:
        if item.key in source_keys:
            continue
        same_value_other_unit = any(
            item.value == other.value for other in source_measurements
        )
        context = next(_iter_contexts(all_text, item.raw), "")
        if same_value_other_unit:
            issues.append(
                ContentIssue(
                    "warning",
                    "UNIT_MISMATCH",
                    "root",
                    f"数值 '{item.raw}' 在资料中单位不同，需核对单位是否被改写",
                    {"value": item.value, "unit": item.unit, "context": context},
                )
            )
        else:
            untraceable.append({"value": item.value, "unit": item.unit, "raw": item.raw})
            issues.append(
                ContentIssue(
                    "warning",
                    "NUMBER_NOT_TRACEABLE",
                    "root",
                    f"数值 '{item.raw}' 未能回溯到任一原始资料",
                    {"value": item.value, "unit": item.unit, "context": context},
                )
            )

    # Evidence statements must be grounded in the *cited* sources.
    ungrounded_evidence: list[dict[str, Any]] = []
    fact_low_coverage: list[dict[str, Any]] = []
    for index, entry in enumerate(evidence):
        statement = entry.get("statement")
        source_ids = entry.get("source_ids")
        if not isinstance(statement, str) or not isinstance(source_ids, list):
            continue
        cited_ids = [sid for sid in source_ids if isinstance(sid, str) and sid in source_by_id]
        cited_text = "".join(source_by_id[sid] for sid in cited_ids)
        cited_keys = measurement_keys(cited_text) | measurement_keys("".join(source_ids))
        for measure in extract_measurements(statement):
            if measure.key in cited_keys:
                continue
            ungrounded_evidence.append(
                {
                    "evidence_index": index,
                    "value": measure.value,
                    "unit": measure.unit,
                    "raw": measure.raw,
                    "source_ids": cited_ids,
                }
            )
            issues.append(
                ContentIssue(
                    "error",
                    "EVIDENCE_NUMBER_UNGROUNDED",
                    f"evidence[{index}].statement",
                    f"依据说明中的数值 '{measure.raw}' 在所列 source_ids 中找不到",
                    {"evidence_index": index, "value": measure.value, "unit": measure.unit},
                )
            )
            break
        if entry.get("kind") == "fact" and cited_text:
            cov = coverage(statement, cited_text)
            if cov < 0.35:
                fact_low_coverage.append({"evidence_index": index, "coverage": cov})
                issues.append(
                    ContentIssue(
                        "warning",
                        "EVIDENCE_FACT_NOT_GROUNDED",
                        f"evidence[{index}].statement",
                        f"kind=fact 的依据与所列资料重合度仅 {cov:.2f}，需确认是否为原文事实",
                        {"evidence_index": index, "coverage": round(cov, 3)},
                    )
                )

    numeric_checks: list[dict[str, Any]] = []
    if checklist:
        expected_numbers = checklist.get("numeric_facts") or []
        for expected in expected_numbers:
            concept = expected.get("concept", "")
            value = expected.get("value", "")
            unit = expected.get("unit", "")
            found = find_measurement(record_measurements, value, unit)
            numeric_checks.append(
                {
                    "concept": concept,
                    "value": value,
                    "unit": unit,
                    "found": bool(found),
                }
            )
            if found:
                continue
            same_value = find_measurement(record_measurements, value, "")
            if same_value is not None:
                issues.append(
                    ContentIssue(
                        "error",
                        "CHECKLIST_NUMERIC_MISMATCH",
                        "root",
                        f"'{concept}' 的数值 {value}{unit} 单位被改写为 {same_value.unit}",
                        {"concept": concept, "expected_value": value, "expected_unit": unit, "actual_unit": same_value.unit},
                    )
                )
                continue
            concept_alternative = _find_concept_with_other_number(concept, all_text)
            if concept_alternative:
                issues.append(
                    ContentIssue(
                        "error",
                        "CHECKLIST_NUMERIC_MISMATCH",
                        "root",
                        f"'{concept}' 出现不同数值 '{concept_alternative.raw}'，资料为 {value}{unit}",
                        {
                            "concept": concept,
                            "expected_value": value,
                            "expected_unit": unit,
                            "actual": concept_alternative.raw,
                        },
                    )
                )
                continue
            issues.append(
                ContentIssue(
                    "error",
                    "CHECKLIST_NUMERIC_MISSING",
                    "root",
                    f"核对清单要求保留 '{concept} {value}{unit}'，正文中未找到",
                    {"concept": concept, "value": value, "unit": unit},
                )
            )

    metrics = {
        "record_measurements": len(record_measurements),
        "untraceable_numbers": untraceable,
        "numeric_checks": numeric_checks,
        "ungrounded_evidence_numbers": ungrounded_evidence,
        "fact_evidence_low_coverage": fact_low_coverage,
    }
    return metrics


def _find_concept_with_other_number(concept: str, text: str) -> Any:
    """Find a value near the concept name when the exact value is absent."""
    if not concept:
        return None
    pos = text.find(concept)
    if pos < 0:
        return None
    window = text[pos : pos + len(concept) + 12]
    window_measurements = extract_measurements(window)
    return window_measurements[0] if window_measurements else None


def _check_checklist_facts(
    chapters: dict[str, str],
    full_text: str,
    checklist: dict[str, Any],
    issues: list[ContentIssue],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    coverages: list[float] = []
    for fact in checklist.get("required_facts") or []:
        text = fact.get("text", "")
        fact_id = fact.get("fact_id", "")
        best_cov = coverage(text, full_text)
        best_field = "full_text"
        for name, section in chapters.items():
            cov = coverage(text, section)
            if cov > best_cov + 1e-9:
                best_cov, best_field = cov, name
        details.append(
            {"fact_id": fact_id, "text": text, "coverage": round(best_cov, 3), "field": best_field}
        )
        coverages.append(best_cov)
        if best_cov < 0.35:
            issues.append(
                ContentIssue(
                    "error",
                    "CHECKLIST_FACT_MISSING",
                    best_field,
                    f"必需事实 {fact_id} '{text}' 未保留（覆盖率 {best_cov:.2f}）",
                    {"fact_id": fact_id, "text": text, "coverage": round(best_cov, 3)},
                )
            )
        elif best_cov < 0.65:
            issues.append(
                ContentIssue(
                    "warning",
                    "CHECKLIST_FACT_PARTIAL",
                    best_field,
                    f"必需事实 {fact_id} '{text}' 仅部分保留（覆盖率 {best_cov:.2f}）",
                    {"fact_id": fact_id, "text": text, "coverage": round(best_cov, 3)},
                )
            )
    mean_cov = sum(coverages) / len(coverages) if coverages else None
    return {
        "required_fact_coverage": round(mean_cov, 3) if mean_cov is not None else None,
        "required_fact_detail": details,
    }


def _check_negation(
    full_text: str,
    review_text: str,
    sources: list[dict[str, Any]],
    checklist: dict[str, Any] | None,
    issues: list[ContentIssue],
) -> dict[str, Any]:
    auto_items = extract_missing_items(sources)
    checklist_items = checklist.get("missing_items") if checklist else None
    items: list[str] = []
    for item in list(checklist_items or []) + auto_items:
        if item not in items:
            items.append(item)

    statements: list[NegativeStatement] = extract_negative_statements(sources)
    preserved: list[str] = []
    rewritten: list[dict[str, str]] = []
    unmentioned: list[str] = []

    for item in items:
        in_review = any(coverage(item, review_item) >= 0.6 for review_item in _split_review_text(review_text))
        hit_normal = _claim_right_after(full_text, item, _NORMAL_CLAIM_WORDS)
        hit_preserved = next(
            (
                context
                for context in _iter_contexts(full_text, item, span=16)
                if any(word in context for word in _PRESERVED_WORDS)
            ),
            None,
        )
        if hit_normal:
            rewritten.append({"item": item, "context": hit_normal})
            issues.append(
                ContentIssue(
                    "error",
                    "NEGATION_REWRITTEN",
                    "root",
                    f"资料中未提供/未见的 '{item}' 在正文中被写成正常/阴性/排除",
                    {"item": item, "context": hit_normal},
                )
            )
        elif in_review or hit_preserved:
            preserved.append(item)
        else:
            unmentioned.append(item)

    # Report missing-item coverage as one aggregated signal instead of one
    # issue per item, so the score does not depend on how many items the
    # extractor happened to pull out of a single sentence.
    if unmentioned:
        shown = "、".join(unmentioned[:6])
        more = f" 等 {len(unmentioned)} 项" if len(unmentioned) > 6 else ""
        issues.append(
            ContentIssue(
                "warning",
                "MISSING_ITEM_UNMENTIONED",
                "root",
                f"资料明确缺失的 {len(unmentioned)} 项未在正文或待核事项中说明：{shown}{more}",
                {"items": list(unmentioned)},
            )
        )

    # "not provided" clauses must not be turned into "excluded" claims.
    for statement in statements:
        if not statement.item:
            continue
        if statement.kind == "negative_observation":
            contexts = list(_iter_contexts(full_text, statement.item, span=10))
            excluded = next(
                (
                    context
                    for context in contexts
                    if ("排除" in context) and ("不等于" not in context)
                ),
                None,
            )
            if excluded:
                issues.append(
                    ContentIssue(
                        "error",
                        "NEGATION_SCOPE_VIOLATION",
                        "root",
                        f"资料为阴性观察的 '{statement.item}'，正文出现排除性表述",
                        {"item": statement.item, "context": excluded, "source_id": statement.source_id},
                    )
                )

    review_items = _split_review_text(review_text)
    review_ok = sum(1 for item in items if any(coverage(item, review_item) >= 0.6 for review_item in review_items))
    if items and not review_items:
        if not any(word in full_text for word in _REVIEW_WORDS):
            issues.append(
                ContentIssue(
                    "error",
                    "REVIEW_ITEMS_EMPTY",
                    "review_items",
                    "资料存在明确缺失信息，但 review_items 为空且正文未提待核事项",
                    {"missing_items": items[:6]},
                )
            )

    metrics = {
        "missing_items": items,
        "preserved_items": preserved,
        "rewritten_items": [entry["item"] for entry in rewritten],
        "unmentioned_items": unmentioned,
        "review_coverage": round(review_ok / len(items), 3) if items else None,
    }
    return metrics


def _split_review_text(text: str) -> list[str]:
    return [part for part in split_clauses(text) if part]


def _check_forbidden_claims(
    full_text: str,
    checklist: dict[str, Any] | None,
    issues: list[ContentIssue],
    manual_review: list[dict[str, str]],
) -> int:
    if not checklist:
        return 0
    compiled = compile_forbidden_claims(checklist.get("forbidden_claims") or [])
    hits = 0
    for pattern in compiled.patterns:
        match = pattern.regex.search(full_text)
        if match:
            hits += 1
            issues.append(
                ContentIssue(
                    "error",
                    "CHECKLIST_FORBIDDEN_CLAIM",
                    "root",
                    f"命中禁止表述 '{pattern.claim}'：'{match.group(0)}'",
                    {"claim": pattern.claim, "context": full_text[max(0, match.start() - 12) : match.end() + 12]},
                )
            )
    for claim, reason in compiled.manual_review:
        manual_review.append({"check": "forbidden_claim", "claim": claim, "reason": reason})
    return hits


def _check_review_targets(
    checklist: dict[str, Any] | None,
    review_text: str,
    full_text: str,
    issues: list[ContentIssue],
    manual_review: list[dict[str, str]],
) -> dict[str, Any]:
    if not checklist:
        return {"review_targets": [], "covered": 0}
    targets = checklist.get("review_targets") or []
    covered: list[str] = []
    for target in targets:
        if coverage(target, review_text) >= 0.4:
            covered.append(target)
            continue
        if coverage(target, full_text) >= 0.6:
            manual_review.append(
                {"check": "review_target", "claim": target, "reason": "正文提及但未列入 review_items，建议人工确认"}
            )
            continue
        issues.append(
            ContentIssue(
                "warning",
                "CHECKLIST_REVIEW_TARGET_MISSING",
                "review_items",
                f"待核事项未覆盖 '{target}'",
                {"target": target},
            )
        )
    return {"review_targets": targets, "covered": len(covered)}


def _check_negation_scope_strings(
    full_text: str,
    checklist: dict[str, Any] | None,
    issues: list[ContentIssue],
    manual_review: list[dict[str, str]],
) -> None:
    """Best-effort handling of checklist negation_scope sentences.

    The format is prose ("现有记录未见明确出血表现，不等于排除所有出血原因").
    Only the "not equal to excluding X" shape can be checked automatically; the
    rest is surfaced as manual review rather than silently ignored.
    """
    if not checklist:
        return
    for scope in checklist.get("negation_scope") or []:
        if "不等于排除" not in scope:
            manual_review.append({"check": "negation_scope", "claim": scope, "reason": "需人工确认阴性范围"})
            continue
        head = scope.split("不等于排除", 1)[0]
        item = ""
        for marker in ("未见明确", "未见", "否认", "无"):
            pos = head.find(marker)
            if pos >= 0:
                item = head[pos + len(marker) :].strip("，,；;。 、")
                break
        if not item or len(item) > 12:
            manual_review.append({"check": "negation_scope", "claim": scope, "reason": "无法抽取阴性条目"})
            continue
        contexts = list(_iter_contexts(full_text, item, span=12))
        excluded = next((context for context in contexts if "排除" in context), None)
        if excluded:
            issues.append(
                ContentIssue(
                    "error",
                    "NEGATION_SCOPE_VIOLATION",
                    "root",
                    f"清单限定 '{item}' 的阴性范围，正文出现排除性表述",
                    {"scope": scope, "context": excluded},
                )
            )


def check_record(
    case: dict[str, Any],
    record: Any,
    checklist: dict[str, Any] | None = None,
) -> ContentCheckResult:
    """Run content-level checks for one generated record.

    ``checklist`` comes from :func:`src.checklist.load_checklist_for_case` (or
    None when the case has no checklist).  Never raises for content problems;
    structural misuse of the API still raises TypeError.
    """
    if not isinstance(case, dict):
        raise TypeError("case must be a dict")
    case_id = str(case.get("case_id", ""))
    result = ContentCheckResult(case_id=case_id)

    if checklist is not None:
        result.checklist_path = checklist.get("_path")
        if str(checklist.get("case_id", "")) and str(checklist["case_id"]) != case_id:
            result.issues.append(
                ContentIssue(
                    "error",
                    "CHECKLIST_CASE_MISMATCH",
                    "root",
                    f"清单 case_id '{checklist['case_id']}' 与输入病例 '{case_id}' 不一致",
                )
            )

    if not isinstance(record, dict):
        result.issues.append(ContentIssue("error", "RECORD_NOT_OBJECT", "root", "medical_record 不是 JSON 对象"))
        result.score = 0
        result.score_breakdown = {name: 0.0 for name in CATEGORY_WEIGHTS}
        return result

    sections = {name: _joined(record.get(name)) for name in SECTIONS}
    evidence = _evidence_entries(record)
    review_items = _review_items(record)
    review_text = "；".join(review_items)
    evidence_text = "".join(
        entry.get("statement", "") for entry in evidence if isinstance(entry.get("statement"), str)
    )
    full_text = "".join(sections.values()) + evidence_text
    source_by_id, sources = _source_index(case)
    source_text = "".join(source_by_id.values()) if source_by_id else "".join(
        str(src.get("content", "")) for src in sources
    )

    char_counts = _check_structure(sections, full_text, result.issues)
    _check_boundaries(sections, full_text, source_text, result.issues)
    number_metrics = _check_numbers(
        sections, evidence, source_by_id, source_text, checklist, result.issues
    )
    fact_metrics = (
        _check_checklist_facts(sections, full_text, checklist, result.issues)
        if checklist
        else {"required_fact_coverage": None, "required_fact_detail": []}
    )
    negation_metrics = _check_negation(full_text, review_text, sources, checklist, result.issues)
    _check_forbidden_claims(full_text, checklist, result.issues, result.manual_review)
    review_metrics = _check_review_targets(checklist, review_text, full_text, result.issues, result.manual_review)
    _check_negation_scope_strings(full_text, checklist, result.issues, result.manual_review)

    cited_sources = {
        sid
        for entry in evidence
        for sid in (entry.get("source_ids") or [])
        if isinstance(sid, str) and sid in source_by_id
    }
    evidence_metrics = {
        "count": len(evidence),
        "fact_count": sum(1 for entry in evidence if entry.get("kind") == "fact"),
        "inference_count": sum(1 for entry in evidence if entry.get("kind") == "inference"),
        "cited_sources": sorted(cited_sources),
        "source_coverage": round(len(cited_sources) / len(source_by_id), 3) if source_by_id else None,
    }

    result.metrics = {
        "char_counts": char_counts,
        "required_fact_coverage": fact_metrics["required_fact_coverage"],
        "required_fact_detail": fact_metrics["required_fact_detail"],
        "numeric_checks": number_metrics["numeric_checks"],
        "untraceable_numbers": number_metrics["untraceable_numbers"],
        "ungrounded_evidence_numbers": number_metrics["ungrounded_evidence_numbers"],
        "evidence": evidence_metrics,
        "negation": negation_metrics,
        "review": {"count": len(review_items), **review_metrics},
        "severity_counts": _severity_counts(result.issues),
    }

    result.score, result.score_breakdown = _score(result.issues)
    return result


def _severity_counts(issues: list[ContentIssue]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    return counts


def _score(issues: list[ContentIssue]) -> tuple[int, dict[str, float]]:
    penalties = {name: 0.0 for name in CATEGORY_WEIGHTS}
    for issue in issues:
        category = _CODE_CATEGORY.get(issue.code, "factual")
        penalties[category] += _SEVERITY_WEIGHT.get(issue.severity, 0.1)
    breakdown: dict[str, float] = {}
    total = 0.0
    for category, weight in CATEGORY_WEIGHTS.items():
        ratio = max(0.0, 1.0 - penalties[category])
        breakdown[category] = round(weight * ratio, 2)
        total += breakdown[category]
    return int(round(total)), breakdown


def summarize(results: Iterable[ContentCheckResult]) -> dict[str, Any]:
    """Aggregate per-case results for validation_report.json / run_log.json."""
    results = list(results)
    issues_by_code: dict[str, int] = {}
    issues_by_severity = {"error": 0, "warning": 0, "info": 0}
    for result in results:
        for issue in result.issues:
            issues_by_code[issue.code] = issues_by_code.get(issue.code, 0) + 1
            issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
    scores = [result.score for result in results]
    fact_coverages = [
        result.metrics["required_fact_coverage"]
        for result in results
        if result.metrics.get("required_fact_coverage") is not None
    ]
    return {
        "checker_version": CHECKER_VERSION,
        "checked_cases": len(results),
        "passed_cases": sum(1 for result in results if result.passed),
        "failed_cases": sum(1 for result in results if not result.passed),
        "issues_by_code": dict(sorted(issues_by_code.items(), key=lambda item: -item[1])),
        "issues_by_severity": issues_by_severity,
        "average_score": round(sum(scores) / len(scores), 2) if scores else None,
        "min_score": min(scores) if scores else None,
        "average_required_fact_coverage": round(sum(fact_coverages) / len(fact_coverages), 3)
        if fact_coverages
        else None,
        "case_scores": {result.case_id: result.score for result in results},
    }
