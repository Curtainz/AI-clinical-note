"""Self-review ("self-audit") pass for generated clinical-note records.

The reviewer is the same deployed model: it receives the case sources, its own
draft, and the rule-checker findings, and returns a structured review.  A
revised draft is only applied when it

* still passes schema/source validation,
* does not lower the content-checker score, and
* introduces no new error-level content issue code.

The pass is opt-in (``main.py --self-review``) so the default runtime keeps the
one-call-per-case contract.  ``requests`` is imported lazily inside
:func:`review_and_revise` to keep the pure helpers testable without the runtime
dependencies installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .content_check import ContentCheckResult, check_record
from .validate import parse_json_content, validate_medical_record

REVIEW_VERDICTS = ("pass", "revise")


@dataclass
class SelfReviewOutcome:
    """Audit record for one self-review pass."""

    called: bool = False
    verdict: str | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    revised_record: dict[str, Any] | None = None
    accepted: bool = False
    reason: str | None = None
    score_before: int = 0
    score_after: int | None = None
    elapsed_seconds: float | None = None
    finish_reason: str | None = None
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "called": self.called,
            "verdict": self.verdict,
            "issues": self.issues,
            "notes": self.notes,
            "revised": self.revised_record is not None,
            "accepted": self.accepted,
            "reason": self.reason,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "elapsed_seconds": self.elapsed_seconds,
            "finish_reason": self.finish_reason,
        }


def build_review_user_content(
    case: dict[str, Any],
    record: dict[str, Any],
    content_result: ContentCheckResult | None = None,
    checklist: dict[str, Any] | None = None,
) -> str:
    """Build the user message for the self-review call (case + draft + findings)."""
    blocks = [
        "【病例资料：只能依据以下 sources；外部医学常识仅用于分析，不得当作本例检查结果】",
        json.dumps(case, ensure_ascii=False, indent=2),
        "【待审查的初稿 medical_record】",
        json.dumps(record, ensure_ascii=False, indent=2),
    ]

    findings: list[str] = []
    if content_result is not None:
        for issue in content_result.issues[:20]:
            findings.append(f"- [{issue.severity}] {issue.code}（{issue.field}）：{issue.message}")
    if checklist:
        facts = [str(item) for item in checklist.get("required_facts") or [] if str(item).strip()]
        if facts:
            findings.append("- 人工清单要求的必含事实：" + "；".join(facts))
        targets = [str(item) for item in checklist.get("review_targets") or [] if str(item).strip()]
        if targets:
            findings.append("- 人工清单要求的待核重点：" + "；".join(targets))

    if findings:
        blocks += ["【规则校验与人工清单已发现的问题】", "\n".join(findings)]

    blocks += ["", "请按系统提示完成审查，只返回一个 JSON 对象，不要输出解释或代码围栏。"]
    return "\n".join(blocks)


def parse_review_response(text: str) -> dict[str, Any]:
    """Parse the reviewer response into a normalised dict.

    Raises:
        ValueError: when the payload is not a review object with a valid
            ``verdict`` (and, for ``revise``, a ``revised_record`` object).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("review response is empty")

    data, _ = parse_json_content(text)
    if not isinstance(data, dict):
        raise ValueError(f"review JSON must be an object, got {type(data).__name__}")

    verdict = data.get("verdict")
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(f"review verdict must be 'pass' or 'revise', got {verdict!r}")

    raw_issues = data.get("issues") or []
    if not isinstance(raw_issues, list):
        raise ValueError("review issues must be a list")
    issues: list[dict[str, Any]] = []
    for index, item in enumerate(raw_issues[:20]):
        if not isinstance(item, dict):
            raise ValueError(f"review issues[{index}] must be an object")
        severity = str(item.get("severity", "warning")).strip().lower()
        if severity not in ("error", "warning", "info"):
            severity = "warning"
        issues.append(
            {
                "code": str(item.get("code") or "REVIEW_ISSUE"),
                "severity": severity,
                "field": str(item.get("field") or "root"),
                "message": str(item.get("message") or ""),
            }
        )

    raw_notes = data.get("notes") or []
    if not isinstance(raw_notes, list):
        raise ValueError("review notes must be a list")
    notes = [str(note) for note in raw_notes if str(note).strip()]

    revised = data.get("revised_record")
    if revised is not None and not isinstance(revised, dict):
        raise ValueError("review revised_record must be an object or null")
    if verdict == "revise" and revised is None:
        raise ValueError("review verdict is 'revise' but revised_record is null")
    if verdict == "pass":
        revised = None

    return {"verdict": verdict, "issues": issues, "notes": notes, "revised_record": revised}


def evaluate_revision(
    case: dict[str, Any],
    original: dict[str, Any],
    revised: dict[str, Any],
    checklist: dict[str, Any] | None = None,
) -> tuple[bool, str | None, int | None, ContentCheckResult | None]:
    """Decide whether a revised draft may replace the original.

    Returns ``(accepted, reason, revised_score, revised_result)``.
    """
    if not isinstance(revised, dict):
        return False, "revised_record is not a JSON object", None, None

    schema_issues = validate_medical_record(revised, case)
    schema_errors = [issue for issue in schema_issues if issue.get("severity") == "error"]
    if schema_errors:
        return (
            False,
            f"revised record failed schema/source validation: {schema_errors[0].get('code')}",
            None,
            None,
        )

    before = check_record(case, original, checklist)
    after = check_record(case, revised, checklist)

    if after.score < before.score:
        return False, f"revised content score {after.score} is lower than original {before.score}", after.score, after

    before_error_codes = {issue.code for issue in before.issues if issue.severity == "error"}
    new_error_codes = sorted({issue.code for issue in after.issues if issue.severity == "error"} - before_error_codes)
    if new_error_codes:
        return (
            False,
            "revised record introduced new error issue(s): " + ", ".join(new_error_codes),
            after.score,
            after,
        )

    return True, None, after.score, after


def review_and_revise(
    case: dict[str, Any],
    record: dict[str, Any],
    system_prompt: str,
    config: dict[str, str],
    generation_params: dict[str, Any] | None = None,
    timeout: tuple[int, int] = (10, 180),
    session: Any = None,
    checklist: dict[str, Any] | None = None,
    content_result: ContentCheckResult | None = None,
) -> SelfReviewOutcome:
    """Run one self-review model call and decide whether to apply its revision.

    Never raises for model/content problems: every failure is reported through
    the returned :class:`SelfReviewOutcome` so the caller keeps the original
    draft and the run stays alive.
    """
    from .model_client import ModelClientError, call_model

    outcome = SelfReviewOutcome()
    before = content_result if content_result is not None else check_record(case, record, checklist)
    outcome.score_before = before.score

    user_content = build_review_user_content(case, record, before, checklist)
    try:
        response = call_model(
            case=case,
            system_prompt=system_prompt,
            config=config,
            generation_params=generation_params,
            timeout=timeout,
            session=session,
            user_content=user_content,
        )
    except ModelClientError as exc:
        outcome.reason = f"{exc.code}: {exc.message}"
        outcome.elapsed_seconds = exc.details.get("elapsed_seconds")
        outcome.raw_response = exc.details.get("raw_response")
        return outcome

    outcome.called = True
    outcome.elapsed_seconds = response.get("elapsed_seconds")
    outcome.finish_reason = response.get("finish_reason")
    outcome.raw_response = response.get("raw_response")

    if response.get("finish_reason") == "length":
        outcome.reason = "self-review output truncated (finish_reason=length)"
        return outcome

    try:
        parsed = parse_review_response(response.get("content") or "")
    except ValueError as exc:
        outcome.reason = f"unparseable self-review JSON: {exc}"
        return outcome

    outcome.verdict = parsed["verdict"]
    outcome.issues = parsed["issues"]
    outcome.notes = parsed["notes"]
    outcome.revised_record = parsed["revised_record"]

    if outcome.revised_record is None:
        outcome.reason = "reviewer requested no revision"
        return outcome

    accepted, reason, score_after, _after_result = evaluate_revision(
        case=case,
        original=record,
        revised=outcome.revised_record,
        checklist=checklist,
    )
    outcome.accepted = accepted
    outcome.reason = reason
    outcome.score_after = score_after
    return outcome
