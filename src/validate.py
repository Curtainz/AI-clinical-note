"""Validation module for input documents, individual cases, and medical records.

Implements Task 2 and Task 3 requirements:
- validate_input_document: checks top-level list, case object structure, and unique non-empty case_id.
- validate_case: checks record_cutoff and sources structure.
- validate_medical_record: enforces templates/medical_record.schema.json, verifies non-empty stripped strings,
  restricts additional fields, validates evidence structure, and ensures all referenced source_ids exist.
- parse_json_content: parses model response with safe single-layer Markdown code fence stripping.
"""

from __future__ import annotations

import json
from typing import Any


class TaskInputError(ValueError):
    """Raised when the input document fails task-level schema validation."""
    pass


def validate_input_document(value: Any) -> None:
    """Validate top-level document structure.

    Raises TaskInputError if:
      - value is not a list.
      - any item is not a dict.
      - case_id is missing, not a string, or empty/whitespace-only.
      - case_id is duplicated.
    """
    if not isinstance(value, list):
        raise TaskInputError("Top-level document must be a JSON array (list)")

    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TaskInputError(f"Item at index {index} must be a JSON object (case dict)")
        if "case_id" not in item:
            raise TaskInputError(f"Item at index {index} is missing required 'case_id'")
        cid = item["case_id"]
        if not isinstance(cid, str):
            raise TaskInputError(f"case_id at index {index} must be a string, got {type(cid).__name__}")
        if not cid.strip():
            raise TaskInputError(f"case_id at index {index} cannot be empty or whitespace-only")
        if cid in seen_ids:
            raise TaskInputError(f"Duplicate case_id found: '{cid}' at index {index}")
        seen_ids.add(cid)


def validate_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate basic input structure of a single case.

    Returns a list of issue dicts. If empty, the case is valid.
    """
    issues: list[dict[str, Any]] = []

    if not isinstance(case, dict):
        return [{
            "severity": "error",
            "code": "INVALID_CASE_INPUT",
            "field": "root",
            "message": "Case must be a JSON object",
        }]

    cutoff = case.get("record_cutoff")
    if not isinstance(cutoff, str) or not cutoff.strip():
        issues.append({
            "severity": "error",
            "code": "INVALID_CASE_INPUT",
            "field": "record_cutoff",
            "message": "record_cutoff must be a non-empty string",
        })

    sources = case.get("sources")
    if not isinstance(sources, list):
        issues.append({
            "severity": "error",
            "code": "INVALID_CASE_INPUT",
            "field": "sources",
            "message": "sources must be a list of source objects",
        })
    else:
        seen_sids: set[str] = set()
        for idx, src in enumerate(sources):
            if not isinstance(src, dict):
                issues.append({
                    "severity": "error",
                    "code": "INVALID_CASE_INPUT",
                    "field": f"sources[{idx}]",
                    "message": "Each source item must be an object",
                })
                continue
            sid = src.get("source_id")
            if not isinstance(sid, str) or not sid.strip():
                issues.append({
                    "severity": "error",
                    "code": "INVALID_CASE_INPUT",
                    "field": f"sources[{idx}].source_id",
                    "message": "source_id must be a non-empty string",
                })
            elif sid in seen_sids:
                issues.append({
                    "severity": "error",
                    "code": "INVALID_CASE_INPUT",
                    "field": f"sources[{idx}].source_id",
                    "message": f"Duplicate source_id within case: '{sid}'",
                })
            else:
                seen_sids.add(sid)

            content = src.get("content")
            if not isinstance(content, str) or not content.strip():
                issues.append({
                    "severity": "error",
                    "code": "INVALID_CASE_INPUT",
                    "field": f"sources[{idx}].content",
                    "message": f"Source content for '{sid}' must be a non-empty string",
                })

    return issues


def parse_json_content(content: str) -> tuple[dict[str, Any], bool]:
    """Parse JSON string from model response, handling single-layer outer markdown fences.

    Returns:
      (parsed_data, fence_stripped: bool)

    Raises:
      json.JSONDecodeError or ValueError on unparseable content.
    """
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data, False
        raise ValueError(f"Parsed JSON must be an object, got {type(data).__name__}")
    except (json.JSONDecodeError, ValueError) as primary_exc:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
                inner = "\n".join(lines[1:-1]).strip()
                data = json.loads(inner)
                if isinstance(data, dict):
                    return data, True
                raise ValueError(f"Parsed fenced JSON must be an object, got {type(data).__name__}") from primary_exc
        raise primary_exc


def validate_medical_record(record: Any, case: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate medical_record object against schema and source reference constraints.

    Returns a list of issue dicts. Any issue with severity='error' invalidates the record.
    """
    issues: list[dict[str, Any]] = []

    if not isinstance(record, dict):
        return [{
            "severity": "error",
            "code": "MODEL_SCHEMA_ERROR",
            "field": "root",
            "message": f"medical_record must be a JSON object, got {type(record).__name__}",
        }]

    required_fields = ["case_features", "diagnostic_discussion", "plan", "evidence", "review_items"]
    record_keys = set(record.keys())

    # Disallow extra fields
    extra_fields = record_keys - set(required_fields)
    if extra_fields:
        issues.append({
            "severity": "error",
            "code": "MODEL_SCHEMA_ERROR",
            "field": "additionalProperties",
            "message": f"medical_record contains prohibited extra fields: {sorted(extra_fields)}",
        })

    # Require all five fields
    missing_fields = set(required_fields) - record_keys
    if missing_fields:
        issues.append({
            "severity": "error",
            "code": "MODEL_SCHEMA_ERROR",
            "field": "required",
            "message": f"medical_record missing required fields: {sorted(missing_fields)}",
        })

    # Check the three text paragraphs
    for field_name in ("case_features", "diagnostic_discussion", "plan"):
        if field_name in record:
            val = record[field_name]
            if not isinstance(val, str) or not val.strip():
                issues.append({
                    "severity": "error",
                    "code": "MODEL_SCHEMA_ERROR",
                    "field": field_name,
                    "message": f"Field '{field_name}' must be a non-empty string",
                })

    # Check review_items
    if "review_items" in record:
        review_items = record["review_items"]
        if not isinstance(review_items, list):
            issues.append({
                "severity": "error",
                "code": "MODEL_SCHEMA_ERROR",
                "field": "review_items",
                "message": "review_items must be a list of strings",
            })
        else:
            for r_idx, item in enumerate(review_items):
                if not isinstance(item, str) or not item.strip():
                    issues.append({
                        "severity": "error",
                        "code": "MODEL_SCHEMA_ERROR",
                        "field": f"review_items[{r_idx}]",
                        "message": "review_items entries must be non-empty strings",
                    })

    # Extract valid source IDs from current case
    valid_source_ids: set[str] = set()
    raw_sources = case.get("sources", [])
    if isinstance(raw_sources, list):
        for s in raw_sources:
            if isinstance(s, dict) and isinstance(s.get("source_id"), str):
                valid_source_ids.add(s["source_id"])

    # Check evidence array
    if "evidence" in record:
        evidence = record["evidence"]
        if not isinstance(evidence, list):
            issues.append({
                "severity": "error",
                "code": "MODEL_SCHEMA_ERROR",
                "field": "evidence",
                "message": "evidence must be an array",
            })
        elif len(evidence) > 5:
            issues.append({
                "severity": "error",
                "code": "MODEL_SCHEMA_ERROR",
                "field": "evidence",
                "message": f"evidence array exceeds maximum allowed items of 5 (got {len(evidence)})",
            })
        else:
            if len(evidence) < 3:
                issues.append({
                    "severity": "warning",
                    "code": "FEW_EVIDENCE_ITEMS",
                    "field": "evidence",
                    "message": f"evidence item count ({len(evidence)}) is below standard recommendation of 3-5",
                })

            for e_idx, ev in enumerate(evidence):
                if not isinstance(ev, dict):
                    issues.append({
                        "severity": "error",
                        "code": "MODEL_SCHEMA_ERROR",
                        "field": f"evidence[{e_idx}]",
                        "message": "evidence entry must be a JSON object",
                    })
                    continue

                ev_extra = set(ev.keys()) - {"statement", "source_ids", "kind"}
                if ev_extra:
                    issues.append({
                        "severity": "error",
                        "code": "MODEL_SCHEMA_ERROR",
                        "field": f"evidence[{e_idx}]",
                        "message": f"evidence entry contains prohibited extra fields: {sorted(ev_extra)}",
                    })

                stmt = ev.get("statement")
                if not isinstance(stmt, str) or not stmt.strip():
                    issues.append({
                        "severity": "error",
                        "code": "MODEL_SCHEMA_ERROR",
                        "field": f"evidence[{e_idx}].statement",
                        "message": "evidence statement must be a non-empty string",
                    })

                kind = ev.get("kind")
                if kind not in ("fact", "inference"):
                    issues.append({
                        "severity": "error",
                        "code": "MODEL_SCHEMA_ERROR",
                        "field": f"evidence[{e_idx}].kind",
                        "message": f"evidence kind must be 'fact' or 'inference', got '{kind}'",
                    })

                source_ids = ev.get("source_ids")
                if not isinstance(source_ids, list) or not source_ids:
                    issues.append({
                        "severity": "error",
                        "code": "MODEL_SCHEMA_ERROR",
                        "field": f"evidence[{e_idx}].source_ids",
                        "message": "evidence source_ids must be a non-empty array of strings",
                    })
                else:
                    for s_item in source_ids:
                        if not isinstance(s_item, str) or not s_item.strip():
                            issues.append({
                                "severity": "error",
                                "code": "MODEL_SCHEMA_ERROR",
                                "field": f"evidence[{e_idx}].source_ids",
                                "message": "source_id element must be a non-empty string",
                            })
                        elif s_item not in valid_source_ids:
                            issues.append({
                                "severity": "error",
                                "code": "INVALID_SOURCE_REFERENCE",
                                "field": f"evidence[{e_idx}].source_ids",
                                "message": f"Referenced source_id '{s_item}' does not exist in case sources",
                                "source_ids": [s_item],
                            })

    return issues
