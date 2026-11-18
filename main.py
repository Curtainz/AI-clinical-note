"""Batch processing entry point for clinical-note-prototype.

Complies with Task 2, execution manual D2, and note.txt diagnostics:
- Reads config (.env), active prompt (prompts/active.txt), and input (input.json).
- Performs task-level input document validation (exit code 2 on protocol/config failure).
- Records completed=False and diagnostic details immediately upon setup or task-level failure.
- Accurately tracks actual model call attempts (attempt=0 when input validation fails before call).
- Captures full diagnostic evidence (HTTP status, elapsed_seconds, raw_response) on model errors.
- Atomically writes output.json via output.json.tmp.
- Saves run_log.json and validation_report.json both in project root (for D5 archive) and artifacts/runs/{run_id}/.
- Exit code 0 on complete run; 1 on runtime write/IO failure; 2 on setup/input validation error.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import requests

from src.model_client import (
    DEFAULT_GENERATION_PARAMS,
    ModelClientError,
    call_model,
    load_config,
)
from src.validate import (
    TaskInputError,
    parse_json_content,
    validate_case,
    validate_input_document,
    validate_medical_record,
)


def compute_sha256(content_bytes: bytes) -> str:
    """Compute hex SHA256 digest of given bytes."""
    return hashlib.sha256(content_bytes).hexdigest()


def main(base_dir: Path | str | None = None) -> int:
    root = Path(base_dir).resolve() if base_dir else BASE_DIR
    started_dt = datetime.now(timezone.utc)
    run_id = started_dt.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]

    run_folder = root / "artifacts" / "runs" / run_id
    raw_folder = root / "artifacts" / "raw" / run_id

    def save_run_log(log_data: dict[str, Any]) -> None:
        text = json.dumps(log_data, ensure_ascii=False, indent=2)
        try:
            run_folder.mkdir(parents=True, exist_ok=True)
            (run_folder / "run_log.json").write_text(text, encoding="utf-8")
            (root / "artifacts" / "run_log.json").write_text(text, encoding="utf-8")
            (root / "run_log.json").write_text(text, encoding="utf-8")
        except Exception:
            pass

    def save_validation_report(report_data: dict[str, Any]) -> None:
        text = json.dumps(report_data, ensure_ascii=False, indent=2)
        try:
            run_folder.mkdir(parents=True, exist_ok=True)
            (run_folder / "validation_report.json").write_text(text, encoding="utf-8")
            (root / "artifacts" / "validation_report.json").write_text(text, encoding="utf-8")
            (root / "validation_report.json").write_text(text, encoding="utf-8")
        except Exception:
            pass

    def record_fatal_failure(code: str, message: str, exit_code: int = 2, config_info: dict[str, str] | None = None) -> int:
        ended_dt = datetime.now(timezone.utc)
        fail_log: dict[str, Any] = {
            "run_id": run_id,
            "started_at_utc": started_dt.isoformat(),
            "ended_at_utc": ended_dt.isoformat(),
            "completed": False,
            "model": config_info.get("MODEL_NAME") if config_info else None,
            "endpoint": config_info.get("ENDPOINT") if config_info else None,
            "parameters": DEFAULT_GENERATION_PARAMS,
            "prompt_sha256": None,
            "input_sha256": None,
            "output_sha256": None,
            "total_cases": 0,
            "successful_cases": 0,
            "failed_cases": 0,
            "total_model_calls": 0,
            "attempt_metric": "actual_http_calls_sent_to_model",
            "error": {"code": code, "message": message},
            "case_runs": [],
        }
        save_run_log(fail_log)
        save_validation_report({
            "run_id": run_id,
            "cases": [],
            "fatal_error": {"code": code, "message": message},
        })
        print(f"{code}: {message}", file=sys.stderr)
        return exit_code

    # --- Step 1: Configuration & Input Loading (Exit code 2 on setup/protocol errors) ---
    try:
        config = load_config(root / ".env")
    except Exception as exc:
        return record_fatal_failure("CONFIGURATION_ERROR", str(exc), exit_code=2)

    prompt_path = root / "prompts" / "active.txt"
    if not prompt_path.is_file():
        return record_fatal_failure("SETUP_ERROR", f"active prompt file not found at {prompt_path}", exit_code=2, config_info=config)

    try:
        prompt_bytes = prompt_path.read_bytes()
        prompt_text = prompt_bytes.decode("utf-8-sig")
        if not prompt_text.strip():
            return record_fatal_failure("SETUP_ERROR", "prompts/active.txt is empty", exit_code=2, config_info=config)
    except Exception as exc:
        return record_fatal_failure("SETUP_ERROR", f"failed reading prompts/active.txt: {exc}", exit_code=2, config_info=config)

    input_path = root / "input.json"
    if not input_path.is_file():
        return record_fatal_failure("SETUP_ERROR", f"input.json not found at {input_path}", exit_code=2, config_info=config)

    try:
        input_bytes = input_path.read_bytes()
        cases = json.loads(input_bytes.decode("utf-8-sig"))
    except Exception as exc:
        return record_fatal_failure("SETUP_ERROR", f"failed parsing input.json as JSON: {exc}", exit_code=2, config_info=config)

    try:
        validate_input_document(cases)
    except TaskInputError as exc:
        return record_fatal_failure("TASK_INPUT_ERROR", str(exc), exit_code=2, config_info=config)

    prompt_sha256 = compute_sha256(prompt_bytes)
    input_sha256 = compute_sha256(input_bytes)

    # Setup artifact directories
    try:
        run_folder.mkdir(parents=True, exist_ok=True)
        raw_folder.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return record_fatal_failure("RUNTIME_ERROR", f"failed creating artifact directories: {exc}", exit_code=1, config_info=config)

    # --- Step 2: Batch Processing ---
    results: list[dict[str, Any]] = []
    case_runs: list[dict[str, Any]] = []
    validation_cases_report: list[dict[str, Any]] = []

    session = requests.Session()
    session.trust_env = False

    try:
        for idx, case in enumerate(cases):
            cid = case["case_id"]
            case_issues = validate_case(case)
            case_has_input_error = any(i.get("severity") == "error" for i in case_issues)

            case_run_log: dict[str, Any] = {
                "index": idx,
                "case_id": cid,
                "attempt": 0,
                "model_called": False,
                "status": "failed",
                "error": None,
                "elapsed_seconds": None,
                "finish_reason": None,
                "fence_stripped": False,
                "raw_response_path": None,
            }

            if case_has_input_error:
                # No model invocation: attempt stays 0, model_called stays False
                first_err = next(i for i in case_issues if i.get("severity") == "error")
                err_dict = {"code": first_err["code"], "message": first_err["message"]}
                results.append({
                    "case_id": cid,
                    "status": "failed",
                    "medical_record": None,
                    "error": err_dict,
                })
                case_run_log["error"] = err_dict
                case_runs.append(case_run_log)
                validation_cases_report.append({
                    "case_id": cid,
                    "protocol_valid": False,
                    "issues": case_issues,
                })
                continue

            # Case input is valid -> initiate actual model call
            case_run_log["attempt"] = 1
            case_run_log["model_called"] = True
            raw_file_rel = f"artifacts/raw/{run_id}/{idx}.txt"
            raw_file_path = raw_folder / f"{idx}.txt"

            try:
                model_resp = call_model(
                    case=case,
                    system_prompt=prompt_text,
                    config=config,
                    session=session,
                )
                case_run_log["elapsed_seconds"] = model_resp["elapsed_seconds"]
                case_run_log["finish_reason"] = model_resp["finish_reason"]
                case_run_log["raw_response_path"] = raw_file_rel

                # Persist raw response
                raw_file_path.write_text(model_resp.get("raw_response", ""), encoding="utf-8")

                if model_resp.get("finish_reason") == "length":
                    err_dict = {
                        "code": "OUTPUT_TRUNCATED",
                        "message": "Output truncated by model (finish_reason=length)",
                    }
                    results.append({
                        "case_id": cid,
                        "status": "failed",
                        "medical_record": None,
                        "error": err_dict,
                    })
                    case_run_log["error"] = err_dict
                    validation_cases_report.append({
                        "case_id": cid,
                        "protocol_valid": False,
                        "issues": [{
                            "severity": "error",
                            "code": "OUTPUT_TRUNCATED",
                            "field": "root",
                            "message": err_dict["message"],
                        }],
                    })
                    case_runs.append(case_run_log)
                    continue

                # Parse JSON content
                try:
                    record, fence_stripped = parse_json_content(model_resp["content"])
                    case_run_log["fence_stripped"] = fence_stripped
                except Exception as exc:
                    err_dict = {
                        "code": "JSON_PARSE_ERROR",
                        "message": f"Failed to parse model content as JSON: {exc}",
                    }
                    results.append({
                        "case_id": cid,
                        "status": "failed",
                        "medical_record": None,
                        "error": err_dict,
                    })
                    case_run_log["error"] = err_dict
                    validation_cases_report.append({
                        "case_id": cid,
                        "protocol_valid": False,
                        "issues": [{
                            "severity": "error",
                            "code": "JSON_PARSE_ERROR",
                            "field": "root",
                            "message": err_dict["message"],
                        }],
                    })
                    case_runs.append(case_run_log)
                    continue

                # Validate medical_record schema and source references
                record_issues = validate_medical_record(record, case)
                has_record_error = any(i.get("severity") == "error" for i in record_issues)

                if has_record_error:
                    first_err = next(i for i in record_issues if i.get("severity") == "error")
                    err_dict = {"code": first_err["code"], "message": first_err["message"]}
                    results.append({
                        "case_id": cid,
                        "status": "failed",
                        "medical_record": None,
                        "error": err_dict,
                    })
                    case_run_log["error"] = err_dict
                    case_runs.append(case_run_log)
                    validation_cases_report.append({
                        "case_id": cid,
                        "protocol_valid": False,
                        "issues": record_issues,
                    })
                else:
                    results.append({
                        "case_id": cid,
                        "status": "success",
                        "medical_record": record,
                        "error": None,
                    })
                    case_run_log["status"] = "success"
                    case_runs.append(case_run_log)
                    validation_cases_report.append({
                        "case_id": cid,
                        "protocol_valid": True,
                        "issues": record_issues,
                    })

            except ModelClientError as exc:
                err_dict = {"code": exc.code, "message": exc.message}
                case_run_log["elapsed_seconds"] = exc.details.get("elapsed_seconds")
                raw_text = exc.details.get("raw_response")
                if raw_text is not None:
                    raw_file_path.write_text(raw_text, encoding="utf-8")
                    case_run_log["raw_response_path"] = raw_file_rel
                else:
                    case_run_log["raw_response_path"] = None

                diag_details = {k: v for k, v in exc.details.items() if k != "raw_response"}
                if diag_details:
                    case_run_log["details"] = diag_details

                results.append({
                    "case_id": cid,
                    "status": "failed",
                    "medical_record": None,
                    "error": err_dict,
                })
                case_run_log["error"] = err_dict
                case_runs.append(case_run_log)
                validation_cases_report.append({
                    "case_id": cid,
                    "protocol_valid": False,
                    "issues": [{
                        "severity": "error",
                        "code": exc.code,
                        "field": "root",
                        "message": exc.message,
                    }],
                })
            except Exception as exc:
                err_dict = {"code": "MODEL_CALL_ERROR", "message": str(exc)}
                results.append({
                    "case_id": cid,
                    "status": "failed",
                    "medical_record": None,
                    "error": err_dict,
                })
                case_run_log["error"] = err_dict
                case_runs.append(case_run_log)
                validation_cases_report.append({
                    "case_id": cid,
                    "protocol_valid": False,
                    "issues": [{
                        "severity": "error",
                        "code": "MODEL_CALL_ERROR",
                        "field": "root",
                        "message": str(exc),
                    }],
                })
    finally:
        session.close()

    # --- Step 3: Atomic Output Writing ---
    output_path = root / "output.json"
    tmp_output_path = root / "output.json.tmp"
    try:
        output_json_text = json.dumps(results, ensure_ascii=False, indent=2)
        tmp_output_path.write_text(output_json_text, encoding="utf-8")
        os.replace(tmp_output_path, output_path)
    except Exception as exc:
        record_fatal_failure("RUNTIME_ERROR", f"failed atomically saving output.json: {exc}", exit_code=1, config_info=config)
        return 1

    ended_dt = datetime.now(timezone.utc)
    output_sha256 = compute_sha256(output_path.read_bytes())

    # --- Step 4: Write Run Log & Validation Report ---
    run_log: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": started_dt.isoformat(),
        "ended_at_utc": ended_dt.isoformat(),
        "completed": True,
        "model": config["MODEL_NAME"],
        "endpoint": config["ENDPOINT"],
        "parameters": DEFAULT_GENERATION_PARAMS,
        "prompt_sha256": prompt_sha256,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "total_cases": len(cases),
        "successful_cases": sum(1 for r in results if r["status"] == "success"),
        "failed_cases": sum(1 for r in results if r["status"] == "failed"),
        "total_model_calls": sum(1 for c in case_runs if c.get("model_called")),
        "attempt_metric": "actual_http_calls_sent_to_model",
        "case_runs": case_runs,
    }

    validation_report: dict[str, Any] = {
        "run_id": run_id,
        "cases": validation_cases_report,
    }

    save_run_log(run_log)
    save_validation_report(validation_report)

    print(f"SUCCESS: Processed {len(cases)} cases (Successful: {run_log['successful_cases']}, Failed: {run_log['failed_cases']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
