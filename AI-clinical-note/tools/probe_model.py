"""One-request local API probe. Not a competition entry point or clinical validator.

Uses only the Python standard library. Reads the two-key .env in the project root,
the first case from examples/DEMO001.input.json and prompts/baseline.txt.
Writes each attempt to a unique artifacts/probe directory; never writes output.json.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_config(path: Path) -> dict[str, str]:
    """Small parser for this kit's two unquoted key=value lines, not full dotenv."""
    config: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {number}: expected key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in {"MODEL_NAME", "ENDPOINT"} or key in config:
            raise ValueError(f"Unknown or duplicate .env key: {key}")
        if value.startswith(('"', "'")):
            raise ValueError("For this probe use unquoted .env values without inline comments")
        config[key] = value
    if not config.get("MODEL_NAME") or not config.get("ENDPOINT"):
        raise ValueError("MODEL_NAME and ENDPOINT must both be non-empty")
    if config["MODEL_NAME"] == "replace_with_model_identifier":
        raise ValueError("Replace MODEL_NAME using the actual service model identifier")
    if not config["ENDPOINT"].startswith(("http://", "https://")):
        raise ValueError("ENDPOINT must be a full HTTP(S) request URL")
    return config


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    try:
        config = read_config(BASE_DIR / ".env")
        cases = json.loads((BASE_DIR / "examples" / "DEMO001.input.json").read_text(encoding="utf-8-sig"))
        if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict):
            raise ValueError("Probe example must be a non-empty JSON array of case objects")
        case = cases[0]
        prompt = (BASE_DIR / "prompts" / "baseline.txt").read_text(encoding="utf-8-sig")
    except (OSError, ValueError) as exc:
        print(f"SETUP_ERROR: {exc}", file=sys.stderr)
        return 2

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = BASE_DIR / "artifacts" / "probe" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    payload = {
        "model": config["MODEL_NAME"],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
        ],
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 2048,
        "seed": 42,
    }
    write_json(folder / "request.json", payload)
    summary: dict[str, object] = {
        "run_id": run_id,
        "purpose": "development_api_probe_not_clinical_validation",
        "started_at_utc": started.isoformat(),
        "endpoint": config["ENDPOINT"],
        "model": config["MODEL_NAME"],
        "case_id": case.get("case_id"),
        "attempt": 1,
        "json_parse_ok": False,
        "schema_checked": False,
        "clinical_validated": False,
        "artifact_directory": str(folder),
    }
    clock_start = time.perf_counter()
    exit_code = 1
    try:
        request = urllib.request.Request(
            config["ENDPOINT"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Disable inherited proxy routing and HTTP redirects. No auth header or retries.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=180) as response:
            summary["http_status"] = response.status
            body = response.read().decode("utf-8")
        (folder / "provider_response.txt").write_text(body, encoding="utf-8")
        provider = json.loads(body)
        choice = provider["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider message.content must be a non-empty string")
        (folder / "content.txt").write_text(content, encoding="utf-8")
        summary["finish_reason"] = choice.get("finish_reason")
        summary["output_truncated"] = choice.get("finish_reason") == "length"
        summary["usage_reported_by_service"] = provider.get("usage")
        try:
            parsed = json.loads(content)
            summary["json_parse_ok"] = True
            summary["json_is_object"] = isinstance(parsed, dict)
            write_json(folder / "parsed_content.json", parsed)
        except json.JSONDecodeError as exc:
            summary["parse_error"] = str(exc)
        summary["probe_completed"] = True
        exit_code = 0
    except urllib.error.HTTPError as exc:
        summary["http_status"] = exc.code
        summary["error"] = f"HTTP_ERROR: {exc}"
        (folder / "provider_error.txt").write_text(exc.read().decode("utf-8", errors="replace"), encoding="utf-8")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["elapsed_seconds"] = round(time.perf_counter() - clock_start, 3)
        summary["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(folder / "probe_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
