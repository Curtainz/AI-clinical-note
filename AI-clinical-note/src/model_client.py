"""Model client encapsulating LLM inference calls via HTTP.

Complies with Task 1 requirements and note.txt diagnostics:
- Reads MODEL_NAME and ENDPOINT from project root .env (compatible with UTF-8 BOM).
- Uses requests.Session with trust_env=False.
- Posts directly to ENDPOINT without appending path or sending Authorization.
- stream=False, allow_redirects=False, timeout=(10, 180), concurrency=1, no retries.
- Classifies failures into ModelTimeoutError, ModelConnectionError, ModelHttpError, ModelResponseError.
- Captures full diagnostic details (elapsed_seconds, http_status, raw_response) on failures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import dotenv
import requests

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_GENERATION_PARAMS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 2048,
    "seed": 42,
}


class ModelClientError(Exception):
    """Base exception for model client errors with explicit error code and details."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}


class ModelTimeoutError(ModelClientError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("MODEL_TIMEOUT", message, details)


class ModelConnectionError(ModelClientError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("MODEL_CONNECTION_ERROR", message, details)


class ModelHttpError(ModelClientError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("MODEL_HTTP_ERROR", message, details)


class ModelResponseError(ModelClientError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__("MODEL_RESPONSE_ERROR", message, details)


def load_config(env_path: Path | str | None = None) -> dict[str, str]:
    """Read .env configuration, supporting UTF-8 and UTF-8 BOM.

    Only accepts MODEL_NAME and ENDPOINT.
    """
    target = Path(env_path) if env_path else BASE_DIR / ".env"
    if not target.is_file():
        raise FileNotFoundError(f"Configuration file not found: {target}")

    values = dotenv.dotenv_values(target, encoding="utf-8-sig")
    config: dict[str, str] = {}
    for key in ("MODEL_NAME", "ENDPOINT"):
        val = values.get(key)
        if not val or not str(val).strip():
            raise ValueError(f"Missing or empty required config key in .env: {key}")
        config[key] = str(val).strip()

    if not config["ENDPOINT"].startswith(("http://", "https://")):
        raise ValueError("ENDPOINT must be a valid HTTP(S) URL")

    return config


def call_model(
    case: dict[str, Any],
    system_prompt: str,
    config: dict[str, str] | None = None,
    generation_params: dict[str, Any] | None = None,
    timeout: tuple[int, int] = (10, 180),
    session: requests.Session | None = None,
    user_content: str | None = None,
) -> dict[str, Any]:
    """Invoke the language model for a single case.

    ``user_content`` overrides the default user message (the JSON case).  It is
    used by the optional self-review pass, which sends the case together with
    the draft record and the rule-checker findings.

    Returns an internal response dictionary:
      - content: parsed content string
      - finish_reason: finish reason reported by model (e.g. 'stop', 'length')
      - usage: token usage dict or None
      - elapsed_seconds: float round to 3 decimals
      - model_name: model identifier reported by service or config
      - http_status: HTTP status code
      - raw_response: raw response text

    Raises:
      ModelTimeoutError, ModelConnectionError, ModelHttpError, ModelResponseError.
      All exceptions carry details with elapsed_seconds, http_status, and raw_response when available.
    """
    cfg = config or load_config()
    params = generation_params if generation_params is not None else DEFAULT_GENERATION_PARAMS

    payload: dict[str, Any] = {
        "model": cfg["MODEL_NAME"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content if user_content is not None else json.dumps(case, ensure_ascii=False),
            },
        ],
        "stream": False,
        **params,
    }

    headers = {"Content-Type": "application/json"}

    created_session = False
    if session is None:
        session = requests.Session()
        session.trust_env = False
        created_session = True
    else:
        session.trust_env = False

    clock_start = time.perf_counter()
    try:
        response = session.post(
            cfg["ENDPOINT"],
            json=payload,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.exceptions.Timeout as exc:
        elapsed = round(time.perf_counter() - clock_start, 3)
        raise ModelTimeoutError(
            f"Model request timed out after {timeout} seconds: {exc}",
            details={"timeout": timeout, "elapsed_seconds": elapsed, "raw_response": None},
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        elapsed = round(time.perf_counter() - clock_start, 3)
        raise ModelConnectionError(
            f"Failed to connect to model endpoint {cfg['ENDPOINT']}: {exc}",
            details={"endpoint": cfg["ENDPOINT"], "elapsed_seconds": elapsed, "raw_response": None},
        ) from exc
    except requests.exceptions.RequestException as exc:
        elapsed = round(time.perf_counter() - clock_start, 3)
        raise ModelConnectionError(
            f"Request error while communicating with model: {exc}",
            details={"endpoint": cfg["ENDPOINT"], "elapsed_seconds": elapsed, "raw_response": None},
        ) from exc
    finally:
        if created_session:
            session.close()

    elapsed = round(time.perf_counter() - clock_start, 3)

    if response.status_code != 200:
        snippet = response.text[:200].replace("\n", " ")
        raise ModelHttpError(
            f"Model endpoint returned HTTP {response.status_code}: {snippet}",
            details={
                "http_status": response.status_code,
                "elapsed_seconds": elapsed,
                "raw_response": response.text,
            },
        )

    try:
        body = response.json()
    except Exception as exc:
        snippet = response.text[:200].replace("\n", " ")
        raise ModelResponseError(
            f"Model response was not valid JSON: {snippet}",
            details={
                "http_status": response.status_code,
                "elapsed_seconds": elapsed,
                "raw_response": response.text,
            },
        ) from exc

    if not isinstance(body, dict):
        raise ModelResponseError(
            "Model response JSON must be an object",
            details={
                "http_status": response.status_code,
                "elapsed_seconds": elapsed,
                "raw_response": response.text,
            },
        )

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelResponseError(
            "Model response choices must be a non-empty list",
            details={
                "http_status": response.status_code,
                "elapsed_seconds": elapsed,
                "raw_response": response.text,
                "body": body,
            },
        )

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ModelResponseError(
            "First choice in model response must be an object",
            details={
                "http_status": response.status_code,
                "elapsed_seconds": elapsed,
                "raw_response": response.text,
                "body": body,
            },
        )

    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelResponseError(
            "Choice message must be an object",
            details={
                "http_status": response.status_code,
                "elapsed_seconds": elapsed,
                "raw_response": response.text,
                "body": body,
            },
        )

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError(
            "Choice message.content must be a non-empty string",
            details={
                "http_status": response.status_code,
                "elapsed_seconds": elapsed,
                "raw_response": response.text,
                "body": body,
            },
        )

    finish_reason = choice.get("finish_reason")
    usage = body.get("usage")
    model_name = body.get("model") or cfg["MODEL_NAME"]

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "model_name": model_name,
        "http_status": response.status_code,
        "raw_response": response.text,
    }
