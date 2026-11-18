"""Unit and integration tests for src.model_client."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import requests

from src.model_client import (
    ModelClientError,
    ModelConnectionError,
    ModelHttpError,
    ModelResponseError,
    ModelTimeoutError,
    call_model,
    load_config,
)


class TestModelClientMocks(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "MODEL_NAME": "test-model",
            "ENDPOINT": "http://127.0.0.1:9999/v1/chat/completions",
        }
        self.case = {"case_id": "TEST001", "record_cutoff": "T0", "sources": []}
        self.prompt = "You are a test system prompt."

    @patch("src.model_client.requests.Session")
    def test_mock_401_http_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid API key"
        mock_session.post.return_value = mock_resp

        with self.assertRaises(ModelHttpError) as cm:
            call_model(self.case, self.prompt, config=self.mock_config)

        self.assertEqual(cm.exception.code, "MODEL_HTTP_ERROR")
        self.assertEqual(cm.exception.details.get("http_status"), 401)
        self.assertEqual(cm.exception.details.get("raw_response"), "Unauthorized: Invalid API key")
        self.assertIn("elapsed_seconds", cm.exception.details)

    @patch("src.model_client.requests.Session")
    def test_mock_non_json_response(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>502 Bad Gateway</body></html>"
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "<html>", 0)
        mock_session.post.return_value = mock_resp

        with self.assertRaises(ModelResponseError) as cm:
            call_model(self.case, self.prompt, config=self.mock_config)

        self.assertEqual(cm.exception.code, "MODEL_RESPONSE_ERROR")
        self.assertEqual(cm.exception.details.get("raw_response"), "<html><body>502 Bad Gateway</body></html>")
        self.assertIn("elapsed_seconds", cm.exception.details)

    @patch("src.model_client.requests.Session")
    def test_mock_content_missing(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"choices": [{"message": {}}]}'
        mock_resp.json.return_value = {"choices": [{"message": {}}]}
        mock_session.post.return_value = mock_resp

        with self.assertRaises(ModelResponseError) as cm:
            call_model(self.case, self.prompt, config=self.mock_config)

        self.assertEqual(cm.exception.code, "MODEL_RESPONSE_ERROR")
        self.assertIn("elapsed_seconds", cm.exception.details)
        self.assertIn("raw_response", cm.exception.details)

    @patch("src.model_client.requests.Session")
    def test_mock_empty_content(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"choices": [{"message": {"content": "   "}}]}'
        mock_resp.json.return_value = {"choices": [{"message": {"content": "   "}}]}
        mock_session.post.return_value = mock_resp

        with self.assertRaises(ModelResponseError) as cm:
            call_model(self.case, self.prompt, config=self.mock_config)

        self.assertEqual(cm.exception.code, "MODEL_RESPONSE_ERROR")
        self.assertIn("elapsed_seconds", cm.exception.details)

    @patch("src.model_client.requests.Session")
    def test_mock_timeout(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.side_effect = requests.exceptions.Timeout("Read timed out")

        with self.assertRaises(ModelTimeoutError) as cm:
            call_model(self.case, self.prompt, config=self.mock_config)

        self.assertEqual(cm.exception.code, "MODEL_TIMEOUT")
        self.assertIsNone(cm.exception.details.get("raw_response"))
        self.assertIn("elapsed_seconds", cm.exception.details)

    @patch("src.model_client.requests.Session")
    def test_mock_connection_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(ModelConnectionError) as cm:
            call_model(self.case, self.prompt, config=self.mock_config)

        self.assertEqual(cm.exception.code, "MODEL_CONNECTION_ERROR")
        self.assertIsNone(cm.exception.details.get("raw_response"))
        self.assertIn("elapsed_seconds", cm.exception.details)


class TestModelClientLiveCall(unittest.TestCase):
    def test_live_call_with_demo001(self):
        if os.environ.get("RUN_LIVE_TESTS") != "1":
            self.skipTest("Live model test skipped by default to avoid project pollution. Set RUN_LIVE_TESTS=1 to run.")

        try:
            config = load_config()
        except Exception as exc:
            self.skipTest(f"Skipping live call: config unavailable: {exc}")

        cases = json.loads((BASE_DIR / "examples" / "DEMO001.input.json").read_text(encoding="utf-8-sig"))
        case = cases[0]
        prompt = (BASE_DIR / "prompts" / "baseline.txt").read_text(encoding="utf-8-sig")

        res = call_model(case, prompt, config=config, timeout=(5, 60))
        self.assertIn("content", res)
        self.assertIsInstance(res["content"], str)
        self.assertTrue(len(res["content"].strip()) > 0)
        self.assertIn("finish_reason", res)
        self.assertIn("elapsed_seconds", res)
        self.assertGreater(res["elapsed_seconds"], 0)
        print("\n[LIVE CALL SUCCESS]")
        print("Model:", res.get("model_name"))
        print("Elapsed seconds:", res.get("elapsed_seconds"))
        print("Finish reason:", res.get("finish_reason"))
        print("Content preview:", res["content"][:150].replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
