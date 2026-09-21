"""AWS Lambda entry point (API Gateway proxy integration).

Deploy with the handler set to ``iim_call_predictor.lambda_handler.lambda_handler``.
Two secrets must be configured as Lambda environment variables:

- ``SECRET_KEY``   — compared against the request's ``client-api-key`` header.
- ``AUTH_TOKEN``   — compared against the request's ``Authorization: Bearer <token>`` header.

Both checks are required (defense in depth); either being unconfigured on
the server, or missing/wrong on the request, fails the request closed.

Request body is JSON, shaped like a ``CandidateInput`` (see
``core.models.CandidateInput``), and is passed straight through to
``predictor.predict_all_colleges``.
"""

from __future__ import annotations

import hmac
import json
import os
from typing import Any, Dict, Optional

from .predictor import predict_all_colleges


class AuthenticationError(Exception):
    """Raised when a request fails authentication."""


def lambda_handler(event: Any, context: Any) -> Dict[str, Any]:
    try:
        _authenticate(event)
    except AuthenticationError as exc:
        return _json_response(401, {"success": False, "message": str(exc), "data": None})

    try:
        candidate_data = _parse_body(event)
    except ValueError as exc:
        return _json_response(400, {"success": False, "message": str(exc), "data": None})

    try:
        result = predict_all_colleges(candidate_data)
    except Exception:  # noqa: BLE001 - never leak internals to the caller
        return _json_response(500, {"success": False, "message": "Internal server error.", "data": None})

    return _json_response(200 if result.get("success") else 400, result)


def _authenticate(event: Any) -> None:
    """Fail-closed on two independent secrets: a static client API key and a
    bearer auth token, each compared in constant time against its own
    configured secret. An unconfigured secret must never mean open access,
    and both checks are required — a wrong/missing value in either one
    fails the whole request, without revealing which one failed.
    """
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        # Fail closed: an unconfigured secret must never mean open access.
        raise AuthenticationError("Server is not configured with SECRET_KEY.")

    raw_headers = event.get("headers") or {} if isinstance(event, dict) else {}
    headers = {str(k).lower(): v for k, v in raw_headers.items()}
    api_key = headers.get("client-api-key", "")

    if not isinstance(api_key, str) or not hmac.compare_digest(api_key, secret_key):
        raise AuthenticationError("Invalid or missing client-api-key.")


def _bearer_token(value: Optional[str]) -> str:
    """Extract the token from an ``Authorization: Bearer <token>`` header value."""
    if not isinstance(value, str):
        return ""
    prefix = "Bearer "
    return value[len(prefix):] if value.startswith(prefix) else ""


def _parse_body(event: Any) -> Dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("Malformed request: event must be an object.")

    body = event.get("body")
    if body is None:
        raise ValueError("Missing request body.")
    if isinstance(body, dict):
        return body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")

    try:
        parsed = json.loads(body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON body: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Request body must be a JSON object.")
    return parsed


def _json_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


# lambda_handler(
#     event={
#         "headers": {
#             "client-api-key": "test-secret",
#             "authorization": "Bearer test-token",
#         },
#         "body": {
#             "ug_pct": 75,
#             "category": "GENERAL",
#             "pwd": False,
#             "cat_overall_percentile": 98,
#             "cat_varc_percentile": 90,
#             "cat_dilr_percentile": 88,
#             "cat_qa_percentile": 85,
#             "cat_varc_raw_score": 30,
#             "cat_dilr_raw_score": 25,
#             "cat_qa_raw_score": 28,
#             "tenth_pct": 90,
#             "twelfth_pct": 86,
#             "gender": "Female",
#             "work_ex_months": 36,
#             "ug_discipline": "Engineering & Technology",
#         },
#     },
#     context=None,
# )