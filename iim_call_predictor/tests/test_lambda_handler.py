"""Tests for the AWS Lambda entry point (auth + request/response shape)."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from iim_call_predictor.lambda_handler import AuthenticationError, _authenticate, lambda_handler

SECRET_KEY = "s3cret-key"
AUTH_TOKEN = "t0ken-value"

VALID_CANDIDATE: Dict[str, Any] = {
    "ug_pct": 75,
    "category": "GENERAL",
    "cat_overall_percentile": 96,
    "cat_varc_percentile": 90,
    "cat_dilr_percentile": 88,
    "cat_qa_percentile": 85,
}


def _event(headers: Dict[str, str] | None = None, body: Any = None) -> Dict[str, Any]:
    return {"headers": headers or {}, "body": json.dumps(body) if body is not None else None}


def _valid_headers() -> Dict[str, str]:
    return {"client-api-key": SECRET_KEY, "Authorization": f"Bearer {AUTH_TOKEN}"}


@pytest.fixture(autouse=True)
def _configured_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", SECRET_KEY)
    monkeypatch.setenv("AUTH_TOKEN", AUTH_TOKEN)


# ---------------------------------------------------------------------------
# _authenticate: fail-closed behaviour
# ---------------------------------------------------------------------------


def test_authenticate_fails_closed_when_secret_key_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(AuthenticationError, match="SECRET_KEY"):
        _authenticate(_event(headers=_valid_headers()))


def test_authenticate_fails_closed_when_auth_token_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    with pytest.raises(AuthenticationError, match="AUTH_TOKEN"):
        _authenticate(_event(headers=_valid_headers()))


def test_authenticate_rejects_missing_api_key() -> None:
    headers = _valid_headers()
    del headers["client-api-key"]
    with pytest.raises(AuthenticationError, match="client-api-key"):
        _authenticate(_event(headers=headers))


def test_authenticate_rejects_wrong_api_key() -> None:
    headers = _valid_headers()
    headers["client-api-key"] = "wrong"
    with pytest.raises(AuthenticationError, match="client-api-key"):
        _authenticate(_event(headers=headers))


def test_authenticate_rejects_missing_auth_token() -> None:
    headers = _valid_headers()
    del headers["Authorization"]
    with pytest.raises(AuthenticationError, match="bearer token"):
        _authenticate(_event(headers=headers))


def test_authenticate_rejects_wrong_auth_token() -> None:
    headers = _valid_headers()
    headers["Authorization"] = "Bearer wrong"
    with pytest.raises(AuthenticationError, match="bearer token"):
        _authenticate(_event(headers=headers))


def test_authenticate_rejects_non_bearer_authorization_header() -> None:
    headers = _valid_headers()
    headers["Authorization"] = AUTH_TOKEN  # missing "Bearer " prefix
    with pytest.raises(AuthenticationError, match="bearer token"):
        _authenticate(_event(headers=headers))


def test_authenticate_header_lookup_is_case_insensitive() -> None:
    headers = {"CLIENT-API-KEY": SECRET_KEY, "authorization": f"Bearer {AUTH_TOKEN}"}
    _authenticate(_event(headers=headers))  # should not raise


def test_authenticate_passes_with_valid_headers() -> None:
    _authenticate(_event(headers=_valid_headers()))  # should not raise


# ---------------------------------------------------------------------------
# lambda_handler: end-to-end request/response shape
# ---------------------------------------------------------------------------


def test_handler_returns_401_when_unauthenticated() -> None:
    response = lambda_handler(_event(headers={}, body=VALID_CANDIDATE), None)
    assert response["statusCode"] == 401
    body = json.loads(response["body"])
    assert body["success"] is False
    assert body["data"] is None


def test_handler_returns_200_with_prediction_when_authenticated() -> None:
    response = lambda_handler(_event(headers=_valid_headers(), body=VALID_CANDIDATE), None)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["success"] is True
    assert "eligible_colleges" in body["data"]


def test_handler_returns_400_on_malformed_body() -> None:
    event = {"headers": _valid_headers(), "body": "{not valid json"}
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert body["success"] is False


def test_handler_returns_400_on_missing_body() -> None:
    event = {"headers": _valid_headers(), "body": None}
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400


def test_handler_returns_400_when_predictor_reports_failure() -> None:
    # Invalid candidate payload: predict_all_colleges itself returns success=False.
    event = {"headers": _valid_headers(), "body": json.dumps({"category": "NOT_REAL"})}
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert body["success"] is False
