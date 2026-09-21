"""Tests for the cross-college aggregator (predict_all_colleges)."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from iim_call_predictor.predictor import predict_all_colleges

FULL_CANDIDATE: Dict[str, Any] = {
    "tenth_pct": 92,
    "twelfth_pct": 88,
    "ug_pct": 75,
    "work_ex_months": 24,
    "gender": "Female",
    "category": "GENERAL",
    "pwd": False,
    "cat_overall_percentile": 99.5,
    "cat_varc_percentile": 96,
    "cat_dilr_percentile": 96,
    "cat_qa_percentile": 96,
    "ug_discipline": "Engineering & Technology",
    "cat_varc_raw_score": 30,
    "cat_dilr_raw_score": 25,
    "cat_qa_raw_score": 28,
}

IIMM_ONLY_CANDIDATE: Dict[str, Any] = {
    "ug_pct": 72,
    "category": "NC_OBC",
    "pwd": False,
    "cat_overall_percentile": 91,
    "cat_varc_percentile": 88,
    "cat_dilr_percentile": 85,
    "cat_qa_percentile": 82,
}


def test_response_shape_and_success() -> None:
    response = predict_all_colleges(FULL_CANDIDATE)
    assert response["success"] is True
    assert response["message"] == "Prediction generated successfully."
    assert "eligible_colleges" in response["data"]


def test_strong_candidate_appears_for_all_colleges() -> None:
    response = predict_all_colleges(FULL_CANDIDATE)
    by_name = {c["college_name"]: c for c in response["data"]["eligible_colleges"]}

    assert set(by_name) == {"IIM Ahmedabad", "IIM Mumbai", "IIM Calcutta", "IIM Bangalore"}
    for entry in by_name.values():
        assert set(entry) == {"college_name", "city", "predicted_call_probability", "recommendation"}
    # IIMA and IIMM have real, non-zero call thresholds -> a strong candidate reads High/Safe.
    assert by_name["IIM Ahmedabad"]["predicted_call_probability"] == "High"
    assert by_name["IIM Ahmedabad"]["recommendation"] == "Safe"
    assert by_name["IIM Mumbai"]["predicted_call_probability"] == "High"
    assert by_name["IIM Mumbai"]["recommendation"] == "Safe"
    assert by_name["IIM Calcutta"]["predicted_call_probability"] == "High"
    assert by_name["IIM Calcutta"]["recommendation"] == "Safe"
    # IIMB's call cutoff still defaults to 0 (no real figures supplied yet), so margin
    # can't be assessed -> Medium/Moderate, even though the call itself is True.
    assert by_name["IIM Bangalore"]["predicted_call_probability"] == "Medium"
    assert by_name["IIM Bangalore"]["recommendation"] == "Moderate"

    assert by_name["IIM Ahmedabad"]["city"] == "Ahmedabad"
    assert by_name["IIM Mumbai"]["city"] == "Mumbai"
    assert by_name["IIM Calcutta"]["city"] == "Kolkata"
    assert by_name["IIM Bangalore"]["city"] == "Bangalore"


def test_college_missing_required_fields_is_skipped_with_warning() -> None:
    response = predict_all_colleges(IIMM_ONLY_CANDIDATE)
    names = {c["college_name"] for c in response["data"]["eligible_colleges"]}

    assert "IIM Ahmedabad" not in names
    assert "IIM Calcutta" not in names
    assert "IIM Bangalore" not in names
    assert any("iima" in w for w in response.get("warnings", []))
    assert any("iimc" in w for w in response.get("warnings", []))
    assert any("iimb" in w for w in response.get("warnings", []))
    assert "IIM Mumbai" in names


def test_below_call_threshold_is_low_risky() -> None:
    response = predict_all_colleges(IIMM_ONLY_CANDIDATE)  # NC_OBC 91 < 92.5 midpoint
    iimm_entry = next(c for c in response["data"]["eligible_colleges"] if c["college_name"] == "IIM Mumbai")

    assert iimm_entry["predicted_call_probability"] == "Low"
    assert iimm_entry["recommendation"] == "Risky"


def test_ineligible_candidate_is_excluded_entirely() -> None:
    ineligible = dict(IIMM_ONLY_CANDIDATE, ug_pct=30)  # below even the relaxed 45% minimum
    response = predict_all_colleges(ineligible)
    assert response["data"]["eligible_colleges"] == []


def test_invalid_candidate_input_returns_failure_response() -> None:
    response = predict_all_colleges({"category": "NOT_A_REAL_CATEGORY"})
    assert response["success"] is False
    assert response["data"] is None
    assert "message" in response
