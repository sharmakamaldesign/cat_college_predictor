"""Tests for the IIM Mumbai plug-in (eligibility + call prediction only).

IIM Mumbai's Academic Performance & Work Experience (APWE) score is used
only for Stage III final selection, after the PI, so it is out of scope
here and is not tested — only eligibility, Stage I minimum CAT-2025 cutoffs
(Table 1 of the official policy), and the resulting call prediction.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from iim_call_predictor.colleges.iimm.model import IIMMModel
from iim_call_predictor.core.models import CandidateInput
from iim_call_predictor.core.registry import get_college, list_colleges


def base_candidate_kwargs(**overrides: Any) -> Dict[str, Any]:
    """A minimal, valid IIMM candidate payload (comfortably above the GENERAL/OPEN
    Stage I minimums); override fields per-test."""
    data: Dict[str, Any] = {
        "ug_pct": 75,
        "category": "GENERAL",
        "pwd": False,
        "cat_overall_percentile": 95,
        "cat_varc_percentile": 90,
        "cat_dilr_percentile": 90,
        "cat_qa_percentile": 90,
    }
    data.update(overrides)
    return data


@pytest.fixture()
def model() -> IIMMModel:
    return IIMMModel()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_iimm_is_registered() -> None:
    assert "iimm" in list_colleges()
    assert isinstance(get_college("iimm"), IIMMModel)


# ---------------------------------------------------------------------------
# Eligibility: UG% >= 50, relaxed to 45 for SC/ST/PwD
# ---------------------------------------------------------------------------


def test_eligibility_default_threshold(model: IIMMModel) -> None:
    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=50)))
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=49.99)))
    assert eligible is False, reasons


def test_eligibility_relaxed_threshold_for_sc_st_and_pwd(model: IIMMModel) -> None:
    eligible, reasons = model.check_basic_eligibility(
        CandidateInput(**base_candidate_kwargs(category="SC", ug_pct=45))
    )
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(
        CandidateInput(**base_candidate_kwargs(category="GENERAL", pwd=True, ug_pct=45))
    )
    assert eligible is True, reasons


# ---------------------------------------------------------------------------
# Stage I: Table 1 minimum CAT-2025 percentile cutoffs, exact boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,pwd,varc,dilr,qa,overall",
    [
        ("GENERAL", False, 80, 80, 75, 85),
        ("NC_OBC", False, 70, 65, 65, 75),
        ("EWS", False, 70, 65, 65, 75),
        ("SC", False, 65, 60, 60, 70),
        ("ST", False, 55, 55, 55, 65),
        ("GENERAL", True, 45, 45, 45, 55),
        ("SC", True, 45, 45, 45, 55),  # PwD row overrides the category row
    ],
)
def test_stage1_cutoff_exactly_on_boundary_passes(
    model: IIMMModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(
            category=category,
            pwd=pwd,
            cat_varc_percentile=varc,
            cat_dilr_percentile=dilr,
            cat_qa_percentile=qa,
            cat_overall_percentile=overall,
        )
    )
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is True, reasons


@pytest.mark.parametrize(
    "category,pwd,varc,dilr,qa,overall",
    [
        ("GENERAL", False, 80, 80, 75, 85),
        ("NC_OBC", False, 70, 65, 65, 75),
        ("EWS", False, 70, 65, 65, 75),
        ("SC", False, 65, 60, 60, 70),
        ("ST", False, 55, 55, 55, 65),
        ("GENERAL", True, 45, 45, 45, 55),
    ],
)
def test_stage1_cutoff_just_below_boundary_fails(
    model: IIMMModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(
            category=category,
            pwd=pwd,
            cat_varc_percentile=varc - 0.01,
            cat_dilr_percentile=dilr,
            cat_qa_percentile=qa,
            cat_overall_percentile=overall,
        )
    )
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is False, reasons

    candidate_overall_fail = CandidateInput(
        **base_candidate_kwargs(
            category=category,
            pwd=pwd,
            cat_varc_percentile=varc,
            cat_dilr_percentile=dilr,
            cat_qa_percentile=qa,
            cat_overall_percentile=overall - 0.01,
        )
    )
    passed2, reasons2 = model.check_cutoffs(candidate_overall_fail)
    assert passed2 is False, reasons2


def test_pwd_row_overrides_category_regardless_of_category(model: IIMMModel) -> None:
    """A PwD candidate uses the (more relaxed) PWD row even if their reservation
    category's own row would otherwise fail."""
    candidate = CandidateInput(
        **base_candidate_kwargs(
            category="GENERAL",
            pwd=True,
            cat_varc_percentile=50,
            cat_dilr_percentile=50,
            cat_qa_percentile=50,
            cat_overall_percentile=60,
        )
    )
    # Would fail GENERAL (80/80/75/85) but passes PWD (45/45/45/55).
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is True, reasons


# ---------------------------------------------------------------------------
# Call prediction: eligible + Stage I passed + overall percentile at/above
# the midpoint of the estimated per-category range (varc/dilr/qa fixed high
# enough above every category's Stage I minimum so only "overall" varies).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,midpoint",
    [
        ("GENERAL", 97.5),
        ("EWS", 90.5),
        ("NC_OBC", 92.5),
        ("SC", 80.5),
        ("ST", 68.5),
    ],
)
def test_call_at_and_below_estimated_overall_midpoint(model: IIMMModel, category: str, midpoint: float) -> None:
    params = model.load_reference_params()

    at_midpoint = CandidateInput(**base_candidate_kwargs(category=category, cat_overall_percentile=midpoint))
    result = model.compute_score(at_midpoint, params=params, mode="reference")
    assert result.cutoff_passed is True, result.cutoff_reasons
    assert result.call is True, result.call_reasons

    below_midpoint = CandidateInput(
        **base_candidate_kwargs(category=category, cat_overall_percentile=midpoint - 0.5)
    )
    result = model.compute_score(below_midpoint, params=params, mode="reference")
    assert result.call is False, result.call_reasons


def test_call_true_when_default_reference_thresholds_met(model: IIMMModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(cat_overall_percentile=99))  # above GENERAL's 97.5 midpoint
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is True
    assert result.cutoff_passed is True
    assert result.call is True, result.call_reasons
    # No AR/APWE score is computed for IIMM.
    assert result.raw_ar is None
    assert result.ncs is None


def test_call_false_when_stage1_not_met(model: IIMMModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(cat_overall_percentile=50, cat_varc_percentile=50, cat_dilr_percentile=50, cat_qa_percentile=50)
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.cutoff_passed is False
    assert result.call is False, result.call_reasons


def test_call_false_when_not_eligible(model: IIMMModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(ug_pct=40))  # below even the relaxed 45% minimum
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is False
    assert result.call is False, result.call_reasons


def test_pwd_call_uses_sectional_fallback(model: IIMMModel) -> None:
    """No overall-percentile estimate is available for PwD yet, so PwD candidates
    fall back to the sectional Stage I PWD minimums for the call decision."""
    params = model.load_reference_params()

    candidate = CandidateInput(
        **base_candidate_kwargs(
            category="GENERAL",
            pwd=True,
            cat_overall_percentile=60,
            cat_varc_percentile=50,
            cat_dilr_percentile=50,
            cat_qa_percentile=50,
        )
    )
    result = model.compute_score(candidate, params=params, mode="reference")
    assert result.cutoff_passed is True  # clears the PWD Stage I row (45/45/45/55)
    assert result.call is True, result.call_reasons  # also clears the PWD call_cutoff row


def test_call_overall_percentile_range_values(model: IIMMModel) -> None:
    params = model.load_reference_params()
    assert params["call_overall_percentile_range"] == {
        "GENERAL": {"low": 97, "high": 98},
        "EWS": {"low": 90, "high": 91},
        "NC_OBC": {"low": 92, "high": 93},
        "SC": {"low": 80, "high": 81},
        "ST": {"low": 68, "high": 69},
    }
