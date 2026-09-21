"""Tests for the IIM Calcutta plug-in (eligibility + Stage I/II call prediction).

IIM Calcutta's Table 4 (final-selection weights: PI, WAT, academic
diversity, work experience) and Table 5 (academic diversity categories)
apply only to Stage III, after the PI/WAT, so they are out of scope here
and are not tested.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from iim_call_predictor.colleges.iimc.model import IIMCModel
from iim_call_predictor.core.models import CandidateInput
from iim_call_predictor.core.registry import get_college, list_colleges


def base_candidate_kwargs(**overrides: Any) -> Dict[str, Any]:
    """A minimal, valid IIMC candidate payload (comfortably above the GENERAL/OPEN
    Stage I minimums, non-negative raw scores); override fields per-test."""
    data: Dict[str, Any] = {
        "ug_pct": 75,
        "category": "GENERAL",
        "pwd": False,
        "cat_overall_percentile": 95,
        "cat_varc_percentile": 90,
        "cat_dilr_percentile": 90,
        "cat_qa_percentile": 90,
        "cat_varc_raw_score": 30,
        "cat_dilr_raw_score": 25,
        "cat_qa_raw_score": 28,
        "tenth_pct": 85,
        "twelfth_pct": 85,
        "gender": "Male",
    }
    data.update(overrides)
    return data


@pytest.fixture()
def model() -> IIMCModel:
    return IIMCModel()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_iimc_is_registered() -> None:
    assert "iimc" in list_colleges()
    assert isinstance(get_college("iimc"), IIMCModel)


# ---------------------------------------------------------------------------
# Eligibility: UG% >= 50, relaxed to 45 for SC/ST/PwD
# ---------------------------------------------------------------------------


def test_eligibility_default_threshold(model: IIMCModel) -> None:
    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=50)))
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=49.99)))
    assert eligible is False, reasons


def test_eligibility_relaxed_threshold_for_sc_st_and_pwd(model: IIMCModel) -> None:
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
        ("EWS", False, 70, 65, 65, 75),
        ("NC_OBC", False, 70, 65, 65, 75),
        ("SC", False, 65, 60, 60, 70),
        ("ST", False, 55, 55, 55, 65),
        ("GENERAL", True, 45, 45, 45, 55),
        ("SC", True, 45, 45, 45, 55),  # PwD row overrides the category row
    ],
)
def test_stage1_cutoff_exactly_on_boundary_passes(
    model: IIMCModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
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
        ("EWS", False, 70, 65, 65, 75),
        ("NC_OBC", False, 70, 65, 65, 75),
        ("SC", False, 65, 60, 60, 70),
        ("ST", False, 55, 55, 55, 65),
        ("GENERAL", True, 45, 45, 45, 55),
    ],
)
def test_stage1_cutoff_just_below_boundary_fails(
    model: IIMCModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
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


def test_pwd_row_overrides_category_regardless_of_category(model: IIMCModel) -> None:
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
# Stage I: non-negative section raw scores
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["cat_varc_raw_score", "cat_dilr_raw_score", "cat_qa_raw_score"])
def test_negative_raw_score_fails_stage1(model: IIMCModel, field: str) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(**{field: -0.5}))
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is False, reasons


def test_zero_raw_scores_pass_stage1(model: IIMCModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(cat_varc_raw_score=0, cat_dilr_raw_score=0, cat_qa_raw_score=0)
    )
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is True, reasons


def test_missing_raw_scores_raises_clear_error(model: IIMCModel) -> None:
    data = base_candidate_kwargs()
    del data["cat_varc_raw_score"]
    candidate = CandidateInput(**data)
    with pytest.raises(ValueError, match="cat_varc_raw_score"):
        model.check_cutoffs(candidate)


# ---------------------------------------------------------------------------
# Stage II: Table 2/3 composite score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pct,expected",
    [(59.99, 0), (60, 2), (64.99, 2), (65, 4), (69.99, 4), (70, 6), (74.99, 6), (75, 8), (79.99, 8), (80, 10)],
)
def test_class10_bands(model: IIMCModel, pct: float, expected: float) -> None:
    assert model._class10_points(pct) == pytest.approx(expected)


@pytest.mark.parametrize(
    "pct,expected",
    [(59.99, 0), (60, 3), (64.99, 3), (65, 6), (69.99, 6), (70, 9), (74.99, 9), (75, 12), (79.99, 12), (80, 15)],
)
def test_class12_bands(model: IIMCModel, pct: float, expected: float) -> None:
    assert model._class12_points(pct) == pytest.approx(expected)


def test_gender_points_awarded_to_non_male_only(model: IIMCModel) -> None:
    assert model._gender_points("Female") == pytest.approx(4)
    assert model._gender_points("Other") == pytest.approx(4)
    assert model._gender_points("Male") == pytest.approx(0)


def test_composite_score_hand_computed(model: IIMCModel) -> None:
    # tenth=92 (>=80 -> 10), twelfth=88 (>=80 -> 15), Female (+4),
    # cat_overall=99 -> CAT term = 99/100*56 = 55.44
    candidate = CandidateInput(
        **base_candidate_kwargs(
            tenth_pct=92, twelfth_pct=88, gender="Female", cat_overall_percentile=99,
            cat_varc_percentile=95, cat_dilr_percentile=95, cat_qa_percentile=95,
        )
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.rating_scores == {"Class10": 10, "Class12": 15, "Gender": 4, "CATTerm": pytest.approx(55.44)}
    assert result.raw_composite == pytest.approx(84.44)


def test_missing_composite_fields_raises_clear_error(model: IIMCModel) -> None:
    data = base_candidate_kwargs()
    del data["tenth_pct"]
    candidate = CandidateInput(**data)
    params = model.load_reference_params()
    with pytest.raises(ValueError, match="tenth_pct"):
        model.compute_score(candidate, params=params, mode="reference")


# ---------------------------------------------------------------------------
# Call prediction: eligible + Stage I passed + composite >= call cutoff
# (midpoint of an externally estimated per-category range; PwD has no
# updated estimate yet and falls back to 0, i.e. no additional bar)
# ---------------------------------------------------------------------------


def test_composite_call_cutoff_uses_range_midpoints(model: IIMCModel) -> None:
    params = model.load_reference_params()
    cutoffs = params["composite_call_cutoff"]
    assert cutoffs["GENERAL"] == pytest.approx(53.0)
    assert cutoffs["EWS"] == pytest.approx(48.5)
    assert cutoffs["NC_OBC"] == pytest.approx(46.0)
    assert cutoffs["SC"] == pytest.approx(40.5)
    assert cutoffs["ST"] == pytest.approx(34.5)
    assert cutoffs["PWD"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "category,midpoint",
    [
        ("GENERAL", 53.0),
        ("EWS", 48.5),
        ("NC_OBC", 46.0),
        ("SC", 40.5),
        ("ST", 34.5),
    ],
)
def test_predict_call_at_and_below_composite_midpoint(model: IIMCModel, category: str, midpoint: float) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(category=category))
    params = model.load_reference_params()

    call, reasons, threshold = model._predict_call(True, True, midpoint, candidate, params)
    assert call is True, reasons
    assert threshold == pytest.approx(midpoint)

    call, reasons, _ = model._predict_call(True, True, midpoint - 0.01, candidate, params)
    assert call is False, reasons


def test_call_true_when_eligible_and_stage1_passed(model: IIMCModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs())
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is True
    assert result.cutoff_passed is True
    assert result.call is True, result.call_reasons


def test_call_false_when_stage1_not_met(model: IIMCModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(cat_overall_percentile=50, cat_varc_percentile=50, cat_dilr_percentile=50, cat_qa_percentile=50)
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.cutoff_passed is False
    assert result.call is False, result.call_reasons


def test_call_false_when_not_eligible(model: IIMCModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(ug_pct=40))  # below even the relaxed 45% minimum
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is False
    assert result.call is False, result.call_reasons


def test_call_false_when_negative_raw_score(model: IIMCModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(cat_qa_raw_score=-1))
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.cutoff_passed is False
    assert result.call is False, result.call_reasons
