"""Tests for the IIM Lucknow plug-in (eligibility + Stage 1a/1b call prediction).

IIM Lucknow's Stage 2 composite (Business Plan + Interview, final merit
list) is out of scope here and not tested — only eligibility, Stage 1a CAT
cutoffs, the Stage 1b composite score, and the resulting call prediction.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from iim_call_predictor.colleges.iiml.model import IIMLModel
from iim_call_predictor.core.models import CandidateInput
from iim_call_predictor.core.registry import get_college, list_colleges


def base_candidate_kwargs(**overrides: Any) -> Dict[str, Any]:
    """A minimal, valid IIML candidate payload (comfortably above the GENERAL
    Stage 1a minimums); override fields per-test."""
    data: Dict[str, Any] = {
        "ug_pct": 75,
        "category": "GENERAL",
        "pwd": False,
        "cat_overall_percentile": 95,
        "cat_varc_percentile": 92,
        "cat_dilr_percentile": 92,
        "cat_qa_percentile": 92,
        "twelfth_pct": 85,
        "work_ex_months": 24,
        "gender": "Male",
    }
    data.update(overrides)
    return data


@pytest.fixture()
def model() -> IIMLModel:
    return IIMLModel()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_iiml_is_registered() -> None:
    assert "iiml" in list_colleges()
    assert isinstance(get_college("iiml"), IIMLModel)


# ---------------------------------------------------------------------------
# Eligibility: UG% >= 50, relaxed to 45 for SC/ST/PwD
# ---------------------------------------------------------------------------


def test_eligibility_default_threshold(model: IIMLModel) -> None:
    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=50)))
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=49.99)))
    assert eligible is False, reasons


def test_eligibility_relaxed_threshold_for_sc_st_and_pwd(model: IIMLModel) -> None:
    eligible, reasons = model.check_basic_eligibility(
        CandidateInput(**base_candidate_kwargs(category="SC", ug_pct=45))
    )
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(
        CandidateInput(**base_candidate_kwargs(category="GENERAL", pwd=True, ug_pct=45))
    )
    assert eligible is True, reasons


# ---------------------------------------------------------------------------
# Stage 1a: Table 1 minimum CAT percentile cutoffs, exact boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,pwd,varc,dilr,qa,overall",
    [
        ("GENERAL", False, 85, 85, 85, 90),
        ("EWS", False, 77, 77, 77, 82),
        ("NC_OBC", False, 77, 77, 77, 82),
        ("SC", False, 55, 55, 55, 65),
        ("ST", False, 50, 50, 50, 60),
        ("GENERAL", True, 50, 50, 50, 60),
        ("SC", True, 50, 50, 50, 60),  # PwD row overrides the category row
    ],
)
def test_stage1_cutoff_exactly_on_boundary_passes(
    model: IIMLModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
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
        ("GENERAL", False, 85, 85, 85, 90),
        ("EWS", False, 77, 77, 77, 82),
        ("NC_OBC", False, 77, 77, 77, 82),
        ("SC", False, 55, 55, 55, 65),
        ("ST", False, 50, 50, 50, 60),
        ("GENERAL", True, 50, 50, 50, 60),
    ],
)
def test_stage1_cutoff_just_below_boundary_fails(
    model: IIMLModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
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


def test_pwd_row_overrides_category_regardless_of_category(model: IIMLModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(
            category="GENERAL",
            pwd=True,
            cat_varc_percentile=52,
            cat_dilr_percentile=52,
            cat_qa_percentile=52,
            cat_overall_percentile=61,
        )
    )
    # Would fail GENERAL (85/85/85/90) but passes PWD (50/50/50/60).
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is True, reasons


# ---------------------------------------------------------------------------
# Stage 1b composite score components
# ---------------------------------------------------------------------------


def test_cat_score_points(model: IIMLModel) -> None:
    assert model._cat_score_points(100, 45) == pytest.approx(45)
    assert model._cat_score_points(50, 45) == pytest.approx(22.5)
    assert model._cat_score_points(0, 45) == pytest.approx(0)


@pytest.mark.parametrize(
    "twelfth_pct,expected",
    [(60, 0), (80, 0), (85, 5), (90, 10), (100, 20), (105, 20)],  # clamped at the 20-point weight
)
def test_twelfth_points(model: IIMLModel, twelfth_pct: float, expected: float) -> None:
    assert model._twelfth_points(twelfth_pct, 20) == pytest.approx(expected)


def test_bachelor_points(model: IIMLModel) -> None:
    assert model._bachelor_points(100, 25) == pytest.approx(25)
    assert model._bachelor_points(80, 25) == pytest.approx(20)
    assert model._bachelor_points(0, 25) == pytest.approx(0)


@pytest.mark.parametrize(
    "months,expected",
    [(0, 0), (6, 0), (7, 0.5), (10, 2.0), (16, 5.0), (24, 5.0)],
)
def test_work_experience_points(model: IIMLModel, months: float, expected: float) -> None:
    assert model._work_experience_points(months, 5) == pytest.approx(expected)


def test_gender_points_only_for_female_not_other(model: IIMLModel) -> None:
    """The policy's own wording is 'IF (Gender = Female)' — narrower than the
    'non-Male' convention used by the other colleges in this project."""
    assert model._gender_points("Female", 5) == pytest.approx(5)
    assert model._gender_points("Other", 5) == pytest.approx(0)
    assert model._gender_points("Male", 5) == pytest.approx(0)


def test_composite_score_hand_computed(model: IIMLModel) -> None:
    # cat_overall=90 -> 45*.9=40.5, twelfth=95 -> clamp(95-80,0,20)=15,
    # ug_pct=84 -> 25*.84=21, work_ex=16mo -> (16-6)*0.5=5.0, Female -> 5
    candidate = CandidateInput(
        **base_candidate_kwargs(
            cat_overall_percentile=90, twelfth_pct=95, ug_pct=84, work_ex_months=16, gender="Female",
        )
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.rating_scores["CATScore"] == pytest.approx(40.5)
    assert result.rating_scores["Twelfth"] == pytest.approx(15.0)
    assert result.rating_scores["Bachelor"] == pytest.approx(21.0)
    assert result.rating_scores["WorkEx"] == pytest.approx(5.0)
    assert result.rating_scores["Gender"] == pytest.approx(5.0)
    assert result.raw_composite == pytest.approx(40.5 + 15.0 + 21.0 + 5.0 + 5.0)


def test_missing_composite_fields_raises_clear_error(model: IIMLModel) -> None:
    data = base_candidate_kwargs()
    del data["work_ex_months"]
    candidate = CandidateInput(**data)
    params = model.load_reference_params()
    with pytest.raises(ValueError, match="work_ex_months"):
        model.compute_score(candidate, params=params, mode="reference")


# ---------------------------------------------------------------------------
# Call prediction: eligible + Stage 1a passed + composite >= the externally
# estimated, user-supplied per-category call cutoff
# ---------------------------------------------------------------------------


def test_call_cutoff_values(model: IIMLModel) -> None:
    params = model.load_reference_params()
    cutoffs = params["pre_pi_call_cutoff"]
    assert cutoffs["GENERAL"] == pytest.approx(52)
    assert cutoffs["EWS"] == pytest.approx(39)
    assert cutoffs["NC_OBC"] == pytest.approx(37.5)
    assert cutoffs["SC"] == pytest.approx(30)
    assert cutoffs["ST"] == pytest.approx(21.5)
    assert cutoffs["PWD"] == pytest.approx(12)


@pytest.mark.parametrize(
    "category,threshold",
    [
        ("GENERAL", 52),
        ("EWS", 39),
        ("NC_OBC", 37.5),
        ("SC", 30),
        ("ST", 21.5),
    ],
)
def test_predict_call_at_and_below_threshold(model: IIMLModel, category: str, threshold: float) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(category=category))
    params = model.load_reference_params()

    call, reasons, used = model._predict_call(True, True, threshold, candidate, params)
    assert call is True, reasons
    assert used == pytest.approx(threshold)

    call, reasons, _ = model._predict_call(True, True, threshold - 0.01, candidate, params)
    assert call is False, reasons


def test_call_true_when_eligible_and_stage1_passed(model: IIMLModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs())
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is True
    assert result.cutoff_passed is True
    assert result.call is True, result.call_reasons


def test_call_false_when_stage1_not_met(model: IIMLModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(cat_overall_percentile=50, cat_varc_percentile=50, cat_dilr_percentile=50, cat_qa_percentile=50)
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.cutoff_passed is False
    assert result.call is False, result.call_reasons


def test_call_false_when_not_eligible(model: IIMLModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(ug_pct=40))  # below even the relaxed 45% minimum
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is False
    assert result.call is False, result.call_reasons
