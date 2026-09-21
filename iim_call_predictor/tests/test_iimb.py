"""Tests for the IIM Bangalore plug-in (eligibility + Stage I/II call prediction).

IIM Bangalore's post-PI composite (final selection, after the PI/WAT), the
COVID-era missing-board-score re-weighting rule, and the "automatic top-10
qualifier" rule all require context out of scope for this single-candidate
call predictor, and are not tested.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from iim_call_predictor.colleges.iimb.model import IIMBModel
from iim_call_predictor.core.models import CandidateInput
from iim_call_predictor.core.registry import get_college, list_colleges


def base_candidate_kwargs(**overrides: Any) -> Dict[str, Any]:
    """A minimal, valid IIMB candidate payload (comfortably above the GENERAL
    Stage I minimums, strictly positive raw scores); override fields per-test."""
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
        "work_ex_months": 24,
        "gender": "Male",
    }
    data.update(overrides)
    return data


@pytest.fixture()
def model() -> IIMBModel:
    return IIMBModel()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_iimb_is_registered() -> None:
    assert "iimb" in list_colleges()
    assert isinstance(get_college("iimb"), IIMBModel)


# ---------------------------------------------------------------------------
# Eligibility: UG% >= 50, relaxed to 45 for SC/ST/PwD
# ---------------------------------------------------------------------------


def test_eligibility_default_threshold(model: IIMBModel) -> None:
    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=50)))
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=49.99)))
    assert eligible is False, reasons


def test_eligibility_relaxed_threshold_for_sc_st_and_pwd(model: IIMBModel) -> None:
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
        ("GENERAL", False, 80, 75, 75, 85),
        ("EWS", False, 70, 65, 65, 75),
        ("NC_OBC", False, 70, 65, 65, 75),
        ("SC", False, 65, 60, 60, 70),
        ("ST", False, 55, 55, 55, 65),
        ("GENERAL", True, 50, 50, 50, 60),
        ("SC", True, 50, 50, 50, 60),  # PwD row overrides the category row
    ],
)
def test_stage1_cutoff_exactly_on_boundary_passes(
    model: IIMBModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
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
        ("GENERAL", False, 80, 75, 75, 85),
        ("EWS", False, 70, 65, 65, 75),
        ("NC_OBC", False, 70, 65, 65, 75),
        ("SC", False, 65, 60, 60, 70),
        ("ST", False, 55, 55, 55, 65),
        ("GENERAL", True, 50, 50, 50, 60),
    ],
)
def test_stage1_cutoff_just_below_boundary_fails(
    model: IIMBModel, category: str, pwd: bool, varc: float, dilr: float, qa: float, overall: float
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


def test_pwd_row_overrides_category_regardless_of_category(model: IIMBModel) -> None:
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
    # Would fail GENERAL (80/75/75/85) but passes PWD (50/50/50/60).
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is True, reasons


# ---------------------------------------------------------------------------
# Stage I: strictly positive (not merely non-negative) section raw scores
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["cat_varc_raw_score", "cat_dilr_raw_score", "cat_qa_raw_score"])
def test_negative_raw_score_fails_stage1(model: IIMBModel, field: str) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(**{field: -0.5}))
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is False, reasons


@pytest.mark.parametrize("field", ["cat_varc_raw_score", "cat_dilr_raw_score", "cat_qa_raw_score"])
def test_zero_raw_score_fails_stage1(model: IIMBModel, field: str) -> None:
    """Unlike IIMC (non-negative), IIMB requires STRICTLY positive raw scores."""
    candidate = CandidateInput(**base_candidate_kwargs(**{field: 0}))
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is False, reasons


def test_positive_raw_scores_pass_stage1(model: IIMBModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(cat_varc_raw_score=0.01, cat_dilr_raw_score=0.01, cat_qa_raw_score=0.01)
    )
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is True, reasons


def test_missing_raw_scores_raises_clear_error(model: IIMBModel) -> None:
    data = base_candidate_kwargs()
    del data["cat_varc_raw_score"]
    candidate = CandidateInput(**data)
    with pytest.raises(ValueError, match="cat_varc_raw_score"):
        model.check_cutoffs(candidate)


# ---------------------------------------------------------------------------
# Pre-PI composite score
# ---------------------------------------------------------------------------


def test_work_experience_formula() -> None:
    model = IIMBModel()
    weight = 10
    assert model._work_experience_points(0, weight) == pytest.approx(0)
    assert model._work_experience_points(18, weight) == pytest.approx(10 * 18 / 36)
    assert model._work_experience_points(36, weight) == pytest.approx(10)
    assert model._work_experience_points(48, weight) == pytest.approx(10)


def test_gender_points_awarded_to_non_male_only(model: IIMBModel) -> None:
    weight = 5
    assert model._gender_points("Female", weight) == pytest.approx(5)
    assert model._gender_points("Other", weight) == pytest.approx(5)
    assert model._gender_points("Male", weight) == pytest.approx(0)


def test_composite_score_hand_computed(model: IIMBModel) -> None:
    # varc=96->19*.96=18.24, dilr=96->21*.96=20.16, qa=96->15*.96=14.4,
    # tenth=92->10*.92=9.2, twelfth=88->10*.88=8.8, bachelor(ug_pct)=75->7.5,
    # work_ex=24mo -> 10*24/36=6.6666..., gender=Female->5
    candidate = CandidateInput(
        **base_candidate_kwargs(
            cat_varc_percentile=96, cat_dilr_percentile=96, cat_qa_percentile=96,
            tenth_pct=92, twelfth_pct=88, ug_pct=75, work_ex_months=24, gender="Female",
        )
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.rating_scores["VARC"] == pytest.approx(18.24)
    assert result.rating_scores["DILR"] == pytest.approx(20.16)
    assert result.rating_scores["QA"] == pytest.approx(14.4)
    assert result.rating_scores["Tenth"] == pytest.approx(9.2)
    assert result.rating_scores["Twelfth"] == pytest.approx(8.8)
    assert result.rating_scores["Bachelor"] == pytest.approx(7.5)
    assert result.rating_scores["WorkEx"] == pytest.approx(10 * 24 / 36)
    assert result.rating_scores["Gender"] == pytest.approx(5)
    assert result.raw_composite == pytest.approx(18.24 + 20.16 + 14.4 + 9.2 + 8.8 + 7.5 + (10 * 24 / 36) + 5)


def test_missing_composite_fields_raises_clear_error(model: IIMBModel) -> None:
    data = base_candidate_kwargs()
    del data["work_ex_months"]
    candidate = CandidateInput(**data)
    params = model.load_reference_params()
    with pytest.raises(ValueError, match="work_ex_months"):
        model.compute_score(candidate, params=params, mode="reference")


# ---------------------------------------------------------------------------
# Call prediction: eligible + Stage I passed + pre-PI composite >= call
# cutoff (defaults to 0 for every category in reference mode, i.e. call ==
# Stage I, until real figures are supplied)
# ---------------------------------------------------------------------------


def test_call_cutoff_defaults_to_zero(model: IIMBModel) -> None:
    params = model.load_reference_params()
    assert all(v == 0 for v in params["pre_pi_call_cutoff"].values())


def test_call_true_when_eligible_and_stage1_passed(model: IIMBModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs())
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is True
    assert result.cutoff_passed is True
    assert result.call is True, result.call_reasons


def test_call_false_when_stage1_not_met(model: IIMBModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(cat_overall_percentile=50, cat_varc_percentile=50, cat_dilr_percentile=50, cat_qa_percentile=50)
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.cutoff_passed is False
    assert result.call is False, result.call_reasons


def test_call_false_when_not_eligible(model: IIMBModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(ug_pct=40))  # below even the relaxed 45% minimum
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.eligible is False
    assert result.call is False, result.call_reasons


def test_call_false_when_raw_score_not_positive(model: IIMBModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(cat_qa_raw_score=0))
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.cutoff_passed is False
    assert result.call is False, result.call_reasons
