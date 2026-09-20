"""Hand-computed test cases for the IIM Ahmedabad plug-in."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import pytest

from iim_call_predictor.colleges.iima.model import IIMAModel
from iim_call_predictor.core.models import CandidateInput
from iim_call_predictor.core.registry import get_college, list_colleges


def base_candidate_kwargs(**overrides: Any) -> Dict[str, Any]:
    """A minimal, valid IIMA candidate payload; override fields per-test."""
    data: Dict[str, Any] = {
        "tenth_pct": 92,
        "twelfth_pct": 88,
        "ug_pct": 75,
        "work_ex_months": 24,
        "gender": "Female",
        "category": "GENERAL",
        "pwd": False,
        "cat_overall_percentile": 99.5,
        "cat_varc_percentile": 99.5,
        "cat_dilr_percentile": 99.5,
        "cat_qa_percentile": 99.5,
        "ug_discipline": "Engineering & Technology",
    }
    data.update(overrides)
    return data


@pytest.fixture()
def model() -> IIMAModel:
    return IIMAModel()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_iima_is_registered() -> None:
    assert "iima" in list_colleges()
    assert isinstance(get_college("iima"), IIMAModel)


def test_unknown_discipline_rejected(model: IIMAModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(ug_discipline="Astrology"))
    params = model.load_reference_params()
    with pytest.raises(ValueError, match="Unknown UG discipline"):
        model.compute_score(candidate, params=params, mode="reference")


# ---------------------------------------------------------------------------
# Test 1: Table 1 rating-band boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pct,expected",
    [
        (55, 1),
        (55.01, 2),
        (60, 2),
        (60.01, 3),
        (70, 3),
        (70.01, 5),
        (80, 5),
        (80.01, 8),
        (90, 8),
        (90.01, 10),
    ],
)
def test_rating_band_boundaries(model: IIMAModel, pct: float, expected: float) -> None:
    assert model._band_score(pct) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 2: Work-experience score D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "months,expected",
    [
        (11, 0.0),
        (12, 0.2),
        (24, 2.6),
        (36, 5.0),
        (37, 5.0),
    ],
)
def test_work_ex_score(model: IIMAModel, months: float, expected: float) -> None:
    assert model._work_ex_score(months) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 3 & 4: full raw AR / normalized AR / raw composite computation
# ---------------------------------------------------------------------------


def test_full_composite_case_3(model: IIMAModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs())
    params = model.load_reference_params()
    assert params["avg_top50_AR"] == pytest.approx(33)
    assert params["top1pct_avg_raw_composite"]["default"] == pytest.approx(1.0)

    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.rating_scores == {"A": 10, "B": 8, "C": 5, "D": pytest.approx(2.6), "E": 3}
    assert result.raw_ar == pytest.approx(28.6)
    assert result.normalized_ar == pytest.approx(0.866667, abs=1e-6)
    assert result.raw_composite == pytest.approx(0.950083, abs=1e-6)
    assert result.ncs == pytest.approx(0.950083, abs=1e-6)
    # NCS 0.950083 >= GENERAL call cutoff of 0.92, eligible, cutoff passed -> call predicted.
    assert result.call is True, result.call_reasons


def test_full_composite_case_4(model: IIMAModel) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(
            tenth_pct=55,
            twelfth_pct=60,
            ug_pct=60,
            work_ex_months=6,
            gender="Male",
            cat_overall_percentile=99,
        )
    )
    params = model.load_reference_params()
    result = model.compute_score(candidate, params=params, mode="reference")

    assert result.raw_ar == pytest.approx(5)
    assert result.normalized_ar == pytest.approx(0.151515, abs=1e-6)
    assert result.raw_composite == pytest.approx(0.696530, abs=1e-6)
    # NCS 0.696530 is well below the GENERAL call cutoff of 0.92 -> no call.
    assert result.call is False, result.call_reasons


# ---------------------------------------------------------------------------
# Test 5: CAT cutoffs (Table 4), boundary values, with/without PwD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,pwd,overall,sectional",
    [
        ("GENERAL", False, 95, 85),
        ("EWS", False, 95, 85),
        ("NC_OBC", False, 90, 80),
        ("SC", False, 85, 75),
        ("ST", False, 75, 65),
        ("GENERAL", True, 85, 75),
        ("EWS", True, 85, 75),
        ("NC_OBC", True, 80, 70),
        ("SC", True, 75, 65),
        ("ST", True, 65, 55),
    ],
)
def test_cutoff_exactly_on_boundary_passes(
    model: IIMAModel, category: str, pwd: bool, overall: float, sectional: float
) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(
            category=category,
            pwd=pwd,
            cat_overall_percentile=overall,
            cat_varc_percentile=sectional,
            cat_dilr_percentile=sectional,
            cat_qa_percentile=sectional,
        )
    )
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is True, reasons


@pytest.mark.parametrize(
    "category,pwd,overall,sectional",
    [
        ("GENERAL", False, 95, 85),
        ("EWS", False, 95, 85),
        ("NC_OBC", False, 90, 80),
        ("SC", False, 85, 75),
        ("ST", False, 75, 65),
        ("GENERAL", True, 85, 75),
        ("EWS", True, 85, 75),
        ("NC_OBC", True, 80, 70),
        ("SC", True, 75, 65),
        ("ST", True, 65, 55),
    ],
)
def test_cutoff_just_below_boundary_fails(
    model: IIMAModel, category: str, pwd: bool, overall: float, sectional: float
) -> None:
    candidate = CandidateInput(
        **base_candidate_kwargs(
            category=category,
            pwd=pwd,
            cat_overall_percentile=overall - 0.01,
            cat_varc_percentile=sectional,
            cat_dilr_percentile=sectional,
            cat_qa_percentile=sectional,
        )
    )
    passed, reasons = model.check_cutoffs(candidate)
    assert passed is False, reasons

    candidate_sectional_fail = CandidateInput(
        **base_candidate_kwargs(
            category=category,
            pwd=pwd,
            cat_overall_percentile=overall,
            cat_varc_percentile=sectional - 0.01,
            cat_dilr_percentile=sectional,
            cat_qa_percentile=sectional,
        )
    )
    passed2, reasons2 = model.check_cutoffs(candidate_sectional_fail)
    assert passed2 is False, reasons2


# ---------------------------------------------------------------------------
# Test: eligibility relies only on UG percentage now (relaxed for SC/ST/PwD)
# ---------------------------------------------------------------------------


def test_eligibility_default_threshold(model: IIMAModel) -> None:
    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=50)))
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(CandidateInput(**base_candidate_kwargs(ug_pct=49.99)))
    assert eligible is False, reasons


def test_eligibility_relaxed_threshold_for_sc_st_and_pwd(model: IIMAModel) -> None:
    eligible, reasons = model.check_basic_eligibility(
        CandidateInput(**base_candidate_kwargs(category="SC", ug_pct=45))
    )
    assert eligible is True, reasons

    eligible, reasons = model.check_basic_eligibility(
        CandidateInput(**base_candidate_kwargs(category="GENERAL", pwd=True, ug_pct=45))
    )
    assert eligible is True, reasons


# ---------------------------------------------------------------------------
# Test 9: NCS-based call prediction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,threshold",
    [
        ("GENERAL", 0.92),
        ("EWS", 0.92),
        ("NC_OBC", 0.88),
        ("SC", 0.80),
        ("ST", 0.80),
    ],
)
def test_ncs_call_cutoff_values(model: IIMAModel, category: str, threshold: float) -> None:
    params = model.load_reference_params()
    assert params["ncs_call_cutoff"][category] == pytest.approx(threshold)


@pytest.mark.parametrize(
    "category,threshold",
    [
        ("GENERAL", 0.92),
        ("EWS", 0.92),
        ("NC_OBC", 0.88),
        ("SC", 0.80),
        ("ST", 0.80),
    ],
)
def test_predict_call_at_and_below_threshold(model: IIMAModel, category: str, threshold: float) -> None:
    candidate = CandidateInput(**base_candidate_kwargs(category=category))
    params = model.load_reference_params()

    call, reasons = model._predict_call(candidate, eligible=True, cutoff_passed=True, ncs=threshold, params=params)
    assert call is True, reasons

    call, reasons = model._predict_call(
        candidate, eligible=True, cutoff_passed=True, ncs=threshold - 0.001, params=params
    )
    assert call is False, reasons


def test_predict_call_false_when_not_eligible(model: IIMAModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs())
    params = model.load_reference_params()
    call, reasons = model._predict_call(candidate, eligible=False, cutoff_passed=True, ncs=0.99, params=params)
    assert call is False, reasons


def test_predict_call_false_when_cutoff_not_met(model: IIMAModel) -> None:
    candidate = CandidateInput(**base_candidate_kwargs())
    params = model.load_reference_params()
    call, reasons = model._predict_call(candidate, eligible=True, cutoff_passed=False, ncs=0.99, params=params)
    assert call is False, reasons


# ---------------------------------------------------------------------------
# Test 8: Pool mode — top-50 / top-1% (min 5, max 100) derivation
# ---------------------------------------------------------------------------


def test_pool_mode_derives_params_from_synthetic_pool(model: IIMAModel, tmp_path: Path) -> None:
    """Synthetic pool generated only inside this test, never shipped as real data."""
    csv_path = tmp_path / "synthetic_pool.csv"
    fieldnames = [
        "tenth_pct",
        "twelfth_pct",
        "ug_pct",
        "work_ex_months",
        "gender",
        "cat_overall_percentile",
        "ug_discipline",
    ]

    rows = []
    # 60 candidates in discipline "Economics": raw AR increases with index, so the
    # top 50 overall (out of 60) are indices 10..59 (0-indexed), i.e. exclude the 10 lowest.
    for i in range(60):
        pct = 50 + i * 0.5  # increasing 10th/12th/UG pct -> increasing raw AR
        rows.append(
            {
                "tenth_pct": pct,
                "twelfth_pct": pct,
                "ug_pct": pct,
                "work_ex_months": 0,
                "gender": "Male",
                "cat_overall_percentile": 50 + i * 0.5,
                "ug_discipline": "Economics",
            }
        )
    # 3 candidates in a tiny discipline, to exercise the min_n=5 clamp (1% of 3 -> 1, clamped to 5,
    # but clamped to at most the pool size of 3 when averaging).
    for i in range(3):
        rows.append(
            {
                "tenth_pct": 90,
                "twelfth_pct": 90,
                "ug_pct": 90,
                "work_ex_months": 0,
                "gender": "Female",
                "cat_overall_percentile": 90,
                "ug_discipline": "Law",
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    params = model.compute_pool_params(csv_path)

    assert params["mode"] == "pool"
    assert params["pool_size"] == 63
    assert params["n_used_for_avg_top50_AR"] == 50

    # avg_top50_AR should equal the average raw AR of the top 50 candidates by raw AR.
    all_raw_ar = []
    for row in rows:
        A = model._band_score(row["tenth_pct"])
        B = model._band_score(row["twelfth_pct"])
        C = model._band_score(row["ug_pct"])
        D = model._work_ex_score(row["work_ex_months"])
        E = model._gender_score(row["gender"])
        all_raw_ar.append(A + B + C + D + E)
    expected_avg_top50 = sum(sorted(all_raw_ar, reverse=True)[:50]) / 50
    assert params["avg_top50_AR"] == pytest.approx(expected_avg_top50)

    # Economics has 60 candidates -> 1% = 0.6 -> ceil = 1 -> clamped to min_n=5.
    assert params["per_discipline_n"]["Economics"] == 5
    # Law has 3 candidates -> 1% = 0.03 -> ceil = 1 -> clamped to min_n=5, but only 3 available.
    assert params["per_discipline_n"]["Law"] == 5

    assert "Economics" in params["top1pct_avg_raw_composite"]
    assert "Law" in params["top1pct_avg_raw_composite"]
    assert "default" in params["top1pct_avg_raw_composite"]

    # Using pool params end-to-end should produce a valid ScoreResult.
    candidate = CandidateInput(**base_candidate_kwargs(ug_discipline="Economics"))
    result = model.compute_score(candidate, params=params, mode="pool")
    assert result.mode == "pool"
    assert result.ncs > 0
