"""IIM Bangalore call prediction — PGP MBA 2026-28, CAT-2025.

Scope: eligibility + Stage I/II call prediction (invitation to Personal
Interview + Writing Ability Test) ONLY, via the official "pre-PI score".
IIM Bangalore's post-PI composite (final selection, after the PI/WAT), the
COVID-era missing-board-score re-weighting rule, and the "automatic top-10
qualifier" rule (needs whole-applicant-pool ranking) are all out of scope
and not implemented — see config.yaml for details.

All tables/weights are read from ``config.yaml`` / ``reference_params.yaml``
next to this file; nothing here is hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ...core.base import CollegeModel
from ...core.models import CandidateInput, ScoreResult
from ...core.registry import register_college

_HERE = Path(__file__).parent


@register_college("iimb")
class IIMBModel(CollegeModel):
    """Eligibility + call (PI/WAT shortlisting) prediction for IIM Bangalore."""

    def __init__(self, config_path: Optional[Path] = None, reference_params_path: Optional[Path] = None) -> None:
        self.config: Dict[str, Any] = _load_yaml(config_path or _HERE / "config.yaml")
        self._reference_params_path = reference_params_path or _HERE / "reference_params.yaml"

    # ------------------------------------------------------------------
    # Eligibility: bachelor's degree (or completed CA/CS/ICWA/FIAI) with
    # minimum marks
    # ------------------------------------------------------------------
    def check_basic_eligibility(self, candidate: CandidateInput) -> Tuple[bool, List[str]]:
        elig = self.config["eligibility"]
        reasons: List[str] = []

        min_pct = elig["min_ug_pct"]["default"]
        relaxed_applies = candidate.category in elig["min_ug_pct"]["relaxed_categories"] or (
            elig["min_ug_pct"]["relaxed_for_pwd"] and candidate.pwd
        )
        if relaxed_applies:
            min_pct = elig["min_ug_pct"]["relaxed"]

        if candidate.ug_pct < min_pct:
            reasons.append(
                f"UG percentage {candidate.ug_pct} is below the required minimum of {min_pct}. "
                "(Candidates whose only qualification is a completed CA/CS/ICWA/FIAI professional "
                "degree should supply their professional-course aggregate percentage as ug_pct.)"
            )
        else:
            reasons.append(f"UG percentage check passed: {candidate.ug_pct} >= {min_pct}.")

        return candidate.ug_pct >= min_pct, reasons

    # ------------------------------------------------------------------
    # Stage I: Table 1 minimum CAT percentile cutoffs + strictly positive
    # section raw scores
    # ------------------------------------------------------------------
    def _cutoff_row(self, table: Dict[str, Dict[str, float]], candidate: CandidateInput) -> Dict[str, float]:
        # A PwD candidate uses the PWD row regardless of their reservation category.
        return table["PWD"] if candidate.pwd else table[candidate.category]

    def check_cutoffs(self, candidate: CandidateInput) -> Tuple[bool, List[str]]:
        self._require_present(candidate, "cat_varc_raw_score", "cat_dilr_raw_score", "cat_qa_raw_score")

        row = self._cutoff_row(self.config["stage1_cutoffs"], candidate)
        reasons: List[str] = []

        percentile_checks = [
            ("Overall", candidate.cat_overall_percentile, row["overall"]),
            ("VARC", candidate.cat_varc_percentile, row["varc"]),
            ("DILR", candidate.cat_dilr_percentile, row["dilr"]),
            ("QA", candidate.cat_qa_percentile, row["qa"]),
        ]
        passed = True
        for label, actual, required in percentile_checks:
            if actual < required:
                passed = False
                reasons.append(f"{label} percentile {actual} is below the Stage I minimum of {required}.")
            else:
                reasons.append(f"{label} percentile {actual} meets the Stage I minimum of {required}.")

        raw_score_checks = [
            ("VARC", candidate.cat_varc_raw_score),
            ("DILR", candidate.cat_dilr_raw_score),
            ("QA", candidate.cat_qa_raw_score),
        ]
        for label, raw_score in raw_score_checks:
            if raw_score <= 0:
                passed = False
                reasons.append(f"{label} raw score {raw_score} is not strictly positive; Stage I requires a positive raw score in every section.")
            else:
                reasons.append(f"{label} raw score {raw_score} is strictly positive.")

        row_label = "PwD" if candidate.pwd else candidate.category
        if passed:
            reasons.append(f"All Stage I minimum CAT-2025 requirements met (category: {row_label}).")
        else:
            reasons.append(
                "Candidate does not meet Stage I minimum CAT-2025 requirements; not eligible for further "
                "consideration (Stage II PI/WAT shortlisting)."
            )
        return passed, reasons

    # ------------------------------------------------------------------
    # Reference parameters (pre-PI composite call cutoff)
    # ------------------------------------------------------------------
    def load_reference_params(self) -> Dict[str, Any]:
        """Load the ASSUMPTION-labelled pre_pi_call_cutoff table from reference_params.yaml."""
        raw = _load_yaml(self._reference_params_path)
        return {"mode": "reference", "pre_pi_call_cutoff": dict(raw["pre_pi_call_cutoff"])}

    # ------------------------------------------------------------------
    # Pre-PI composite building blocks
    # ------------------------------------------------------------------
    def _scaled(self, val: float, weight: float) -> float:
        """Approximate the official z-score standardization by linearly
        scaling a 0-100 percentage/percentile onto [0, weight]."""
        return (val / 100.0) * weight

    def _work_experience_points(self, months: float, weight: float) -> float:
        full_credit_months = self.config["work_experience"]["full_credit_months"]
        if months <= 0:
            return 0.0
        if months < full_credit_months:
            return weight * months / full_credit_months
        return float(weight)

    def _gender_points(self, gender: str, weight: float) -> float:
        return weight if gender != "Male" else 0.0

    # ------------------------------------------------------------------
    # compute_score: eligibility + cutoffs + pre-PI composite + call
    # ------------------------------------------------------------------
    def compute_score(
        self,
        candidate: CandidateInput,
        params: Optional[Dict[str, Any]] = None,
        mode: str = "reference",
    ) -> ScoreResult:
        self._require_present(candidate, "tenth_pct", "twelfth_pct", "gender", "work_ex_months")

        if params is None:
            params = self.load_reference_params()
            mode = "reference"

        weights = self.config["composite_weights"]

        varc_points = self._scaled(candidate.cat_varc_percentile, weights["varc_weight"])
        dilr_points = self._scaled(candidate.cat_dilr_percentile, weights["dilr_weight"])
        qa_points = self._scaled(candidate.cat_qa_percentile, weights["qa_weight"])
        tenth_points = self._scaled(candidate.tenth_pct, weights["tenth_weight"])
        twelfth_points = self._scaled(candidate.twelfth_pct, weights["twelfth_weight"])
        bachelor_points = self._scaled(candidate.ug_pct, weights["bachelor_weight"])
        work_ex_points = self._work_experience_points(candidate.work_ex_months, weights["work_experience_weight"])
        gender_points = self._gender_points(candidate.gender, weights["gender_diversity_points"])

        composite = (
            varc_points
            + dilr_points
            + qa_points
            + tenth_points
            + twelfth_points
            + bachelor_points
            + work_ex_points
            + gender_points
        )

        eligible, eligibility_reasons = self.check_basic_eligibility(candidate)
        cutoff_passed, cutoff_reasons = self.check_cutoffs(candidate)
        call, call_reasons, call_threshold = self._predict_call(eligible, cutoff_passed, composite, candidate, params)

        warnings: List[str] = [
            "IIM Bangalore's post-PI composite (PI=40, WAT=10, CAT=25, re-weighted board/work-ex scores) "
            "determines FINAL SELECTION, after the PI/WAT — not the call itself. This predictor only "
            "checks eligibility, Stage I CAT cutoffs, and the pre-PI composite score that determines the "
            "Stage II (PI/WAT) call.",
            "The official pre-PI formula standardizes CAT sections, 10th, 12th and Bachelor's scores "
            "against an unpublished population mean/SD (max(0, min(wt, wt/2 + ((val-mean)/sd)*wt/6))); "
            "this tool does not collect a candidate pool by default, so each such component is "
            "approximated as (value / 100) * weight instead — a rough proxy, not the official calculation. "
            "Work experience uses the exact published formula (not approximated).",
            "The 'automatic top-10 qualifier' rule (by total CAT score, adjusted bachelor's score, and "
            "professional-course score) and the COVID-era missing-10th/12th-score re-weighting rule both "
            "require whole-applicant-pool context this single-candidate tool doesn't have, and are not "
            "implemented.",
            "IIM Bangalore does not publish the pre-PI composite cut-off used to call candidates for "
            "Stage II. pre_pi_call_cutoff (reference_params.yaml) defaults to 0 for every category as an "
            "ASSUMPTION, i.e. by default 'call' == 'meets Stage I'; this likely OVER-predicts calls until "
            "real figures are supplied.",
        ]
        if not cutoff_passed:
            warnings.append(
                "Candidate did not meet Stage I minimum requirements; no call is possible regardless of "
                "the pre-PI composite score."
            )

        return ScoreResult(
            college=self.name,
            eligible=eligible,
            eligibility_reasons=eligibility_reasons,
            cutoff_passed=cutoff_passed,
            cutoff_reasons=cutoff_reasons,
            rating_scores={
                "VARC": varc_points,
                "DILR": dilr_points,
                "QA": qa_points,
                "Tenth": tenth_points,
                "Twelfth": twelfth_points,
                "Bachelor": bachelor_points,
                "WorkEx": work_ex_points,
                "Gender": gender_points,
            },
            raw_composite=composite,
            call=call,
            call_reasons=call_reasons,
            call_metric="Pre-PI Composite Score (approx.)",
            call_metric_value=composite,
            call_threshold=call_threshold,
            params_used=params,
            mode=mode,
            warnings=warnings,
        )

    def _predict_call(
        self,
        eligible: bool,
        cutoff_passed: bool,
        composite: float,
        candidate: CandidateInput,
        params: Dict[str, Any],
    ) -> Tuple[bool, List[str], Optional[float]]:
        reasons: List[str] = []

        if not eligible:
            reasons.append("Candidate is not eligible (see eligibility_reasons); no call predicted.")
            return False, reasons, None
        if not cutoff_passed:
            reasons.append(
                "Candidate did not meet Stage I minimum CAT-2025 requirements (see cutoff_reasons); no call predicted."
            )
            return False, reasons, None

        row_label = "PWD" if candidate.pwd else candidate.category
        threshold = params["pre_pi_call_cutoff"][row_label]
        call = composite >= threshold
        if call:
            reasons.append(
                f"Pre-PI composite score {composite:.4f} meets the estimated call cutoff of {threshold} for "
                f"category {row_label}."
            )
        else:
            reasons.append(
                f"Pre-PI composite score {composite:.4f} is below the estimated call cutoff of {threshold} "
                f"for category {row_label}."
            )
        return call, reasons, threshold

    def _require_present(self, candidate: CandidateInput, *field_names: str) -> None:
        missing = [name for name in field_names if getattr(candidate, name) is None]
        if missing:
            raise ValueError(
                f"{self.config['college']['name']} scoring requires the following fields: {', '.join(missing)}."
            )

    # ------------------------------------------------------------------
    # Dynamic form description
    # ------------------------------------------------------------------
    def describe_required_inputs(self) -> List[Dict[str, Any]]:
        categories = list(self.config["categories"])
        return [
            {"name": "ug_pct", "type": "float", "required": True, "description": "UG (or, for professional-degree-only candidates, professional-course) aggregate percentage (0-100)."},
            {
                "name": "category",
                "type": "enum",
                "required": True,
                "options": categories,
                "description": "Reservation category.",
            },
            {
                "name": "pwd",
                "type": "bool",
                "required": False,
                "description": "Person with disability status; overrides the category row with the PWD cutoff row.",
            },
            {"name": "cat_overall_percentile", "type": "float", "required": True, "description": "CAT-2025 overall percentile (0-100)."},
            {"name": "cat_varc_percentile", "type": "float", "required": True, "description": "CAT-2025 VARC sectional percentile (0-100)."},
            {"name": "cat_dilr_percentile", "type": "float", "required": True, "description": "CAT-2025 DILR sectional percentile (0-100)."},
            {"name": "cat_qa_percentile", "type": "float", "required": True, "description": "CAT-2025 QA sectional percentile (0-100)."},
            {"name": "cat_varc_raw_score", "type": "float", "required": True, "description": "CAT-2025 VARC section raw score (must be > 0 to pass Stage I)."},
            {"name": "cat_dilr_raw_score", "type": "float", "required": True, "description": "CAT-2025 DILR section raw score (must be > 0 to pass Stage I)."},
            {"name": "cat_qa_raw_score", "type": "float", "required": True, "description": "CAT-2025 QA section raw score (must be > 0 to pass Stage I)."},
            {"name": "tenth_pct", "type": "float", "required": True, "description": "10th standard percentage (0-100); feeds the pre-PI composite."},
            {"name": "twelfth_pct", "type": "float", "required": True, "description": "12th standard percentage (0-100); feeds the pre-PI composite."},
            {"name": "work_ex_months", "type": "float", "required": True, "description": "Full-time work experience in completed months, as of July 2025."},
            {"name": "gender", "type": "enum", "required": True, "options": ["Male", "Female", "Other"], "description": "Gender; Female/Other earn the gender-diversity points."},
        ]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
