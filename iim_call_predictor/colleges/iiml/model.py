"""IIM Lucknow call prediction — MBA (Entrepreneurship & Innovation) Batch
2024-26, CAT-2023.

Scope: eligibility + Stage 1a/1b call prediction (invitation to the
Personal Interview + Business Plan presentation) ONLY, via the official
Stage 1b composite score. IIM Lucknow's Stage 2 composite (Business Plan
40% + Interview 60%, with an 18/60 interview pass mark) determines FINAL
SELECTION, after the PI — out of scope, and not implemented. CAT track
only; GMAT is not supported (no GMAT input fields exist in this tool).

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
from ...core.utils import clamp

_HERE = Path(__file__).parent


@register_college("iiml")
class IIMLModel(CollegeModel):
    """Eligibility + call (PI/Business-Plan shortlisting) prediction for IIM Lucknow."""

    def __init__(self, config_path: Optional[Path] = None, reference_params_path: Optional[Path] = None) -> None:
        self.config: Dict[str, Any] = _load_yaml(config_path or _HERE / "config.yaml")
        self._reference_params_path = reference_params_path or _HERE / "reference_params.yaml"

    # ------------------------------------------------------------------
    # Eligibility: bachelor's degree with minimum marks (per CAT's own
    # eligibility baseline, which this policy references but doesn't restate)
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
            reasons.append(f"UG percentage {candidate.ug_pct} is below the required minimum of {min_pct}.")
        else:
            reasons.append(f"UG percentage check passed: {candidate.ug_pct} >= {min_pct}.")

        return candidate.ug_pct >= min_pct, reasons

    # ------------------------------------------------------------------
    # Stage 1a: Table 1 minimum CAT percentile cutoffs
    # ------------------------------------------------------------------
    def _cutoff_row(self, table: Dict[str, Dict[str, float]], candidate: CandidateInput) -> Dict[str, float]:
        # A PwD candidate uses the PWD row regardless of their reservation category.
        return table["PWD"] if candidate.pwd else table[candidate.category]

    def check_cutoffs(self, candidate: CandidateInput) -> Tuple[bool, List[str]]:
        row = self._cutoff_row(self.config["stage1_cutoffs"], candidate)
        reasons: List[str] = []

        checks = [
            ("Overall", candidate.cat_overall_percentile, row["overall"]),
            ("VARC", candidate.cat_varc_percentile, row["varc"]),
            ("DILR", candidate.cat_dilr_percentile, row["dilr"]),
            ("QA", candidate.cat_qa_percentile, row["qa"]),
        ]
        passed = True
        for label, actual, required in checks:
            if actual < required:
                passed = False
                reasons.append(f"{label} percentile {actual} is below the Stage 1a minimum of {required}.")
            else:
                reasons.append(f"{label} percentile {actual} meets the Stage 1a minimum of {required}.")

        row_label = "PwD" if candidate.pwd else candidate.category
        if passed:
            reasons.append(f"All Stage 1a minimum CAT-2023 percentile cutoffs met (category: {row_label}).")
        else:
            reasons.append(
                "Candidate does not meet Stage 1a minimum CAT-2023 cutoffs; not eligible for further "
                "consideration (Stage 1b PI/Business-Plan shortlisting)."
            )
        return passed, reasons

    # ------------------------------------------------------------------
    # Reference parameters (Stage 1b composite call cutoff)
    # ------------------------------------------------------------------
    def load_reference_params(self) -> Dict[str, Any]:
        """Load the ASSUMPTION-labelled pre_pi_call_cutoff table from reference_params.yaml."""
        raw = _load_yaml(self._reference_params_path)
        return {"mode": "reference", "pre_pi_call_cutoff": dict(raw["pre_pi_call_cutoff"])}

    # ------------------------------------------------------------------
    # Stage 1b composite building blocks
    # ------------------------------------------------------------------
    def _cat_score_points(self, cat_overall_percentile: float, weight: float) -> float:
        """Approximates CS = (candidate's CAT score / highest CAT score in the pool) * weight."""
        return (cat_overall_percentile / 100.0) * weight

    def _twelfth_points(self, twelfth_pct: float, weight: float) -> float:
        """Approximates 12M = [{max(P, 80) - 80} / 20] * 20, where P is officially the
        candidate's Class 12 percentile WITHIN the applicant pool; approximated here
        by treating twelfth_pct as if it were already that pool percentile."""
        return clamp(twelfth_pct - 80.0, 0.0, weight)

    def _bachelor_points(self, ug_pct: float, weight: float) -> float:
        """Approximates GM, officially a z-score against the pool's per-discipline mean/SD."""
        return (ug_pct / 100.0) * weight

    def _work_experience_points(self, months: float, weight: float) -> float:
        we = self.config["work_experience"]
        threshold = we["threshold_months"]
        if months <= threshold:
            return 0.0
        return min((months - threshold) * we["per_month_rate"], weight)

    def _gender_points(self, gender: str, weight: float) -> float:
        # The policy's own wording is "IF (Gender = Female)" — narrower than the
        # "non-Male" convention used elsewhere in this project.
        return weight if gender == "Female" else 0.0

    # ------------------------------------------------------------------
    # compute_score: eligibility + cutoffs + Stage 1b composite + call
    # ------------------------------------------------------------------
    def compute_score(
        self,
        candidate: CandidateInput,
        params: Optional[Dict[str, Any]] = None,
        mode: str = "reference",
    ) -> ScoreResult:
        self._require_present(candidate, "twelfth_pct", "gender", "work_ex_months")

        if params is None:
            params = self.load_reference_params()
            mode = "reference"

        weights = self.config["composite_weights"]

        cs_points = self._cat_score_points(candidate.cat_overall_percentile, weights["cat_score_weight"])
        twelfth_points = self._twelfth_points(candidate.twelfth_pct, weights["twelfth_weight"])
        bachelor_points = self._bachelor_points(candidate.ug_pct, weights["bachelor_weight"])
        work_ex_points = self._work_experience_points(candidate.work_ex_months, weights["work_experience_weight"])
        gender_points = self._gender_points(candidate.gender, weights["gender_diversity_points"])

        composite = cs_points + twelfth_points + bachelor_points + work_ex_points + gender_points

        eligible, eligibility_reasons = self.check_basic_eligibility(candidate)
        cutoff_passed, cutoff_reasons = self.check_cutoffs(candidate)
        call, call_reasons, call_threshold = self._predict_call(eligible, cutoff_passed, composite, candidate, params)

        warnings: List[str] = [
            "IIM Lucknow's Stage 2 composite (Business Plan 40% + Interview 60%, with an 18/60 interview "
            "pass mark) determines FINAL SELECTION, after the PI — not the call itself. This predictor "
            "only checks eligibility, Stage 1a CAT cutoffs, and the Stage 1b composite score that "
            "determines the PI/Business-Plan call.",
            "This programme also accepts a valid GMAT score as an alternative to CAT; only the CAT track "
            "is implemented here (no GMAT input fields exist in this tool).",
            "Three Stage 1b components are officially computed against unpublished applicant-pool "
            "statistics (highest CAT score in the pool; Class 12 percentile within the pool by board and "
            "discipline; graduation-score mean/SD by discipline). Each is approximated instead: CAT score "
            "as (cat_overall_percentile/100)*45, Class 12 as clamp(twelfth_pct-80, 0, 20), and graduation "
            "as (ug_pct/100)*25 — rough proxies, not the official calculations. Work experience and the "
            "gender bonus use their exact published formulas.",
            "IIM Lucknow does not publish the Stage 1b composite cut-off used to call candidates for the "
            "PI/Business-Plan stage. The threshold used here (reference_params.yaml -> "
            "pre_pi_call_cutoff) is an externally estimated, user-supplied figure per category, not "
            "published by IIM Lucknow; the PWD figure is explicitly noted as low-confidence.",
        ]
        if not cutoff_passed:
            warnings.append(
                "Candidate did not meet Stage 1a minimum cutoffs; no call is possible regardless of the "
                "Stage 1b composite score."
            )

        return ScoreResult(
            college=self.name,
            eligible=eligible,
            eligibility_reasons=eligibility_reasons,
            cutoff_passed=cutoff_passed,
            cutoff_reasons=cutoff_reasons,
            rating_scores={
                "CATScore": cs_points,
                "Twelfth": twelfth_points,
                "Bachelor": bachelor_points,
                "WorkEx": work_ex_points,
                "Gender": gender_points,
            },
            raw_composite=composite,
            call=call,
            call_reasons=call_reasons,
            call_metric="Stage 1b Composite Score (approx.)",
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
                "Candidate did not meet Stage 1a minimum CAT-2023 cutoffs (see cutoff_reasons); no call predicted."
            )
            return False, reasons, None

        row_label = "PWD" if candidate.pwd else candidate.category
        threshold = params["pre_pi_call_cutoff"][row_label]
        call = composite >= threshold
        if call:
            reasons.append(
                f"Stage 1b composite score {composite:.4f} meets the estimated call cutoff of {threshold} "
                f"for category {row_label}."
            )
        else:
            reasons.append(
                f"Stage 1b composite score {composite:.4f} is below the estimated call cutoff of "
                f"{threshold} for category {row_label}."
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
            {"name": "ug_pct", "type": "float", "required": True, "description": "UG aggregate percentage (0-100)."},
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
            {"name": "cat_overall_percentile", "type": "float", "required": True, "description": "CAT-2023 overall percentile (0-100)."},
            {"name": "cat_varc_percentile", "type": "float", "required": True, "description": "CAT-2023 VARC sectional percentile (0-100)."},
            {"name": "cat_dilr_percentile", "type": "float", "required": True, "description": "CAT-2023 DILR sectional percentile (0-100)."},
            {"name": "cat_qa_percentile", "type": "float", "required": True, "description": "CAT-2023 QA sectional percentile (0-100)."},
            {"name": "twelfth_pct", "type": "float", "required": True, "description": "12th standard percentage (0-100); feeds the Stage 1b composite."},
            {"name": "work_ex_months", "type": "float", "required": True, "description": "Full-time work experience in completed months after graduation, as of July 2023."},
            {"name": "gender", "type": "enum", "required": True, "options": ["Male", "Female", "Other"], "description": "Gender; only 'Female' earns the diversity bonus (per the policy's own wording)."},
        ]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
