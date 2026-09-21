"""IIM Calcutta call prediction — MBA Batch 2026-28, CAT-2025.

Scope: eligibility + Stage I/II CAT-percentile + composite-score call
prediction ONLY. IIM Calcutta's Table 4 (final-selection weights: PI, WAT,
academic diversity, work experience) and Table 5 (academic diversity
categories) apply only to Stage III, AFTER the PI/WAT — out of scope here,
and not implemented.

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
from ...core.utils import range_band_score

_HERE = Path(__file__).parent


@register_college("iimc")
class IIMCModel(CollegeModel):
    """Eligibility + call (PI/WAT shortlisting) prediction for IIM Calcutta."""

    def __init__(self, config_path: Optional[Path] = None, reference_params_path: Optional[Path] = None) -> None:
        self.config: Dict[str, Any] = _load_yaml(config_path or _HERE / "config.yaml")
        self._reference_params_path = reference_params_path or _HERE / "reference_params.yaml"

    # ------------------------------------------------------------------
    # Eligibility: bachelor's degree with minimum marks
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
    # Stage I: Table 1 minimum CAT-2025 percentile cutoffs + non-negative
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
            if raw_score < 0:
                passed = False
                reasons.append(f"{label} raw score {raw_score} is negative; Stage I requires non-negative section raw scores.")
            else:
                reasons.append(f"{label} raw score {raw_score} is non-negative.")

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
    # Reference parameters (composite call cutoff)
    # ------------------------------------------------------------------
    def load_reference_params(self) -> Dict[str, Any]:
        """Load the ASSUMPTION-labelled composite call parameters from reference_params.yaml.

        `composite_call_cutoff` is derived as the midpoint of each category's
        `composite_call_range`, plus a flat PwD fallback (no updated PwD
        estimate is available yet).
        """
        raw = _load_yaml(self._reference_params_path)
        composite_call_range = dict(raw["composite_call_range"])
        composite_call_cutoff = {
            category: (row["low"] + row["high"]) / 2 for category, row in composite_call_range.items()
        }
        composite_call_cutoff["PWD"] = raw["composite_call_cutoff_pwd_fallback"]
        return {
            "mode": "reference",
            "composite_call_range": composite_call_range,
            "composite_call_cutoff": composite_call_cutoff,
        }

    # ------------------------------------------------------------------
    # Stage II: Table 2/3 composite score building blocks
    # ------------------------------------------------------------------
    def _class10_points(self, pct: float) -> float:
        return range_band_score(pct, self.config["class10_bands"])

    def _class12_points(self, pct: float) -> float:
        return range_band_score(pct, self.config["class12_bands"])

    def _gender_points(self, gender: str) -> float:
        return self.config["composite_weights"]["gender_diversity_points"] if gender != "Male" else 0.0

    # ------------------------------------------------------------------
    # compute_score: eligibility + cutoffs + Stage II composite + call
    # ------------------------------------------------------------------
    def compute_score(
        self,
        candidate: CandidateInput,
        params: Optional[Dict[str, Any]] = None,
        mode: str = "reference",
    ) -> ScoreResult:
        self._require_present(candidate, "tenth_pct", "twelfth_pct", "gender")

        if params is None:
            params = self.load_reference_params()
            mode = "reference"

        weights = self.config["composite_weights"]

        class10_points = self._class10_points(candidate.tenth_pct)
        class12_points = self._class12_points(candidate.twelfth_pct)
        gender_points = self._gender_points(candidate.gender)
        cat_term = (candidate.cat_overall_percentile / 100.0) * weights["cat_score_weight"]
        composite = cat_term + class10_points + class12_points + gender_points

        eligible, eligibility_reasons = self.check_basic_eligibility(candidate)
        cutoff_passed, cutoff_reasons = self.check_cutoffs(candidate)
        call, call_reasons, call_threshold = self._predict_call(eligible, cutoff_passed, composite, candidate, params)

        warnings: List[str] = [
            "IIM Calcutta's Table 4 (final-selection weights: PI, WAT, academic diversity, work experience) "
            "and Table 5 (academic diversity categories) apply only to Stage III, AFTER the PI/WAT — not to "
            "the call itself. This predictor only checks eligibility, Stage I CAT cutoffs, and the Stage II "
            "composite score.",
            "The Stage II 'CAT term' is officially (candidate's CAT scaled score / max possible scaled "
            "score) * 56; neither figure is published, and this tool does not collect a raw scaled score, "
            "so it is approximated as (cat_overall_percentile / 100) * 56 — a rough proxy, not the official "
            "calculation.",
            "IIM Calcutta states the Stage II composite cut-off is decided per-category, per-year, at the "
            "Institute's discretion, and is not published. The threshold used here is the midpoint of an "
            "externally estimated composite-score range per category (reference_params.yaml -> "
            "composite_call_range), NOT published by IIM Calcutta. PwD candidates have no updated estimate "
            "yet and fall back to a cutoff of 0 (no additional bar beyond Stage I).",
        ]
        if not cutoff_passed:
            warnings.append(
                "Candidate did not meet Stage I minimum requirements; no call is possible regardless of "
                "the composite score."
            )

        return ScoreResult(
            college=self.name,
            eligible=eligible,
            eligibility_reasons=eligibility_reasons,
            cutoff_passed=cutoff_passed,
            cutoff_reasons=cutoff_reasons,
            rating_scores={
                "Class10": class10_points,
                "Class12": class12_points,
                "Gender": gender_points,
                "CATTerm": cat_term,
            },
            raw_composite=composite,
            call=call,
            call_reasons=call_reasons,
            call_metric="Stage II Composite Score (approx.)",
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
        threshold = params["composite_call_cutoff"][row_label]
        call = composite >= threshold

        range_row = params.get("composite_call_range", {}).get(row_label)
        range_note = f" (estimated range: {range_row['low']}-{range_row['high']})" if range_row else ""
        if call:
            reasons.append(
                f"Composite score {composite:.4f} meets the estimated call cutoff of {threshold} for "
                f"category {row_label}{range_note}."
            )
        else:
            reasons.append(
                f"Composite score {composite:.4f} is below the estimated call cutoff of {threshold} for "
                f"category {row_label}{range_note}."
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
                "description": "Reservation category ('GENERAL' represents IIM Calcutta's 'OPEN' category).",
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
            {"name": "cat_varc_raw_score", "type": "float", "required": True, "description": "CAT-2025 VARC section raw score (must be >= 0 to pass Stage I)."},
            {"name": "cat_dilr_raw_score", "type": "float", "required": True, "description": "CAT-2025 DILR section raw score (must be >= 0 to pass Stage I)."},
            {"name": "cat_qa_raw_score", "type": "float", "required": True, "description": "CAT-2025 QA section raw score (must be >= 0 to pass Stage I)."},
            {"name": "tenth_pct", "type": "float", "required": True, "description": "10th standard percentage (0-100); feeds the Stage II composite."},
            {"name": "twelfth_pct", "type": "float", "required": True, "description": "12th standard percentage (0-100); feeds the Stage II composite."},
            {"name": "gender", "type": "enum", "required": True, "options": ["Male", "Female", "Other"], "description": "Gender; Female/Other earn the gender-diversity points."},
        ]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
