"""IIM Mumbai call prediction — MBA Batch 2026-2028, CAT-2025.

Scope: eligibility + Stage I/II CAT-percentile call prediction ONLY. IIM
Mumbai's Academic Performance & Work Experience (APWE) score and Personal
Interview score are used only for Stage III final selection, after the PI —
out of scope here, and not implemented.

All tables/cutoffs are read from ``config.yaml`` / ``reference_params.yaml``
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


@register_college("iimm")
class IIMMModel(CollegeModel):
    """Eligibility + call (Personal Interview shortlisting) prediction for IIM Mumbai."""

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
    # Stage I: Table 1 minimum CAT-2025 percentile cutoffs
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
                reasons.append(f"{label} percentile {actual} is below the Stage I minimum of {required}.")
            else:
                reasons.append(f"{label} percentile {actual} meets the Stage I minimum of {required}.")

        row_label = "PwD" if candidate.pwd else candidate.category
        if passed:
            reasons.append(f"All Stage I minimum CAT-2025 percentile cutoffs met (category: {row_label}).")
        else:
            reasons.append(
                "Candidate does not meet Stage I minimum CAT-2025 cutoffs; not eligible for further "
                "consideration (Stage II PI shortlisting)."
            )
        return passed, reasons

    # ------------------------------------------------------------------
    # Reference parameters (call cutoff)
    # ------------------------------------------------------------------
    def load_reference_params(self) -> Dict[str, Any]:
        """Load the static ASSUMPTION-labelled call parameters from reference_params.yaml."""
        raw = _load_yaml(self._reference_params_path)
        return {
            "mode": "reference",
            "call_overall_percentile_range": dict(raw["call_overall_percentile_range"]),
            "call_cutoff": dict(raw["call_cutoff"]),
        }

    # ------------------------------------------------------------------
    # Call prediction
    # ------------------------------------------------------------------
    def _predict_call(
        self, candidate: CandidateInput, eligible: bool, cutoff_passed: bool, params: Dict[str, Any]
    ) -> Tuple[bool, List[str], Optional[float]]:
        """Returns (call, reasons, overall_percentile_threshold_used)."""
        reasons: List[str] = []

        if not eligible:
            reasons.append("Candidate is not eligible (see eligibility_reasons); no call predicted.")
            return False, reasons, None
        if not cutoff_passed:
            reasons.append(
                "Candidate did not meet Stage I minimum CAT-2025 cutoffs (see cutoff_reasons); no call predicted."
            )
            return False, reasons, None

        if candidate.pwd:
            # No updated overall-percentile estimate has been supplied for PwD
            # candidates yet; fall back to the sectional Stage I PWD minimums.
            row = params["call_cutoff"]["PWD"]
            checks = [
                ("Overall", candidate.cat_overall_percentile, row["overall"]),
                ("VARC", candidate.cat_varc_percentile, row["varc"]),
                ("DILR", candidate.cat_dilr_percentile, row["dilr"]),
                ("QA", candidate.cat_qa_percentile, row["qa"]),
            ]
            call = True
            for label, actual, required in checks:
                if actual < required:
                    call = False
                    reasons.append(f"{label} percentile {actual} is below the estimated call cutoff of {required}.")
                else:
                    reasons.append(f"{label} percentile {actual} meets the estimated call cutoff of {required}.")
            if call:
                reasons.append("Candidate meets all estimated Stage II call-cutoff percentiles (PwD).")
            return call, reasons, row["overall"]

        range_row = params["call_overall_percentile_range"][candidate.category]
        threshold = (range_row["low"] + range_row["high"]) / 2
        actual = candidate.cat_overall_percentile
        call = actual >= threshold
        if call:
            reasons.append(
                f"Overall percentile {actual} meets the estimated call cutoff of {threshold} for category "
                f"{candidate.category} (estimated range: {range_row['low']}-{range_row['high']})."
            )
        else:
            reasons.append(
                f"Overall percentile {actual} is below the estimated call cutoff of {threshold} for category "
                f"{candidate.category} (estimated range: {range_row['low']}-{range_row['high']})."
            )
        return call, reasons, threshold

    # ------------------------------------------------------------------
    # compute_score: eligibility + cutoffs + call only (no APWE/AR score)
    # ------------------------------------------------------------------
    def compute_score(
        self,
        candidate: CandidateInput,
        params: Optional[Dict[str, Any]] = None,
        mode: str = "reference",
    ) -> ScoreResult:
        if params is None:
            params = self.load_reference_params()
            mode = "reference"

        eligible, eligibility_reasons = self.check_basic_eligibility(candidate)
        cutoff_passed, cutoff_reasons = self.check_cutoffs(candidate)
        call, call_reasons, call_threshold = self._predict_call(candidate, eligible, cutoff_passed, params)

        warnings: List[str] = [
            "IIM Mumbai's Academic Performance & Work Experience (APWE) score and Personal Interview score "
            "are used only for Stage III final selection, AFTER the PI — not for the PI call itself. This "
            "predictor only checks eligibility and CAT-2025 percentile cutoffs; no AR/composite score is "
            "computed.",
            "Stage I cutoffs (config.yaml) are IIM Mumbai's own published MINIMUM requirements, only a "
            "necessary condition. The call prediction itself uses an externally estimated overall CAT-2025 "
            "percentile range per category (reference_params.yaml -> call_overall_percentile_range), NOT "
            "published by IIM Mumbai; its midpoint is used as the go/no-go threshold. PwD candidates fall "
            "back to a sectional Stage-I-minimum check (no updated PwD estimate is available yet).",
        ]
        if not cutoff_passed:
            warnings.append(
                "Candidate did not meet Stage I minimum cutoffs; no call is possible regardless of the "
                "estimated call cutoff."
            )

        return ScoreResult(
            college=self.name,
            eligible=eligible,
            eligibility_reasons=eligibility_reasons,
            cutoff_passed=cutoff_passed,
            cutoff_reasons=cutoff_reasons,
            call=call,
            call_reasons=call_reasons,
            call_metric="CAT Overall Percentile",
            call_metric_value=candidate.cat_overall_percentile,
            call_threshold=call_threshold,
            params_used=params,
            mode=mode,
            warnings=warnings,
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
                "description": "Reservation category ('GENERAL' represents IIM Mumbai's 'OPEN' category).",
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
        ]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
