"""IIM Ahmedabad shortlisting score model — PGP 2027-29 batch, CAT-2026.

All tables, weights, and cutoffs are read from ``config.yaml`` /
``reference_params.yaml`` next to this file; nothing here is hardcoded.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ...core.base import CollegeModel
from ...core.models import CandidateInput, ScoreResult
from ...core.registry import register_college
from ...core.utils import band_score, clamped_top_pct_count, top_n_average

_HERE = Path(__file__).parent


@register_college("iima")
class IIMAModel(CollegeModel):
    """Score calculation + CAT cutoff check for IIM Ahmedabad."""

    def __init__(self, config_path: Optional[Path] = None, reference_params_path: Optional[Path] = None) -> None:
        self.config: Dict[str, Any] = _load_yaml(config_path or _HERE / "config.yaml")
        self._reference_params_path = reference_params_path or _HERE / "reference_params.yaml"

    # ------------------------------------------------------------------
    # Step 1: basic eligibility
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

        eligible = candidate.ug_pct >= min_pct
        return eligible, reasons

    # ------------------------------------------------------------------
    # Step 7: CAT cutoffs (Table 4)
    # ------------------------------------------------------------------
    def check_cutoffs(self, candidate: CandidateInput) -> Tuple[bool, List[str]]:
        table = self.config["cutoffs"]["pwd" if candidate.pwd else "non_pwd"]
        row = table[candidate.category]
        reasons: List[str] = []

        checks = [
            ("Overall", candidate.cat_overall_percentile, row["overall"]),
            ("VARC", candidate.cat_varc_percentile, row["sectional"]),
            ("DILR", candidate.cat_dilr_percentile, row["sectional"]),
            ("QA", candidate.cat_qa_percentile, row["sectional"]),
        ]
        passed = True
        for label, actual, required in checks:
            if actual < required:
                passed = False
                reasons.append(f"{label} percentile {actual} is below the required cutoff of {required}.")
            else:
                reasons.append(f"{label} percentile {actual} meets the required cutoff of {required}.")

        pwd_label = " (PwD)" if candidate.pwd else ""
        if passed:
            reasons.append(
                f"All CAT-2026 percentile cutoffs met for category {candidate.category}{pwd_label}."
            )
        else:
            reasons.append(
                "Candidate does not meet the CAT-2026 cutoffs; the score below is still computed for "
                "reference but does not represent an actual shortlist outcome."
            )
        return passed, reasons

    # ------------------------------------------------------------------
    # Reference / pool parameters
    # ------------------------------------------------------------------
    def load_reference_params(self) -> Dict[str, Any]:
        """Load the static ASSUMPTION-labelled parameters from reference_params.yaml."""
        raw = _load_yaml(self._reference_params_path)
        return {
            "mode": "reference",
            "avg_top50_AR": raw["avg_top50_AR"],
            "top1pct_avg_raw_composite": dict(raw["top1pct_avg_raw_composite"]),
            "ncs_call_cutoff": dict(raw["ncs_call_cutoff"]),
        }

    def compute_pool_params(self, csv_path: Path | str) -> Dict[str, Any]:
        """Derive avg_top50_AR and per-discipline top-1% averages from a candidate-pool CSV.

        Expected CSV columns: tenth_pct, twelfth_pct, ug_pct, work_ex_months,
        gender, cat_overall_percentile, ug_discipline.
        """
        rows = _read_pool_csv(csv_path)
        if not rows:
            raise ValueError(f"No rows found in pool CSV: {csv_path}")

        weights = self.config["composite_weights"]
        by_discipline: Dict[str, List[float]] = {}
        all_raw_ar: List[float] = []
        row_raw_ar: List[float] = []

        for row in rows:
            A = self._band_score(row["tenth_pct"])
            B = self._band_score(row["twelfth_pct"])
            C = self._band_score(row["ug_pct"])
            D = self._work_ex_score(row["work_ex_months"])
            E = self._gender_score(row["gender"])
            raw_ar = A + B + C + D + E
            all_raw_ar.append(raw_ar)
            row_raw_ar.append(raw_ar)

        top_n = self.config["pool"]["top_n_for_avg_ar"]
        avg_top50_AR = top_n_average(all_raw_ar, top_n)

        ncs_cfg = self.config["ncs"]
        for row, raw_ar in zip(rows, row_raw_ar):
            normalized_ar = raw_ar / avg_top50_AR
            raw_composite = (
                weights["normalized_ar_weight"] * normalized_ar
                + weights["cat_percentile_weight"] * (row["cat_overall_percentile"] / 100.0)
            )
            by_discipline.setdefault(row["ug_discipline"], []).append(raw_composite)

        top1pct_avg_raw_composite: Dict[str, float] = {}
        per_discipline_n: Dict[str, int] = {}
        for discipline, composites in by_discipline.items():
            n = clamped_top_pct_count(
                len(composites), ncs_cfg["top_pct"], ncs_cfg["min_n"], ncs_cfg["max_n"]
            )
            per_discipline_n[discipline] = n
            top1pct_avg_raw_composite[discipline] = top_n_average(composites, n)

        overall_composites = [c for values in by_discipline.values() for c in values]
        n_default = clamped_top_pct_count(
            len(overall_composites), ncs_cfg["top_pct"], ncs_cfg["min_n"], ncs_cfg["max_n"]
        )
        top1pct_avg_raw_composite["default"] = top_n_average(overall_composites, n_default)

        # ncs_call_cutoff is not derived from the pool (there's no ground-truth call
        # outcome in a plain candidate CSV); reuse the static ASSUMPTION values.
        reference_raw = _load_yaml(self._reference_params_path)

        return {
            "mode": "pool",
            "avg_top50_AR": avg_top50_AR,
            "top1pct_avg_raw_composite": top1pct_avg_raw_composite,
            "ncs_call_cutoff": dict(reference_raw["ncs_call_cutoff"]),
            "pool_size": len(rows),
            "n_used_for_avg_top50_AR": min(top_n, len(rows)),
            "per_discipline_n": per_discipline_n,
        }

    # ------------------------------------------------------------------
    # Rating-score building blocks (Step 2)
    # ------------------------------------------------------------------
    def _band_score(self, pct: float) -> float:
        return band_score(pct, self.config["rating_bands"])

    def _work_ex_score(self, months: float) -> float:
        we = self.config["work_experience"]
        if months < we["formula_min_months"]:
            return float(we["below_min_months_score"])
        if months <= we["formula_max_months"]:
            return we["formula_multiplier"] * (months - we["formula_offset"])
        return float(we["above_max_months_score"])

    def _gender_score(self, gender: str) -> float:
        gs = self.config["gender_score"]
        return float(gs["male_score"]) if gender == "Male" else float(gs["non_male_score"])

    def _validate_discipline(self, discipline: str) -> None:
        allowed = self.config["disciplines"]
        if discipline not in allowed:
            raise ValueError(
                f"Unknown UG discipline '{discipline}'. Must be one of: {', '.join(allowed)}."
            )

    def _predict_call(
        self,
        candidate: CandidateInput,
        eligible: bool,
        cutoff_passed: bool,
        ncs: float,
        params: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Predict an AWT/PI call: eligible, CAT cutoffs met, and NCS at/above the
        category's approximate NCS call cutoff (see reference_params.yaml)."""
        reasons: List[str] = []

        if not eligible:
            reasons.append("Candidate is not eligible (see eligibility_reasons); no call predicted.")
            return False, reasons
        if not cutoff_passed:
            reasons.append("Candidate did not meet CAT-2026 cutoffs (see cutoff_reasons); no call predicted.")
            return False, reasons

        threshold = params.get("ncs_call_cutoff", {}).get(candidate.category)
        if threshold is None:
            reasons.append(
                f"No NCS call cutoff configured for category '{candidate.category}'; call cannot be predicted."
            )
            return False, reasons

        call = ncs >= threshold
        if call:
            reasons.append(
                f"NCS {ncs:.6f} meets the approximate call cutoff of {threshold} for category {candidate.category}."
            )
        else:
            reasons.append(
                f"NCS {ncs:.6f} is below the approximate call cutoff of {threshold} for category {candidate.category}."
            )
        return call, reasons

    # ------------------------------------------------------------------
    # Steps 3-6: score computation
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

        avg_top50_AR = params["avg_top50_AR"]
        top1pct_map = params["top1pct_avg_raw_composite"]
        weights = self.config["composite_weights"]

        A = self._band_score(candidate.tenth_pct)
        B = self._band_score(candidate.twelfth_pct)
        D = self._work_ex_score(candidate.work_ex_months)
        E = self._gender_score(candidate.gender)

        discipline = candidate.ug_discipline
        self._validate_discipline(discipline)

        C = self._band_score(candidate.ug_pct)
        raw_ar = A + B + C + D + E
        normalized_ar = raw_ar / avg_top50_AR
        raw_composite = (
            weights["normalized_ar_weight"] * normalized_ar
            + weights["cat_percentile_weight"] * (candidate.cat_overall_percentile / 100.0)
        )
        top1pct_avg = top1pct_map.get(discipline, top1pct_map.get("default"))
        if top1pct_avg is None:
            raise ValueError(
                f"No top1pct_avg_raw_composite value found for discipline '{discipline}' and no 'default' set."
            )
        ncs = raw_composite / top1pct_avg

        eligible, eligibility_reasons = self.check_basic_eligibility(candidate)
        cutoff_passed, cutoff_reasons = self.check_cutoffs(candidate)
        call, call_reasons = self._predict_call(candidate, eligible, cutoff_passed, ncs, params)

        warnings: List[str] = [
            f"NCS uses assumed/derived denominators (avg_top50_AR={avg_top50_AR}, "
            f"top1pct_avg_raw_composite[{discipline}]={top1pct_avg}); mode='{mode}'."
        ]
        if mode == "reference":
            warnings.append(
                "Running in 'reference' mode: avg_top50_AR and top1pct_avg_raw_composite are static "
                "ASSUMPTIONs from reference_params.yaml, not official IIMA data. Treat NCS as an estimate."
            )
        if not cutoff_passed:
            warnings.append(
                "Candidate did not meet CAT-2026 cutoffs; the score is informational only, not an actual "
                "shortlist outcome."
            )
        warnings.append(
            "Call prediction uses an approximate, user-supplied NCS cutoff per category "
            "(reference_params.yaml -> ncs_call_cutoff), not an officially published IIMA threshold."
        )

        return ScoreResult(
            college=self.name,
            eligible=eligible,
            eligibility_reasons=eligibility_reasons,
            cutoff_passed=cutoff_passed,
            cutoff_reasons=cutoff_reasons,
            rating_scores={"A": A, "B": B, "C": C, "D": D, "E": E},
            raw_ar=raw_ar,
            normalized_ar=normalized_ar,
            raw_composite=raw_composite,
            ncs=ncs,
            call=call,
            call_reasons=call_reasons,
            discipline_used=discipline,
            params_used=copy.deepcopy(params),
            mode=mode,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Dynamic form description
    # ------------------------------------------------------------------
    def describe_required_inputs(self) -> List[Dict[str, Any]]:
        disciplines = self.config["disciplines"]
        categories = list(self.config["cutoffs"]["non_pwd"].keys())
        return [
            {"name": "tenth_pct", "type": "float", "required": True, "description": "10th standard percentage (0-100)."},
            {"name": "twelfth_pct", "type": "float", "required": True, "description": "12th standard percentage (0-100)."},
            {"name": "ug_pct", "type": "float", "required": True, "description": "UG aggregate percentage (0-100)."},
            {"name": "work_ex_months", "type": "float", "required": True, "description": "Full-time work experience in months."},
            {"name": "gender", "type": "enum", "required": True, "options": ["Male", "Female", "Other"], "description": "Gender."},
            {"name": "category", "type": "enum", "required": True, "options": categories, "description": "Reservation category."},
            {"name": "pwd", "type": "bool", "required": False, "description": "Person with disability status."},
            {"name": "cat_overall_percentile", "type": "float", "required": True, "description": "CAT-2026 overall percentile (0-100)."},
            {"name": "cat_varc_percentile", "type": "float", "required": True, "description": "CAT-2026 VARC sectional percentile (0-100)."},
            {"name": "cat_dilr_percentile", "type": "float", "required": True, "description": "CAT-2026 DILR sectional percentile (0-100)."},
            {"name": "cat_qa_percentile", "type": "float", "required": True, "description": "CAT-2026 QA sectional percentile (0-100)."},
            {"name": "ug_discipline", "type": "enum", "required": True, "options": disciplines, "description": "Primary UG discipline."},
        ]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_pool_csv(csv_path: Path | str) -> List[Dict[str, Any]]:
    required_cols = {
        "tenth_pct",
        "twelfth_pct",
        "ug_pct",
        "work_ex_months",
        "gender",
        "cat_overall_percentile",
        "ug_discipline",
    }
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pool CSV is missing required columns: {sorted(missing)}")
        for raw_row in reader:
            rows.append(
                {
                    "tenth_pct": float(raw_row["tenth_pct"]),
                    "twelfth_pct": float(raw_row["twelfth_pct"]),
                    "ug_pct": float(raw_row["ug_pct"]),
                    "work_ex_months": float(raw_row["work_ex_months"]),
                    "gender": raw_row["gender"],
                    "cat_overall_percentile": float(raw_row["cat_overall_percentile"]),
                    "ug_discipline": raw_row["ug_discipline"],
                }
            )
    return rows
