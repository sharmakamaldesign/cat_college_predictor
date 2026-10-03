"""Aggregate call predictions across every registered college for one candidate.

This is the "check every IIM we support at once" entry point: given a single
candidate payload, it runs each registered college's own eligibility / CAT
cutoff / call logic (in its default 'reference' mode) and returns one
combined response.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import ValidationError

from . import colleges  # noqa: F401  (import triggers registration of all colleges)
from .core.models import CandidateInput
from .core.registry import get_college, list_colleges
from .core.utils import call_probability_tier

_PROBABILITY_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}
_RECOMMENDATION_LABELS = {"high": "Safe", "medium": "Moderate", "low": "Risky"}

MODEL_VERSION = "bytlbs_cat_clg_v1"


def _meta() -> Dict[str, str]:
    return {
        "model_used": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }


def predict_all_colleges(candidate_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run every registered college's prediction for one candidate payload.

    Returns::

        {
            "success": bool,
            "message": str,
            "data": {"eligible_colleges": [
                {
                    "college_name": str,
                    "city": str,
                    "predicted_call_probability": "High" | "Medium" | "Low",
                    "recommendation": "Safe" | "Moderate" | "Risky",
                },
                ...
            ]} | None,
            "warnings": [str, ...],   # always present on success; [] when there's nothing to report
            "meta": {"model_used": str, "generated_at": str},  # always present, incl. on failure
        }

    A college is left out of ``eligible_colleges`` when the candidate does
    not meet its *basic* eligibility (e.g. UG percentage too low) — there is
    no point predicting a call for a college the candidate can't apply to at
    all. A college whose model needs candidate fields this payload doesn't
    supply is skipped and noted under ``warnings`` rather than raising.

    ``predicted_call_probability`` / ``recommendation`` are a fixed,
    deterministic bucketing of each college's own call decision and margin
    over its own call threshold — not a probability/ML model.
    """
    try:
        candidate = CandidateInput(**candidate_data)
    except ValidationError as exc:
        return {
            "success": False,
            "message": f"Invalid candidate input: {_format_validation_error(exc)}",
            "data": None,
            "meta": _meta(),
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Invalid candidate input: {exc}",
            "data": None,
            "meta": _meta(),
        }

    eligible_colleges: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for code in list_colleges():
        model = get_college(code)
        try:
            params = model.load_reference_params()
            result = model.compute_score(candidate, params=params, mode=params["mode"])
        except ValueError as exc:
            warnings.append(f"{code}: skipped — {exc}")
            continue

        if not result.eligible:
            continue

        college_meta = model.config["college"]
        tier = call_probability_tier(result.call, result.call_metric_value, result.call_threshold)

        eligible_colleges.append(
            {
                "college_name": college_meta["name"],
                "city": college_meta.get("city", ""),
                "predicted_call_probability": _PROBABILITY_LABELS[tier],
                # "previous_cutoff": result.call_threshold,
                "recommendation": _RECOMMENDATION_LABELS[tier],
            }
        )

    return {
        "success": True,
        "message": "Prediction generated successfully.",
        "data": {"eligible_colleges": eligible_colleges},
        "warnings": warnings,
        "meta": _meta(),
    }


def _format_validation_error(exc: ValidationError) -> str:
    """Turn pydantic's verbose ValidationError into a short, plain-English message,
    e.g. "cat_overall_percentile is a required field." instead of the full dump."""
    missing: List[str] = []
    invalid: List[str] = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "input"
        if err["type"] == "missing":
            missing.append(field)
        else:
            invalid.append(f"{field} ({err['msg']})")

    parts: List[str] = []
    if missing:
        if len(missing) == 1:
            parts.append(f"{missing[0]} is a required field")
        else:
            parts.append(f"{', '.join(missing)} are required fields")
    if invalid:
        parts.append(f"invalid value for {', '.join(invalid)}")
    return "; ".join(parts) + "." if parts else str(exc)
