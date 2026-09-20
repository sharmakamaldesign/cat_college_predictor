"""Pydantic models shared by every college plug-in.

These models intentionally stay generic. A college that needs fields beyond
the common ones can define its own small pydantic model for college-specific
structured inputs.

Nothing in ``core`` should need to change when a new college is added.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

Gender = Literal["Male", "Female", "Other"]
Category = Literal["GENERAL", "EWS", "NC_OBC", "SC", "ST"]


class CandidateInput(BaseModel):
    """Common candidate fields used across IIM shortlisting calculators."""

    tenth_pct: float = Field(..., ge=0, le=100, description="10th standard percentage.")
    twelfth_pct: float = Field(..., ge=0, le=100, description="12th standard percentage.")
    ug_pct: float = Field(..., ge=0, le=100, description="Undergraduate aggregate percentage.")
    work_ex_months: float = Field(..., ge=0, description="Full-time work experience in completed months.")
    gender: Gender
    category: Category
    pwd: bool = Field(default=False, description="Person with disability status.")

    cat_overall_percentile: float = Field(..., ge=0, le=100)
    cat_varc_percentile: float = Field(..., ge=0, le=100)
    cat_dilr_percentile: float = Field(..., ge=0, le=100)
    cat_qa_percentile: float = Field(..., ge=0, le=100)

    ug_discipline: str = Field(..., description="Candidate's primary UG discipline.")


class ScoreResult(BaseModel):
    """Full, JSON-serializable, step-by-step breakdown of a college's score computation."""

    college: str = Field(..., description="Registry key of the college this result was computed for.")

    eligible: bool
    eligibility_reasons: List[str] = Field(default_factory=list)

    cutoff_passed: bool
    cutoff_reasons: List[str] = Field(default_factory=list)

    rating_scores: Dict[str, float] = Field(default_factory=dict, description="e.g. {'A':.., 'B':.., ...}")
    raw_ar: float
    normalized_ar: float
    raw_composite: float
    ncs: float

    discipline_used: str

    call: bool = Field(..., description="Whether the candidate is predicted to receive an AWT/PI call.")
    call_reasons: List[str] = Field(default_factory=list)

    params_used: Dict[str, Any] = Field(default_factory=dict)
    mode: str = Field(..., description="'reference' or 'pool'.")
    warnings: List[str] = Field(default_factory=list)
