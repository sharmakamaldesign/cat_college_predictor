"""Pydantic models shared by every college plug-in.

These models intentionally stay generic. A college that needs fields beyond
the common ones can define its own small pydantic model for college-specific
structured inputs.

Nothing in ``core`` should need to change when a new college is added.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Gender = Literal["Male", "Female", "Other"]
Category = Literal["GENERAL", "EWS", "NC_OBC", "SC", "ST"]


class CandidateInput(BaseModel):
    """Common candidate fields used across IIM shortlisting calculators.

    Not every college needs every field (e.g. IIM Mumbai's call prediction is
    CAT-percentile-only and doesn't use academics/work-ex/discipline at all),
    so fields only some colleges need are optional here; each college's model
    validates that the fields *it* needs are present and raises a clear error
    otherwise.
    """

    ug_pct: float = Field(..., ge=0, le=100, description="Undergraduate aggregate percentage.")
    category: Category
    pwd: bool = Field(default=False, description="Person with disability status.")

    cat_overall_percentile: float = Field(..., ge=0, le=100)
    cat_varc_percentile: float = Field(..., ge=0, le=100)
    cat_dilr_percentile: float = Field(..., ge=0, le=100)
    cat_qa_percentile: float = Field(..., ge=0, le=100)

    tenth_pct: Optional[float] = Field(default=None, ge=0, le=100, description="10th standard percentage.")
    twelfth_pct: Optional[float] = Field(default=None, ge=0, le=100, description="12th standard percentage.")
    work_ex_months: Optional[float] = Field(
        default=None, ge=0, description="Full-time work experience in completed months."
    )
    gender: Optional[Gender] = None
    ug_discipline: Optional[str] = Field(default=None, description="Candidate's primary UG discipline.")


class ScoreResult(BaseModel):
    """Full, JSON-serializable, step-by-step breakdown of a college's score computation."""

    college: str = Field(..., description="Registry key of the college this result was computed for.")

    eligible: bool
    eligibility_reasons: List[str] = Field(default_factory=list)

    cutoff_passed: bool
    cutoff_reasons: List[str] = Field(default_factory=list)

    rating_scores: Dict[str, float] = Field(default_factory=dict, description="e.g. {'A':.., 'B':.., ...}")
    raw_ar: Optional[float] = None
    normalized_ar: Optional[float] = None
    raw_composite: Optional[float] = None
    ncs: Optional[float] = None

    discipline_used: Optional[str] = None

    call: bool = Field(..., description="Whether the candidate is predicted to receive a shortlisting call.")
    call_reasons: List[str] = Field(default_factory=list)

    params_used: Dict[str, Any] = Field(default_factory=dict)
    mode: str = Field(..., description="'reference' or 'pool'.")
    warnings: List[str] = Field(default_factory=list)
