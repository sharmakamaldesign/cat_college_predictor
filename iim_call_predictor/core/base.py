"""Abstract interface every college plug-in must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .models import CandidateInput, ScoreResult


class CollegeModel(ABC):
    """A single college's shortlisting logic.

    Concrete subclasses (e.g. ``IIMAModel``) load all of their numbers,
    tables and cutoffs from their own ``config.yaml`` / reference-params
    file and expose them through this fixed interface. ``core`` never
    needs to know about a specific college's tables to drive it.
    """

    #: Registry key, e.g. "iima". Set by subclasses.
    name: str

    @abstractmethod
    def check_basic_eligibility(self, candidate: CandidateInput) -> Tuple[bool, List[str]]:
        """Return (is_eligible, reasons). Reasons are always populated with
        the specific checks performed, whether they passed or failed."""
        raise NotImplementedError

    @abstractmethod
    def check_cutoffs(self, candidate: CandidateInput) -> Tuple[bool, List[str]]:
        """Return (cutoff_passed, reasons) for the college's admission-test cutoffs."""
        raise NotImplementedError

    @abstractmethod
    def compute_score(
        self,
        candidate: CandidateInput,
        params: Optional[Dict[str, Any]] = None,
        mode: str = "reference",
    ) -> ScoreResult:
        """Compute the full, step-by-step score breakdown for ``candidate``.

        ``params`` supplies any pool-dependent assumptions the college's
        formula needs (e.g. normalization denominators). When omitted, the
        college falls back to its own reference/default parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def describe_required_inputs(self) -> List[Dict[str, Any]]:
        """Describe the fields this college needs, so a UI can build a form
        dynamically. Each item is a dict with at least 'name', 'type',
        'required' and 'description' keys."""
        raise NotImplementedError
