"""Small, dependency-free helpers shared by every college plug-in."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


def band_score(value: float, bands: Sequence[Dict[str, Any]]) -> float:
    """Look up a piecewise-constant score from a sorted list of bands.

    Each band is ``{"upper": <inclusive upper bound or None>, "score": <value>}``.
    Bands must be sorted ascending by ``upper``, with a final catch-all band
    using ``upper: None`` for "greater than the previous bound".
    """
    for band in bands:
        upper = band.get("upper")
        if upper is None or value <= upper:
            return float(band["score"])
    raise ValueError(f"No band matched value {value!r}; check band configuration.")


def top_n_average(values: Sequence[float], n: int) -> float:
    """Average of the top ``n`` values (or all values, if fewer than ``n`` are given)."""
    if not values:
        raise ValueError("Cannot compute top-N average of an empty sequence.")
    n = min(n, len(values))
    top = sorted(values, reverse=True)[:n]
    return sum(top) / len(top)


def clamped_top_pct_count(total_count: int, pct: float, min_n: int, max_n: int) -> int:
    """Number of entries that make up the top ``pct``% of a pool, clamped to [min_n, max_n]."""
    if total_count <= 0:
        raise ValueError("total_count must be positive.")
    raw_n = math.ceil(total_count * pct / 100.0)
    return int(clamp(raw_n, min_n, max_n))
