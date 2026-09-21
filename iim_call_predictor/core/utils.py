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


def range_band_score(value: float, bands: Sequence[Dict[str, Any]]) -> float:
    """Look up a piecewise-constant score from ``[lower, upper)`` range bands.

    Each band is ``{"lower": <inclusive lower bound or None>, "upper":
    <exclusive upper bound or None>, "score": <value>}``. ``None`` means
    unbounded on that side (e.g. the first band omits "lower", the last
    band omits "upper"). Unlike :func:`band_score` (inclusive upper bound),
    this matches tables phrased as ">= X and < Y".
    """
    for band in bands:
        lower = band.get("lower")
        upper = band.get("upper")
        if (lower is None or value >= lower) and (upper is None or value < upper):
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


def call_probability_tier(
    call: bool,
    metric_value: Optional[float],
    threshold: Optional[float],
    high_margin_ratio: float = 0.02,
) -> str:
    """Bucket a call decision into 'high'/'medium'/'low' confidence.

    Works generically across colleges: each college reports whatever metric
    it compares to its own call threshold (e.g. NCS for IIMA, CAT overall
    percentile for IIMM) via ``call_metric_value``/``call_threshold`` on
    ``ScoreResult``; this function only needs the ratio between them, not
    what the metric actually is.

    'low' whenever the call itself is False. Otherwise (call is True):
    'high' if the candidate clears the threshold by at least
    ``high_margin_ratio`` (default 2%), 'medium' if they clear it by less
    than that, and 'medium' (rather than a misleading 'high' or 'low') if
    the margin can't be computed at all — e.g. a college whose call
    threshold hasn't been tuned yet defaults to 0, which isn't a
    meaningful denominator for a ratio.
    """
    if not call:
        return "low"
    if metric_value is None or threshold is None or threshold == 0:
        return "medium"
    margin_ratio = (metric_value - threshold) / threshold
    return "high" if margin_ratio >= high_margin_ratio else "medium"
