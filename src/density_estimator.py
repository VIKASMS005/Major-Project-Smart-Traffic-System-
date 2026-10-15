"""
Traffic density estimation (report section 4.6).

The number of vehicles visible on a lane is turned into a congestion level.
Thresholds are configurable because a 2-lane road and a 6-lane highway saturate
at very different counts.
"""
from __future__ import annotations

from collections import Counter

DEFAULT_THRESHOLDS = {"low_max": 5, "medium_max": 12}   # <=5 Low, <=12 Medium, 13+ High
LEVELS = ("Low", "Medium", "High")


def classify_density(count: int, low_max: int = DEFAULT_THRESHOLDS["low_max"],
                     medium_max: int = DEFAULT_THRESHOLDS["medium_max"]) -> str:
    """Return 'Low', 'Medium' or 'High' for a vehicle count."""
    if count <= low_max:
        return "Low"
    if count <= medium_max:
        return "Medium"
    return "High"


def count_by_type(detections: list[dict]) -> dict[str, int]:
    """Count detections per vehicle class, always returning all four classes."""
    counts = Counter(d["class"] for d in detections)
    return {c: counts.get(c, 0) for c in ("car", "motorcycle", "bus", "truck")}
