"""
Traffic density estimation (report section 4.6).

Mixed Indian traffic cannot be measured by a plain vehicle count: a bus takes
the road space of several cars and a scooter about half of one. Each vehicle is
therefore converted to Passenger Car Units (PCU, IRC:106-1990) and the density
of a lane is its PCU load. Thresholds are configurable because a 2-lane road
and a 6-lane highway saturate at very different values.
"""
from __future__ import annotations

from collections import Counter

from vehicle_detector import CATEGORIES, PCU

DEFAULT_THRESHOLDS = {"low_max": 5, "medium_max": 12}   # PCU: <=5 Low, <=12 Medium, above High
LEVELS = ("Low", "Medium", "High")


def classify_density(load: float, low_max: float = DEFAULT_THRESHOLDS["low_max"],
                     medium_max: float = DEFAULT_THRESHOLDS["medium_max"]) -> str:
    """Return 'Low', 'Medium' or 'High' for a PCU load (or a plain count)."""
    if load <= low_max:
        return "Low"
    if load <= medium_max:
        return "Medium"
    return "High"


def count_by_type(detections: list[dict]) -> dict[str, int]:
    """Count detections per category, always returning all four categories."""
    counts = Counter(d["class"] for d in detections)
    return {c: counts.get(c, 0) for c in CATEGORIES}


def pcu_load(by_type: dict[str, float]) -> float:
    """Passenger Car Units for a per-category count."""
    return round(sum(PCU[c] * n for c, n in by_type.items()), 1)
