"""
Adaptive signal control (report section 4.7).

Rule-based adaptive timing: each lane gets a green time proportional to the
number of vehicles waiting on it,

    green_time = base_time + vehicle_count * per_vehicle_time

clamped to [min_green, max_green] so an empty lane still gets a short green
and a jammed lane cannot starve the others.

`SignalController` runs the timing live: it cycles through the lanes in order,
gives each its green, then yellow, then an all-red clearance, and skips any lane
that is blocked by an incident.
"""
from __future__ import annotations

import time

DEFAULT_TIMING = {"base_time": 10, "per_vehicle_time": 2, "min_green": 10,
                  "max_green": 60, "yellow_time": 3, "all_red_time": 1}


def compute_green_time(vehicle_count: int, base_time: float = 10, per_vehicle_time: float = 2,
                       min_green: float = 10, max_green: float = 60) -> int:
    """Green time in seconds for one lane."""
    green = base_time + vehicle_count * per_vehicle_time
    return int(max(min_green, min(max_green, green)))


def compute_signal_plan(lane_counts: dict[str, int], **timing) -> dict[str, int]:
    """Map lane name -> allocated green time for every lane."""
    t = {**DEFAULT_TIMING, **timing}
    return {lane: compute_green_time(c, t["base_time"], t["per_vehicle_time"],
                                     t["min_green"], t["max_green"])
            for lane, c in lane_counts.items()}


class SignalController:
    """Live signal state machine: GREEN -> YELLOW -> ALL RED -> next lane.

    The green time of a lane is decided at the moment its turn starts, from the
    latest vehicle count, so the plan keeps adapting to traffic.
    """

    def __init__(self, lanes: list[str], timing: dict | None = None):
        self.lanes = lanes
        self.timing = {**DEFAULT_TIMING, **(timing or {})}
        self.index = -1
        self.phase = "ALL_RED"
        self.phase_end = time.time()
        self.green_for = {}
        self.cycle = 0
        self.history = []          # list of (timestamp, lane, green seconds, count)

    def _next_lane(self, blocked: set[str]) -> int | None:
        for step in range(1, len(self.lanes) + 1):
            i = (self.index + step) % len(self.lanes)
            if self.lanes[i] not in blocked:
                return i
        return None

    def update(self, counts: dict[str, int], blocked: set[str]) -> dict:
        """Advance the state machine; call this a few times per second."""
        now = time.time()
        t = self.timing
        if self.phase == "GREEN" and self.lanes[self.index] in blocked:
            # incident appeared on the lane that has green: end its green early
            self.phase, self.phase_end = "YELLOW", now + t["yellow_time"]
        if now >= self.phase_end:
            if self.phase == "GREEN":
                self.phase, self.phase_end = "YELLOW", now + t["yellow_time"]
            elif self.phase == "YELLOW":
                self.phase, self.phase_end = "ALL_RED", now + t["all_red_time"]
            else:
                nxt = self._next_lane(blocked)
                if nxt is None:                       # every lane blocked
                    self.phase_end = now + 1
                else:
                    if nxt <= self.index:
                        self.cycle += 1
                    self.index = nxt
                    lane = self.lanes[nxt]
                    green = compute_green_time(counts.get(lane, 0), t["base_time"],
                                               t["per_vehicle_time"], t["min_green"], t["max_green"])
                    self.green_for[lane] = green
                    self.history.append((now, lane, green, counts.get(lane, 0)))
                    self.history = self.history[-500:]
                    self.phase, self.phase_end = "GREEN", now + green
        return self.state(counts, blocked)

    def state(self, counts: dict[str, int], blocked: set[str]) -> dict:
        now = time.time()
        active = self.lanes[self.index] if self.index >= 0 else None
        lights = {}
        for lane in self.lanes:
            if lane in blocked:
                lights[lane] = "BLOCKED"
            elif lane == active and self.phase in ("GREEN", "YELLOW"):
                lights[lane] = self.phase
            else:
                lights[lane] = "RED"
        plan = compute_signal_plan(counts, **{k: self.timing[k] for k in
                                              ("base_time", "per_vehicle_time", "min_green", "max_green")})
        return {"active_lane": active, "phase": self.phase,
                "remaining": max(0.0, round(self.phase_end - now, 1)),
                "lights": lights, "plan": plan, "cycle": self.cycle,
                "upcoming": self._upcoming(blocked)}

    def _upcoming(self, blocked: set[str]) -> list[str]:
        order, i = [], self.index
        for _ in range(len(self.lanes)):
            i = (i + 1) % len(self.lanes)
            if self.lanes[i] not in blocked:
                order.append(self.lanes[i])
        return order
