"""
Incident / blockage detection (project objective 3).

A heuristic, not a trained model. Vehicles are tracked from one detection to
the next by nearest-neighbour matching of their centre points. If a vehicle's
centre stays inside a small radius for longer than `stationary_frames`
(90 frames ~ 3 seconds at 30 fps), it is flagged as a possible accident,
breakdown or blockage.

Two refinements over a plain pixel radius keep false alarms down:
  * the radius scales with the vehicle's box height, so a car far away (which
    moves only a few pixels per second) is not mistaken for a stopped one;
  * vehicles smaller than `min_box_frac` of the frame height are ignored,
    because they are too far away to judge their motion reliably.
"""
from __future__ import annotations

import math


class IncidentDetector:
    def __init__(self, stationary_frames: int = 90, radius_frac: float = 0.15,
                 min_box_frac: float = 0.08, match_dist_frac: float = 2.0, max_missed: int = 25):
        self.stationary_frames = stationary_frames
        self.radius_frac = radius_frac
        self.min_box_frac = min_box_frac
        self.match_dist_frac = match_dist_frac
        self.max_missed = max_missed
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.flagged_ids: set[int] = set()
        # vehicle counting: a track counts once it has been seen `confirm_hits` times,
        # which filters out one-off false detections
        self.confirm_hits = 3
        self.confirmed = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

    def reset(self):
        """Forget tracks (e.g. when a video loops); vehicle totals are kept."""
        self.tracks.clear()
        self.flagged_ids.clear()

    @property
    def vehicles_counted(self) -> int:
        return sum(self.confirmed.values())

    def update(self, detections: list[dict], frame_idx: int, frame_h: int) -> tuple[set[int], list[dict]]:
        """Match detections to tracks.

        Returns (indices of detections that are currently stationary,
                 list of NEW incidents raised on this call).
        """
        unmatched = set(self.tracks)
        stationary_idx, new_incidents = set(), []
        for i, d in enumerate(detections):
            cx, cy = d["center"]
            x1, y1, x2, y2 = d["bbox"]
            box_h = y2 - y1
            best, best_dist = None, box_h * self.match_dist_frac + 5
            for tid in unmatched:
                tx, ty = self.tracks[tid]["last"]
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_dist:
                    best, best_dist = tid, dist
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"anchor": (cx, cy), "since": frame_idx, "last": (cx, cy),
                                     "seen": frame_idx, "class": d["class"], "hits": 0}
            else:
                unmatched.discard(best)
            tr = self.tracks[best]
            tr["last"], tr["seen"], tr["class"] = (cx, cy), frame_idx, d["class"]
            tr["hits"] += 1
            if tr["hits"] == self.confirm_hits:
                self.confirmed[d["class"]] += 1
            ax, ay = tr["anchor"]
            if math.hypot(cx - ax, cy - ay) > max(4.0, self.radius_frac * box_h):
                tr["anchor"], tr["since"] = (cx, cy), frame_idx      # moved: restart the clock
            elif (box_h >= self.min_box_frac * frame_h
                  and frame_idx - tr["since"] >= self.stationary_frames):
                stationary_idx.add(i)
                if best not in self.flagged_ids:
                    self.flagged_ids.add(best)
                    new_incidents.append({"track_id": best, "vehicle": d["class"],
                                          "bbox": d["bbox"], "frame": frame_idx,
                                          "stationary_frames": frame_idx - tr["since"]})
        for tid in list(unmatched):
            if frame_idx - self.tracks[tid]["seen"] > self.max_missed:
                self.tracks.pop(tid)
                self.flagged_ids.discard(tid)
        return stationary_idx, new_incidents

    def active(self) -> bool:
        """True while at least one flagged vehicle is still being tracked."""
        return any(tid in self.tracks for tid in self.flagged_ids)
