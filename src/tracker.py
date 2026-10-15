"""
Lightweight multi-object tracker (IoU + centre distance, greedy matching).

Gives every detected vehicle a stable id across frames so the system can
follow its path (for lane discovery), measure its speed and count it once.
No extra dependencies beyond NumPy.
"""
from __future__ import annotations

import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    __slots__ = ("id", "bbox", "cls", "hits", "missed", "history", "velocity", "lane", "counted")

    def __init__(self, tid, det, frame_idx):
        self.id = tid
        self.bbox = det["bbox"]
        self.cls = det["class"]
        self.hits = 1
        self.missed = 0
        x1, y1, x2, y2 = det["bbox"]
        self.history = [(frame_idx, (x1 + x2) / 2, y2, x2 - x1)]   # ground point + box width
        self.velocity = np.zeros(2)
        self.lane = None
        self.counted = False

    @property
    def ground(self):
        return self.history[-1][1], self.history[-1][2]


class Tracker:
    def __init__(self, max_missed=6, min_iou=0.15, max_jump=1.5):
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.max_jump = max_jump      # max centre jump between updates, in box widths
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    def update(self, detections, frame_idx):
        """Match detections to tracks. Sets det['track'] on every detection."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            tx1, ty1, tx2, ty2 = t.bbox
            tw = max(tx2 - tx1, 1)
            # predicted position using the last velocity
            gap = frame_idx - t.history[-1][0]
            px, py = t.velocity * gap
            pb = (tx1 + px, ty1 + py, tx2 + px, ty2 + py)
            for di, d in enumerate(detections):
                iou = _iou(pb, d["bbox"])
                dx = (d["bbox"][0] + d["bbox"][2]) / 2 - (pb[0] + pb[2]) / 2
                dy = d["bbox"][3] - pb[3]
                dist = (dx * dx + dy * dy) ** 0.5 / tw
                if iou >= self.min_iou or dist < self.max_jump * 0.5:
                    pairs.append((iou - 0.2 * dist, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], detections[di]
            x1, y1, x2, y2 = d["bbox"]
            last_f, lx, ly, _ = t.history[-1]
            gx, gy = (x1 + x2) / 2, y2
            if frame_idx > last_f:
                v = np.array([gx - lx, gy - ly]) / (frame_idx - last_f)
                t.velocity = 0.6 * t.velocity + 0.4 * v if t.hits > 1 else v
            t.bbox = d["bbox"]
            t.hits += 1
            t.missed = 0
            if d["confidence"] > 0.4:
                t.cls = d["class"]
            t.history.append((frame_idx, gx, gy, x2 - x1))
            d["track"] = t
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                t = Track(self._next, d, frame_idx)
                self._next += 1
                self.tracks.append(t)
                d["track"] = t
        alive = []
        for t in self.tracks:
            (alive if t.missed <= self.max_missed else self.finished).append(t)
        self.tracks = alive
        return self.tracks

    def trajectories(self):
        """All tracks seen so far as id -> [(x, y, width), ...]."""
        out = {}
        for t in self.finished + self.tracks:
            out[t.id] = [(x, y, w) for _, x, y, w in t.history]
        return out
