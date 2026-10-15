"""
Automatic lane and direction discovery (report section 4.6, traffic analysis).

No lanes are drawn by hand. The system watches the first seconds of the
uploaded video, tracks every vehicle, and learns the road layout from where
vehicles actually drive:

1. Trajectories: the ground-contact point (bottom-centre of the box) of each
   tracked vehicle over time.
2. Traffic streams (approaches): trajectories are grouped by direction of
   motion. Each group is one approach of the junction / one carriageway.
3. Lanes: inside a stream, trajectories are clustered by their sideways
   position. The distance between two trajectories is measured in
   "vehicle widths" at each depth of the image, so perspective (far vehicles
   look smaller) does not break the clustering.
4. Each lane gets a centre line and a width profile, which become the lane
   polygon drawn on the video and used to assign every detection to a lane.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

N_BINS = 14                 # depth bins along each stream
MIN_TRACK_POINTS = 5
SAME_LANE_DIST = 0.85       # max separation (in vehicle widths) for one lane
STREAM_ANGLE_DEG = 55       # trajectories within this angle belong to one stream (allows for perspective)

LANE_COLORS = [(255, 170, 60), (90, 220, 120), (60, 190, 255), (230, 120, 255),
               (80, 240, 240), (255, 120, 140), (180, 255, 90), (255, 220, 90)]


@dataclass
class Lane:
    id: str
    stream: int
    index_in_stream: int
    centre: np.ndarray          # (k, 2) image points, ordered along travel direction
    half_width: np.ndarray      # (k,) pixels
    color: tuple
    polygon: np.ndarray = field(default=None)
    area: float = 0.0

    def build_polygon(self, direction: np.ndarray):
        n = np.array([-direction[1], direction[0]])
        left = self.centre - n * self.half_width[:, None]
        right = self.centre + n * self.half_width[:, None]
        self.polygon = np.vstack([left, right[::-1]]).astype(np.int32)
        self.area = float(cv2.contourArea(self.polygon))


@dataclass
class Stream:
    index: int
    direction: np.ndarray       # unit vector of motion in the image
    name: str                   # e.g. "North approach"
    compass: str                # North / South / East / West (where traffic comes from)
    heading: str                # human description of motion in the image
    lanes: list = field(default_factory=list)
    track_count: int = 0


def _compass(d: np.ndarray) -> tuple[str, str]:
    """Name an approach from its motion in the image.

    Moving down the image = coming towards the camera = treated as traffic
    arriving from the North; moving up = from the South; moving right = from
    the West; moving left = from the East.
    """
    dx, dy = d
    if abs(dy) >= abs(dx):
        return ("North", "towards camera") if dy > 0 else ("South", "away from camera")
    return ("West", "left to right") if dx > 0 else ("East", "right to left")


def _cluster_angles(angles: np.ndarray, weights: np.ndarray) -> list[list[int]]:
    """Greedy circular clustering of motion angles."""
    order = np.argsort(angles)
    clusters: list[list[int]] = []
    centres: list[float] = []
    thr = math.radians(STREAM_ANGLE_DEG)
    for i in order:
        a = angles[i]
        best, bd = None, thr
        for k, c in enumerate(centres):
            d = abs((a - c + math.pi) % (2 * math.pi) - math.pi)
            if d < bd:
                best, bd = k, d
        if best is None:
            clusters.append([i])
            centres.append(a)
        else:
            clusters[best].append(i)
            idx = clusters[best]
            centres[best] = math.atan2(np.sum(weights[idx] * np.sin(angles[idx])),
                                       np.sum(weights[idx] * np.cos(angles[idx])))
    return clusters


def _agglomerate(dist: np.ndarray, thr: float) -> list[list[int]]:
    """Average-linkage agglomerative clustering on a distance matrix."""
    groups = [[i] for i in range(len(dist))]
    D = dist.copy()
    np.fill_diagonal(D, np.inf)
    while len(groups) > 1:
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if D[i, j] > thr:
            break
        ni, nj = len(groups[i]), len(groups[j])
        newrow = (D[i] * ni + D[j] * nj) / (ni + nj)
        groups[i] = groups[i] + groups[j]
        D[i, :] = newrow
        D[:, i] = newrow
        D[i, i] = np.inf
        D = np.delete(np.delete(D, j, 0), j, 1)
        groups.pop(j)
    return groups


def discover_lanes(trajectories: dict[int, list[tuple[float, float, float]]],
                   frame_shape: tuple[int, int]) -> list[Stream]:
    """trajectories: track id -> list of (x, y, box_width) ground points in time order."""
    h, w = frame_shape[:2]
    tracks = []
    for pts in trajectories.values():
        if len(pts) < MIN_TRACK_POINTS:
            continue
        p = np.array(pts, dtype=float)
        disp = p[-1, :2] - p[0, :2]
        dist = np.linalg.norm(disp)
        if dist < max(30.0, 1.2 * np.median(p[:, 2])):
            continue                                   # parked / barely moving
        tracks.append((p, disp / dist, dist))
    if len(tracks) < 3:
        return []

    angles = np.array([math.atan2(u[1], u[0]) for _, u, _ in tracks])
    weights = np.array([d for _, _, d in tracks])
    streams: list[Stream] = []
    min_tracks = max(3, int(0.06 * len(tracks)))
    for idx in _cluster_angles(angles, weights):
        if len(idx) < min_tracks:
            continue
        d = np.sum([tracks[i][1] * tracks[i][2] for i in idx], axis=0)
        d /= np.linalg.norm(d)
        compass, heading = _compass(d)
        streams.append(Stream(index=len(streams), direction=d, name="", compass=compass,
                              heading=heading, track_count=len(idx)))
        streams[-1]._members = idx

    # unique approach names
    seen: dict[str, int] = {}
    for s in sorted(streams, key=lambda s: -s.track_count):
        seen[s.compass] = seen.get(s.compass, 0) + 1
        s.name = f"{s.compass} approach" + ("" if seen[s.compass] == 1 else f" {seen[s.compass]}")

    lane_no = 0
    for s in streams:
        d = s.direction
        n = np.array([-d[1], d[0]])
        members = [tracks[i][0] for i in s._members]
        allp = np.vstack(members)
        S = allp[:, :2] @ d
        s_min, s_max = np.percentile(S, 1), np.percentile(S, 99)
        edges = np.linspace(s_min, s_max, N_BINS + 1)
        # median vehicle width per depth bin (handles perspective)
        b_all = np.clip(np.digitize(S, edges) - 1, 0, N_BINS - 1)
        W = np.array([np.median(allp[b_all == b, 2]) if np.any(b_all == b) else np.nan
                      for b in range(N_BINS)])
        W = _fill_nan(W)
        # per-track lateral profile
        prof = np.full((len(members), N_BINS), np.nan)
        for k, p in enumerate(members):
            sb = np.clip(np.digitize(p[:, :2] @ d, edges) - 1, 0, N_BINS - 1)
            t = p[:, :2] @ n
            for b in np.unique(sb):
                prof[k, b] = np.mean(t[sb == b])
        # pairwise distance in vehicle widths
        m = len(members)
        D = np.zeros((m, m))
        for a in range(m):
            for b in range(a + 1, m):
                both = ~np.isnan(prof[a]) & ~np.isnan(prof[b])
                if both.sum() >= 1:
                    dd = np.median(np.abs(prof[a, both] - prof[b, both]) / W[both])
                else:   # no overlap in depth: compare nearest observed bins
                    ia, ib = np.where(~np.isnan(prof[a]))[0], np.where(~np.isnan(prof[b]))[0]
                    ka = ia[np.argmin(np.abs(ia[:, None] - ib[None, :]).min(1))]
                    kb = ib[np.argmin(np.abs(ib[:, None] - ia[None, :]).min(1))]
                    dd = abs(prof[a, ka] - prof[b, kb]) / ((W[ka] + W[kb]) / 2) + 0.3
                D[a, b] = D[b, a] = dd
        groups = _agglomerate(D, SAME_LANE_DIST)
        min_lane = max(2, int(0.05 * m))
        groups = [g for g in groups if len(g) >= min_lane] or [max(groups, key=len)]

        centres = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for g in groups:
                centres.append(np.nanmedian(prof[g], axis=0))
        # order lanes left-to-right relative to the direction of travel
        order = np.argsort([np.nanmean(c) for c in centres])
        mids = (edges[:-1] + edges[1:]) / 2
        lane_lines = []
        for rank, gi in enumerate(order):
            c = centres[gi]
            ok = ~np.isnan(c)
            # observed depth range of this lane, slightly extended
            lo, hi = np.where(ok)[0][[0, -1]]
            lo, hi = max(0, lo - 1), min(N_BINS - 1, hi + 1)
            bins = np.arange(lo, hi + 1)
            cc = np.interp(bins, np.where(ok)[0], c[ok])
            # smooth the centre line: lanes are straight or gently curved
            deg = 2 if ok.sum() >= 5 else 1 if ok.sum() >= 2 else 0
            cc = np.polyval(np.polyfit(np.where(ok)[0], c[ok], deg), bins)
            lane_lines.append((rank, gi, bins, cc))
        for rank, gi, bins, cc in lane_lines:
            half = np.polyval(np.polyfit(np.arange(N_BINS), W, 1), bins) * 0.85
            half = np.maximum(half, 4)
            # don't overlap neighbouring lanes
            for orank, ogi, obins, occ in lane_lines:
                if orank == rank:
                    continue
                common = np.intersect1d(bins, obins)
                for b in common:
                    gap = abs(cc[bins == b][0] - occ[obins == b][0]) / 2
                    half[bins == b] = np.minimum(half[bins == b], max(gap, 4))
            pts = np.outer(mids[bins], d) + np.outer(cc, n)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            lane_no += 1
            lane = Lane(id=f"L{lane_no}", stream=s.index, index_in_stream=rank + 1, centre=pts,
                        half_width=half, color=LANE_COLORS[(lane_no - 1) % len(LANE_COLORS)])
            lane.build_polygon(d)
            s.lanes.append(lane)
        del s._members
    return [s for s in streams if s.lanes]


def fallback_layout(frame_shape: tuple[int, int]) -> list[Stream]:
    """Used when too few vehicles moved during learning: the whole road is one lane."""
    h, w = frame_shape[:2]
    d = np.array([0.0, 1.0])
    s = Stream(index=0, direction=d, name="Main road", compass="North", heading="whole frame")
    centre = np.array([[w / 2, 0.0], [w / 2, h - 1.0]])
    lane = Lane(id="L1", stream=0, index_in_stream=1, centre=centre,
                half_width=np.array([w / 2, w / 2]), color=LANE_COLORS[0])
    lane.build_polygon(d)
    s.lanes.append(lane)
    return [s]


def _fill_nan(a: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(a)
    if not ok.any():
        return np.full_like(a, 40.0)
    return np.interp(np.arange(len(a)), np.where(ok)[0], a[ok])


def assign_lane(point: tuple[float, float], velocity: np.ndarray | None,
                streams: list[Stream]) -> Lane | None:
    """Lane a ground point belongs to, or None if it is outside every lane.

    When the vehicle's motion is known, only lanes whose traffic moves the same
    way are considered, which keeps crossing streams at a junction apart.
    """
    best, best_score = None, 1.0
    p = np.asarray(point, dtype=float)
    for s in streams:
        if velocity is not None and s.heading != "whole frame" and np.linalg.norm(velocity) > 2:
            if float(np.dot(velocity / np.linalg.norm(velocity), s.direction)) < 0.3:
                continue
        for lane in s.lanes:
            if cv2.pointPolygonTest(lane.polygon, (float(p[0]), float(p[1])), False) >= 0:
                score = 0.0
            else:
                dist = -cv2.pointPolygonTest(lane.polygon, (float(p[0]), float(p[1])), True)
                score = dist / (np.median(lane.half_width) + 1e-6)
            if score < best_score:
                best, best_score = lane, score
            if score == 0.0:
                return lane
    return best


def draw_lanes(img: np.ndarray, streams: list[Stream], stats: dict | None = None) -> np.ndarray:
    """Semi-transparent lane bands, centre lines, direction arrows and labels."""
    overlay = img.copy()
    for s in streams:
        for lane in s.lanes:
            cv2.fillPoly(overlay, [lane.polygon], lane.color)
    out = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    for s in streams:
        for lane in s.lanes:
            cv2.polylines(out, [lane.polygon], True, lane.color, 1, cv2.LINE_AA)
            pts = lane.centre.astype(int)
            for a, b in zip(pts[:-1:2], pts[1::2]):
                cv2.line(out, tuple(a), tuple(b), lane.color, 1, cv2.LINE_AA)
            if len(pts) >= 2:
                cv2.arrowedLine(out, tuple(pts[-2]), tuple(pts[-1]), lane.color, 2, cv2.LINE_AA, tipLength=0.6)
            # label near the end of the lane closest to the camera (largest y)
            k = int(np.argmax(pts[:, 1]))
            x, y = pts[k]
            txt = lane.id
            if stats and lane.id in stats:
                txt += f" | {stats[lane.id]}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = int(np.clip(x - tw // 2, 2, img.shape[1] - tw - 6))
            y = int(np.clip(y - 8, th + 4, img.shape[0] - 4))
            cv2.rectangle(out, (x - 3, y - th - 4), (x + tw + 3, y + 3), lane.color, -1)
            cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out
