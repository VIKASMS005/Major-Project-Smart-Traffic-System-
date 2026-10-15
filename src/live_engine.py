"""
Live processing engine behind the web control centre.

One background thread plays the four lane camera videos in real time (looping),
runs YOLO detection on each lane every few frames, estimates density, checks for
incidents and drives the adaptive signal controller. The web server only reads
the latest state from here, so any number of browser tabs can watch at once.
"""
from __future__ import annotations

import glob
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2

from density_estimator import DEFAULT_THRESHOLDS, classify_density, count_by_type
from incident_detector import IncidentDetector
from signal_optimizer import DEFAULT_TIMING, SignalController
from vehicle_detector import VehicleDetector

DIRECTIONS = ["North", "East", "South", "West"]
LIGHT_BGR = {"GREEN": (60, 200, 60), "YELLOW": (0, 200, 255), "RED": (50, 50, 230),
             "BLOCKED": (50, 50, 230)}
DENSITY_BGR = {"Low": (60, 200, 60), "Medium": (0, 180, 255), "High": (50, 50, 230)}

DEFAULT_SETTINGS = {
    **DEFAULT_THRESHOLDS, **DEFAULT_TIMING,
    "stationary_frames": 90,      # incident after ~3.6 s stationary at 25 fps
    "detect_every": 5,            # run YOLO on every Nth frame of each lane
    "confidence": 0.3,
    "incident_hold_s": 8,         # keep lane blocked this long after the vehicle moves
}


class LaneFeed:
    """State of one camera / approach."""

    def __init__(self, idx: int, path: str):
        self.idx = idx
        self.id = f"lane{idx + 1}"
        self.name = f"Lane {idx + 1}"
        self.direction = DIRECTIONS[idx % 4]
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.start = time.time()
        self.frame = None
        self.jpeg = None
        self.detections: list[dict] = []
        self.flagged: set[int] = set()
        self.last_detect = -999
        self.count = 0
        self.by_type = {c: 0 for c in ("car", "motorcycle", "bus", "truck")}
        self.density = "Low"
        self.incident = IncidentDetector()
        self.incident_until = 0.0
        self.incident_info = None
        self.history = deque(maxlen=600)   # (unix time, count) once per second
        self.infer_ms = 0.0

    @property
    def vehicles_seen(self) -> int:
        return self.incident.vehicles_counted

    @property
    def class_seen(self) -> dict:
        return self.incident.confirmed

    def rewind(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self.start = time.time()
        self.last_detect = -999
        self.incident.reset()


class TrafficEngine:
    def __init__(self, video_dir: str, incident_dir: str, settings: dict | None = None):
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        paths = sorted(glob.glob(os.path.join(video_dir, "lane*.mp4")))[:4]
        if not paths:
            raise FileNotFoundError(f"No lane*.mp4 videos found in {video_dir}")
        self.lanes = [LaneFeed(i, p) for i, p in enumerate(paths)]
        self.detector = VehicleDetector(conf=self.settings["confidence"])
        self.controller = SignalController([l.id for l in self.lanes], self._timing())
        self.signal = {}
        self.incidents: list[dict] = []
        self.incident_dir = incident_dir
        os.makedirs(incident_dir, exist_ok=True)
        self.started = time.time()
        self.lock = threading.Lock()
        self._stop = False
        self._last_hist = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    # ----------------------------------------------------------- settings
    def _timing(self):
        return {k: self.settings[k] for k in DEFAULT_TIMING}

    def update_settings(self, new: dict):
        with self.lock:
            for k, v in new.items():
                if k in DEFAULT_SETTINGS:
                    v = float(v)
                    self.settings[k] = int(v) if v.is_integer() else v
            self.controller.timing.update(self._timing())
            self.detector.conf = self.settings["confidence"]
            for lane in self.lanes:
                lane.incident.stationary_frames = self.settings["stationary_frames"]

    # --------------------------------------------------------------- loop
    def start(self):
        self.thread.start()

    def _run(self):
        while not self._stop:
            for lane in self.lanes:
                self._step_lane(lane)
            self._step_signal()
            time.sleep(0.005)

    def _step_lane(self, lane: LaneFeed):
        # keep each video in real time: skip frames if detection made us fall behind
        target = int((time.time() - lane.start) * lane.fps)
        if lane.frame is not None and target <= lane.frame_idx:
            return
        frame = None
        while lane.frame_idx <= target:
            ok, f = lane.cap.read()
            if not ok:
                lane.rewind()
                return
            frame = f
            lane.frame_idx += 1
        lane.frame = frame
        s = self.settings
        if lane.frame_idx - lane.last_detect >= s["detect_every"]:
            t0 = time.perf_counter()
            dets = self.detector.detect(frame)
            lane.infer_ms = (time.perf_counter() - t0) * 1000
            lane.last_detect = lane.frame_idx
            flagged, new = lane.incident.update(dets, lane.frame_idx, frame.shape[0])
            lane.detections, lane.flagged = dets, flagged
            lane.count = len(dets)
            lane.by_type = count_by_type(dets)
            lane.density = classify_density(lane.count, s["low_max"], s["medium_max"])
            if lane.incident.active():
                lane.incident_until = time.time() + s["incident_hold_s"]
            for inc in new:
                self._log_incident(lane, frame, inc)
        lane.jpeg = self._render(lane, frame)

    def _log_incident(self, lane: LaneFeed, frame, inc):
        # the same stopped vehicle can be re-detected under a new track id after a
        # brief miss; don't report it again if it is where the last incident was
        x1, y1, x2, y2 = inc["bbox"]
        cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
        last = lane.incident_info
        if last and time.time() - last["_t"] < 20 and \
                abs(cx - last["_cx"]) + abs(cy - last["_cy"]) < 1.5 * size:
            return
        ts = datetime.now()
        snap = frame.copy()
        cv2.rectangle(snap, (x1, y1), (x2, y2), (40, 40, 235), 3)
        name = f"{ts:%Y%m%d_%H%M%S}_{lane.id}_{inc['track_id']}.jpg"
        cv2.imwrite(os.path.join(self.incident_dir, name), snap)
        record = {"id": len(self.incidents) + 1, "time": ts.strftime("%H:%M:%S"),
                  "date": ts.strftime("%d %b %Y"), "lane": lane.id, "lane_name": lane.name,
                  "direction": lane.direction, "vehicle": inc["vehicle"],
                  "duration_s": round(inc["stationary_frames"] / lane.fps, 1),
                  "type": "Stationary vehicle: possible accident or blockage",
                  "snapshot": name, "status": "Active", "_t": time.time(), "_cx": cx, "_cy": cy}
        if lane.incident_info and lane.incident_info["status"] == "Active":
            lane.incident_info["status"] = "Cleared"
        lane.incident_info = record
        self.incidents.insert(0, record)

    def _step_signal(self):
        now = time.time()
        counts = {l.id: l.count for l in self.lanes}
        blocked = {l.id for l in self.lanes if l.incident_until > now}
        for l in self.lanes:
            if l.incident_info and l.id not in blocked and l.incident_info["status"] == "Active":
                l.incident_info["status"] = "Cleared"
        with self.lock:
            self.signal = self.controller.update(counts, blocked)
        if now - self._last_hist >= 1:
            self._last_hist = now
            for l in self.lanes:
                l.history.append((round(now, 1), l.count))

    # ------------------------------------------------------------ drawing
    def _render(self, lane: LaneFeed, frame) -> bytes:
        out = self.detector.draw(frame, lane.detections, lane.flagged)
        h, w = out.shape[:2]
        light = self.signal.get("lights", {}).get(lane.id, "RED")
        # top bar
        bar = out.copy()
        cv2.rectangle(bar, (0, 0), (w, 34), (20, 20, 20), -1)
        cv2.addWeighted(bar, 0.6, out, 0.4, 0, dst=out)
        cv2.putText(out, f"CAM {lane.idx + 1:02d} | {lane.direction.upper()} APPROACH", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        stamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(out, stamp, (w - 190, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.circle(out, (w - 205, 18), 5, (40, 40, 235), -1)    # recording dot
        # bottom-left info chip
        chip = out.copy()
        cv2.rectangle(chip, (8, h - 42), (300, h - 8), (20, 20, 20), -1)
        cv2.addWeighted(chip, 0.6, out, 0.4, 0, dst=out)
        cv2.putText(out, f"Vehicles: {lane.count}", (16, h - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"Density: {lane.density}", (140, h - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    DENSITY_BGR[lane.density], 1, cv2.LINE_AA)
        # signal lamp
        cv2.circle(out, (w - 26, h - 26), 14, (20, 20, 20), -1)
        cv2.circle(out, (w - 26, h - 26), 10, LIGHT_BGR[light], -1)
        if light == "BLOCKED":
            cv2.rectangle(out, (0, 40), (w, 76), (40, 40, 200), -1)
            cv2.putText(out, "INCIDENT DETECTED - LANE BLOCKED", (12, 66), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpg.tobytes() if ok else None

    # -------------------------------------------------------------- state
    def snapshot(self) -> dict:
        now = time.time()
        with self.lock:
            sig = dict(self.signal)
        lanes = []
        for l in self.lanes:
            lanes.append({
                "id": l.id, "name": l.name, "direction": l.direction, "count": l.count,
                "density": l.density, "by_type": l.by_type, "light": sig.get("lights", {}).get(l.id, "RED"),
                "green_time": sig.get("plan", {}).get(l.id, 0),
                "blocked": l.incident_until > now, "vehicles_seen": l.vehicles_seen,
                "class_seen": l.class_seen, "infer_ms": round(l.infer_ms, 1),
                "incident": l.incident_info if l.incident_until > now else None,
            })
        total_now = sum(l["count"] for l in lanes)
        return {
            "time": datetime.now().strftime("%H:%M:%S"), "uptime_s": int(now - self.started),
            "signal": sig, "lanes": lanes, "total_vehicles_now": total_now,
            "vehicles_seen": sum(l["vehicles_seen"] for l in lanes),
            "active_incidents": sum(1 for l in lanes if l["blocked"]),
            "incidents_total": len(self.incidents),
        }

    def analytics(self) -> dict:
        hist = {l.id: list(l.history) for l in self.lanes}
        with self.lock:
            sig_hist = list(self.controller.history)
        return {
            "lanes": [{"id": l.id, "name": l.name, "direction": l.direction} for l in self.lanes],
            "history": hist,
            "class_seen": {c: sum(l.class_seen[c] for l in self.lanes) for c in ("car", "motorcycle", "bus", "truck")},
            "vehicles_seen": {l.id: l.vehicles_seen for l in self.lanes},
            "signal_history": [{"time": datetime.fromtimestamp(t).strftime("%H:%M:%S"), "lane": lane,
                                "green": g, "count": c} for t, lane, g, c in reversed(sig_hist)],
        }
