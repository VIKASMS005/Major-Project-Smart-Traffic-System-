"""
Live processing engine behind the web control centre.

The user uploads one traffic video. The engine then:

1. Learns the road layout: it tracks every vehicle over the first seconds of
   the video and discovers the traffic directions (approaches) and the lanes
   inside each one (lane_detector.py).
2. Runs live: plays the video in real time (looping), detects and classifies
   vehicles, assigns each one to a lane, measures density per lane in PCU,
   counts traffic flow, watches for stopped vehicles and drives the adaptive
   signal controller.

Signal groups: when the video shows two or more traffic directions, each
direction (approach) gets its own green phase. When it shows a single
direction, every lane is its own signal group.

The web server only reads the latest state from here, so any number of
browser tabs can watch at once.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from density_estimator import DEFAULT_THRESHOLDS, classify_density, pcu_load
from lane_detector import assign_lane, discover_lanes, draw_lanes, fallback_layout
from signal_optimizer import DEFAULT_TIMING, SignalController
from simulation import compare_controllers
from tracker import Tracker
from vehicle_detector import CATEGORIES, CATEGORY_NAMES, VehicleDetector

LIGHT_BGR = {"GREEN": (60, 200, 60), "YELLOW": (0, 200, 255), "RED": (50, 50, 230),
             "BLOCKED": (50, 50, 230)}

DEFAULT_SETTINGS = {
    **DEFAULT_THRESHOLDS, **DEFAULT_TIMING,
    "learn_seconds": 30,         # video time used to learn lanes
    "stationary_s": 6,           # incident after a vehicle is stopped this long
    "detect_every": 2,           # run detection on every Nth frame (when the CPU keeps up)
    "confidence": 0.3,
    "incident_hold_s": 8,        # keep the group blocked this long after the vehicle moves
}


def _zeros():
    return {c: 0 for c in CATEGORIES}


class LaneState:
    def __init__(self, lane, stream):
        self.lane = lane
        self.stream = stream
        self.now = _zeros()          # vehicles in the lane right now, per category
        self.counted = _zeros()      # unique vehicles that used the lane
        self.events = deque()        # video-times at which vehicles were counted
        self.speeds = deque(maxlen=60)

    @property
    def count(self):
        return sum(self.now.values())

    @property
    def pcu(self):
        return pcu_load(self.now)


class TrafficEngine:
    def __init__(self, incident_dir: str, settings: dict | None = None, realtime: bool = True):
        """realtime=False processes every frame as fast as possible and stops at the end
        of the video (used by main.py); on_frame(img) then receives each annotated frame."""
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.realtime = realtime
        self.on_frame = None
        self.detector = VehicleDetector(conf=self.settings["confidence"])
        self.incident_dir = incident_dir
        os.makedirs(incident_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.thread = None
        self._gen = 0
        self._reset(None, None)

    # ------------------------------------------------------------- loading
    def _reset(self, path, name):
        self.video_path, self.video_name = path, name
        self.status = "empty" if path is None else "learning"
        self.message = ""
        self.progress = 0.0
        self.jpeg = None
        self.layout_jpeg = None
        self.streams = []
        self.lanes: dict[str, LaneState] = {}
        self.groups: list[dict] = []
        self.lane_group: dict[str, str] = {}
        self.group_mode = None
        self.controller = None
        self.signal = {}
        self.incidents: list[dict] = []
        self.blocked_until: dict[str, float] = {}
        self.active_incident: dict[str, dict] = {}
        self.history = deque(maxlen=900)     # one row per second
        self.started = time.time()
        self.loops = 0
        self.fps = 25.0
        self.frame_size = (0, 0)
        self.total_frames = 0
        self.video_time = 0.0
        self.infer_ms = 0.0
        self.det_fps = 0.0
        self.vehicles_counted = 0
        self.tracker = Tracker()
        self._perf = None
        self._perf_t = 0.0

    def load(self, path: str, name: str | None = None):
        """Start analysing a new video (replaces the current one)."""
        with self.lock:
            self._gen += 1
            gen = self._gen
            self._reset(path, name or os.path.basename(path))
        self.thread = threading.Thread(target=self._run, args=(gen,), daemon=True)
        self.thread.start()

    def relearn(self):
        if self.video_path:
            self.load(self.video_path, self.video_name)

    # ------------------------------------------------------------ settings
    def _timing(self):
        return {k: self.settings[k] for k in DEFAULT_TIMING}

    def update_settings(self, new: dict):
        with self.lock:
            for k, v in new.items():
                if k in DEFAULT_SETTINGS:
                    v = float(v)
                    self.settings[k] = int(v) if v.is_integer() else v
            if self.controller:
                self.controller.timing.update(self._timing())
            self.detector.conf = self.settings["confidence"]
            self._perf = None

    # ---------------------------------------------------------------- run
    def _alive(self, gen):
        return gen == self._gen

    def _run(self, gen):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("This file could not be opened as a video.")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if self.fps > 120 or self.fps < 1:
                self.fps = 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._learn(cap, gen)
            if self._alive(gen):
                self._live(cap, gen)
        except Exception as e:      # shown on the page instead of crashing the server
            if self._alive(gen):
                self.status, self.message = "error", str(e)
                import traceback
                traceback.print_exc()

    # -- phase 1: learn the road layout ----------------------------------
    def _learn(self, cap, gen):
        s = self.settings
        n_frames = int(s["learn_seconds"] * self.fps)
        if self.total_frames > 0:
            n_frames = min(n_frames, self.total_frames)
        stride = max(1, round(self.fps / 6))           # about 6 detections per video second
        tracker = Tracker(max_missed=4)
        frame, f = None, 0
        while f < n_frames and self._alive(gen):
            ok, fr = cap.read()
            if not ok:
                break
            self.frame_size = fr.shape[:2]
            fr = _prep(fr)
            frame = fr
            if f % stride == 0:
                dets = self.detector.detect(fr)
                tracker.update(dets, f)
                self.progress = round(100 * f / max(n_frames, 1), 1)
                self.message = (f"Learning the road layout from vehicle paths: "
                                f"{len(tracker.trajectories())} vehicles tracked")
                self.jpeg = self._encode(self._render_learning(fr, dets, tracker))
            f += 1
        if frame is None:
            raise RuntimeError("The video has no readable frames.")
        if not self._alive(gen):
            return
        streams = discover_lanes(tracker.trajectories(), frame.shape)
        self.message = ""
        if not streams:
            streams = fallback_layout(frame.shape)
            self.message = "Too few moving vehicles to separate lanes, so the road is treated as one lane."
        self._setup_layout(streams, frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _setup_layout(self, streams, frame):
        self.streams = streams
        self.lanes = {l.id: LaneState(l, s) for s in streams for l in s.lanes}
        if len(streams) >= 2:
            self.group_mode = "approach"
            self.groups = [{"id": f"G{k + 1}", "name": s.name, "compass": s.compass,
                            "heading": s.heading, "lanes": [l.id for l in s.lanes]}
                           for k, s in enumerate(streams)]
        else:
            self.group_mode = "lane"
            s = streams[0]
            self.groups = [{"id": f"G{i + 1}", "name": f"Lane {l.id[1:]}", "compass": s.compass,
                            "heading": s.heading, "lanes": [l.id]} for i, l in enumerate(s.lanes)]
        self.lane_group = {lid: g["id"] for g in self.groups for lid in g["lanes"]}
        self.controller = SignalController([g["id"] for g in self.groups], self._timing())
        self.layout_jpeg = self._encode(draw_lanes(frame, streams))
        self.status = "live"
        self.progress = 100.0
        self.started = time.time()

    # -- phase 2: live analysis ------------------------------------------
    def _live(self, cap, gen):
        start = time.time()
        frame_idx = 0
        last_detect = -999
        dets, flagged = [], set()
        last_hist = -1.0
        det_times = deque(maxlen=20)
        while self._alive(gen):
            target = int((time.time() - start) * self.fps) if self.realtime else frame_idx
            if target <= frame_idx and self.realtime:
                self._step_signal()
                time.sleep(0.005)
                continue
            frame = None
            while frame_idx <= target:
                ok, fr = cap.read()
                if not ok:
                    break
                frame = fr
                frame_idx += 1
            if frame is None and not self.realtime:   # batch mode: done
                self.status = "done"
                return
            if frame is None:                       # end of video: loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.loops += 1
                self.tracker = Tracker()
                start, frame_idx, last_detect = time.time(), 0, -999
                continue
            frame = _prep(frame)
            length = self.total_frames / self.fps if self.total_frames else 0
            self.video_time = self.loops * length + frame_idx / self.fps
            if frame_idx - last_detect >= self.settings["detect_every"]:
                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                self.infer_ms = (time.perf_counter() - t0) * 1000
                det_times.append(time.time())
                if len(det_times) > 1:
                    self.det_fps = (len(det_times) - 1) / max(det_times[-1] - det_times[0], 1e-6)
                last_detect = frame_idx
                self.tracker.update(dets, frame_idx)
                flagged = self._analyse(dets, frame, frame_idx)
            self._step_signal()
            if self.video_time - last_hist >= 1 or self.video_time < last_hist:
                last_hist = self.video_time
                self._record_history()
            img = self._render(frame, dets, flagged)
            if self.on_frame:
                self.on_frame(img)
            else:
                self.jpeg = self._encode(img)

    def _analyse(self, dets, frame, frame_idx):
        s = self.settings
        h = frame.shape[0]
        now_counts = {lid: _zeros() for lid in self.lanes}
        flagged = set()
        stat_frames = s["stationary_s"] * self.fps
        moving_in_stream: dict[int, int] = {}
        stopped = []
        for i, d in enumerate(dets):
            t = d["track"]
            gx, gy = t.ground
            lane = assign_lane((gx, gy), t.velocity if t.hits > 2 else None, self.streams)
            d["lane"] = lane.id if lane else None
            if lane is None:
                continue
            t.lane = lane.id
            ls = self.lanes[lane.id]
            now_counts[lane.id][d["class"]] += 1
            width = max(t.history[-1][3], 1)
            speed = float(np.linalg.norm(t.velocity)) * self.fps / width    # vehicle widths per second
            if t.hits >= 3:
                ls.speeds.append(speed)
            if speed > 0.5:
                moving_in_stream[ls.stream.index] = moving_in_stream.get(ls.stream.index, 0) + 1
            if t.hits >= 4 and not t.counted:
                t.counted = True
                ls.counted[d["class"]] += 1
                ls.events.append(self.video_time)
                self.vehicles_counted += 1
            # stopped-vehicle check over the configured window
            if (d["bbox"][3] - d["bbox"][1]) >= 0.06 * h and frame_idx - t.history[0][0] >= stat_frames:
                window = [p for p in t.history if frame_idx - p[0] <= stat_frames]
                if len(window) >= 3:
                    xs = np.array([(p[1], p[2]) for p in window])
                    before = [p for p in t.history if frame_idx - p[0] > stat_frames]
                    # it must have been driving before it stopped (vehicles that were
                    # already standing in a queue when first seen are not incidents)
                    moved_before = len(before) >= 2 and np.hypot(before[-1][1] - before[0][1],
                                                                 before[-1][2] - before[0][2]) > 1.0 * width
                    if moved_before and np.max(np.linalg.norm(xs - xs.mean(0), axis=1)) < 0.3 * width:
                        stopped.append((i, d, t, ls, frame_idx - window[0][0]))
        for lid, c in now_counts.items():
            self.lanes[lid].now = c
        for i, d, t, ls, dur in stopped:
            # traffic queued at a red light also stands still; a vehicle counts as an
            # incident only when other vehicles in its direction keep moving past it
            if moving_in_stream.get(ls.stream.index, 0) >= 2:
                flagged.add(i)
                self._incident(ls, d, t, frame, dur)
        for ls in self.lanes.values():
            while ls.events and ls.events[0] < self.video_time - 60:
                ls.events.popleft()
        return flagged

    def _incident(self, ls, d, t, frame, dur_frames):
        gid = self.lane_group[ls.lane.id]
        self.blocked_until[gid] = self._now() + self.settings["incident_hold_s"]
        cur = self.active_incident.get(gid)
        x1, y1, x2, y2 = d["bbox"]
        if cur:             # one active incident per signal group; keep it up to date
            cur["duration_s"] = round(max(cur["duration_s"], dur_frames / self.fps), 1)
            return
        for old in self.incidents[:10]:   # the same stopped vehicle seen again under a new id
            if self.video_time - old["_vt"] < 30 and old["lane"] == ls.lane.id and \
                    abs((x1 + x2) / 2 - old["_cx"]) + abs(y2 - old["_cy"]) < 1.5 * (x2 - x1):
                old["status"] = "Active"
                self.active_incident[gid] = old
                return
        ts = datetime.now()
        snap = draw_lanes(frame, self.streams)
        cv2.rectangle(snap, (x1, y1), (x2, y2), (40, 40, 235), 3)
        name = f"{ts:%Y%m%d_%H%M%S}_{ls.lane.id}_{t.id}.jpg"
        cv2.imwrite(os.path.join(self.incident_dir, name), snap)
        group = next(g for g in self.groups if g["id"] == gid)
        rec = {"id": len(self.incidents) + 1, "time": ts.strftime("%H:%M:%S"), "date": ts.strftime("%d %b %Y"),
               "video_time": _fmt_t(self.video_time), "lane": ls.lane.id, "group": gid,
               "group_name": group["name"], "vehicle": CATEGORY_NAMES[d["class"]], "kind": d.get("kind", ""),
               "duration_s": round(dur_frames / self.fps, 1),
               "type": "Stationary vehicle: possible breakdown, accident or blockage",
               "snapshot": name, "status": "Active", "_track": t.id, "_cx": (x1 + x2) / 2, "_cy": y2,
               "_vt": self.video_time}
        self.active_incident[gid] = rec
        self.incidents.insert(0, rec)

    def _group_counts(self):
        out = {}
        for g in self.groups:
            by = _zeros()
            for lid in g["lanes"]:
                for c, n in self.lanes[lid].now.items():
                    by[c] += n
            out[g["id"]] = by
        return out

    def _step_signal(self):
        if not self.controller:
            return
        now = self._now()
        loads = {gid: pcu_load(by) for gid, by in self._group_counts().items()}
        blocked = {g for g, t in self.blocked_until.items() if t > now}
        for gid, rec in list(self.active_incident.items()):
            if gid not in blocked:
                rec["status"] = "Cleared"
                del self.active_incident[gid]
        with self.lock:
            self.signal = self.controller.update(loads, blocked)

    def _now(self):
        return time.time() if self.realtime else self.video_time

    def _record_history(self):
        row = {"t": datetime.now().strftime("%H:%M:%S"), "vt": round(self.video_time, 1), "groups": {}}
        for gid, by in self._group_counts().items():
            row["groups"][gid] = {"vehicles": sum(by.values()), "pcu": pcu_load(by)}
        self.history.append(row)

    # ------------------------------------------------------------ drawing
    def _render_learning(self, frame, dets, tracker):
        out = self.detector.draw(frame, dets)
        for traj in tracker.trajectories().values():
            if len(traj) > 1:
                p = np.array(traj)[:, :2].astype(np.int32)
                cv2.polylines(out, [p], False, (255, 255, 255), 1, cv2.LINE_AA)
        h, w = out.shape[:2]
        bar = out.copy()
        cv2.rectangle(bar, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.addWeighted(bar, 0.65, out, 0.35, 0, dst=out)
        cv2.putText(out, f"LEARNING ROAD LAYOUT  {self.progress:.0f}%", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, 34), (int(w * self.progress / 100), 37), (0, 220, 255), -1)
        return out

    def _render(self, frame, dets, flagged):
        lane_tags = {lid: str(ls.count) for lid, ls in self.lanes.items()}
        out = draw_lanes(frame, self.streams, lane_tags)
        out = self.detector.draw(out, dets, flagged)
        h, w = out.shape[:2]
        bar = out.copy()
        cv2.rectangle(bar, (0, 0), (w, 34), (20, 20, 20), -1)
        cv2.addWeighted(bar, 0.6, out, 0.4, 0, dst=out)
        cv2.circle(out, (16, 17), 5, (40, 40, 235), -1)
        cv2.putText(out, f"LIVE  {(self.video_name or '')[:40]}", (28, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        stamp = f"{datetime.now():%d-%m-%Y %H:%M:%S}"
        cv2.putText(out, stamp, (w - 175, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        # one signal lamp per group, bottom-right
        lights = self.signal.get("lights", {})
        x = w - 10
        for g in reversed(self.groups):
            label = g["name"].replace(" approach", "")
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            x0 = x - tw - 32
            chip = out.copy()
            cv2.rectangle(chip, (x0, h - 34), (x, h - 8), (20, 20, 20), -1)
            cv2.addWeighted(chip, 0.7, out, 0.3, 0, dst=out)
            cv2.circle(out, (x0 + 12, h - 21), 7, LIGHT_BGR[lights.get(g["id"], "RED")], -1)
            cv2.putText(out, label, (x0 + 24, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
            x = x0 - 6
        if flagged:
            cv2.rectangle(out, (0, 40), (w, 72), (40, 40, 200), -1)
            cv2.putText(out, "INCIDENT: STATIONARY VEHICLE", (12, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return out

    @staticmethod
    def _encode(img):
        if img.shape[1] > 1280:
            img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return jpg.tobytes() if ok else None

    # -------------------------------------------------------------- state
    def _flow(self, ls):
        """Vehicles per minute over the last minute of video."""
        span = min(60.0, max(self.video_time, 1e-6))
        return round(len(ls.events) * 60 / span, 1)

    def snapshot(self) -> dict:
        now = time.time()
        with self.lock:
            sig = dict(self.signal)
        s = self.settings
        lights, plan = sig.get("lights", {}), sig.get("plan", {})
        gcounts = self._group_counts() if self.groups else {}
        lanes = []
        for lid, ls in list(self.lanes.items()):
            sp = list(ls.speeds)
            n_in_group = max(1, len(self._group(lid)["lanes"]))
            lanes.append({"id": lid, "group": self.lane_group[lid], "stream": ls.stream.name,
                          "color": "#%02x%02x%02x" % tuple(ls.lane.color[::-1]),
                          "count": ls.count, "pcu": ls.pcu, "by_type": ls.now, "counted": ls.counted,
                          "counted_total": sum(ls.counted.values()), "flow_per_min": self._flow(ls),
                          "speed": round(float(np.median(sp)), 2) if sp else 0.0,
                          "density": classify_density(ls.pcu, max(2, s["low_max"] / n_in_group),
                                                      max(4, s["medium_max"] / n_in_group))})
        groups = []
        for g in self.groups:
            by = gcounts.get(g["id"], _zeros())
            load = pcu_load(by)
            gl = [l for l in lanes if l["group"] == g["id"]]
            inc = self.active_incident.get(g["id"])
            groups.append({**g, "count": sum(by.values()), "pcu": load, "by_type": by,
                           "density": classify_density(load, s["low_max"], s["medium_max"]),
                           "light": lights.get(g["id"], "RED"), "green_time": plan.get(g["id"], 0),
                           "flow_per_min": round(sum(l["flow_per_min"] for l in gl), 1),
                           "counted": sum(l["counted_total"] for l in gl),
                           "blocked": self.blocked_until.get(g["id"], 0) > self._now(),
                           "incident": {k: v for k, v in inc.items() if not k.startswith("_")} if inc else None})
        mix = _zeros()
        for ls in list(self.lanes.values()):
            for c, n in ls.counted.items():
                mix[c] += n
        return {
            "status": self.status, "message": self.message, "progress": self.progress,
            "video": {"name": self.video_name, "fps": round(self.fps, 1), "width": self.frame_size[1],
                      "height": self.frame_size[0],
                      "duration_s": round(self.total_frames / self.fps, 1) if self.total_frames else 0,
                      "time": _fmt_t(self.video_time), "loops": self.loops},
            "model": self.detector.model_name, "supports_3w": self.detector.supports_3w,
            "infer_ms": round(self.infer_ms), "det_fps": round(self.det_fps, 1),
            "time": datetime.now().strftime("%H:%M:%S"), "uptime_s": int(now - self.started),
            "group_mode": self.group_mode, "signal": sig, "groups": groups, "lanes": lanes,
            "n_streams": len(self.streams), "total_now": sum(g["count"] for g in groups),
            "pcu_now": round(sum(g["pcu"] for g in groups), 1), "vehicles_counted": self.vehicles_counted,
            "mix_counted": mix, "active_incidents": sum(1 for g in groups if g["blocked"]),
            "incidents_total": len(self.incidents), "settings": s,
        }

    def _group(self, lid):
        gid = self.lane_group[lid]
        return next(g for g in self.groups if g["id"] == gid)

    def analytics(self) -> dict:
        with self.lock:
            sig_hist = list(self.controller.history) if self.controller else []
        names = {g["id"]: g["name"] for g in self.groups}
        return {
            "groups": [{"id": g["id"], "name": g["name"]} for g in self.groups],
            "history": list(self.history),
            "signal_history": [{"time": datetime.fromtimestamp(t).strftime("%H:%M:%S"), "group": gid,
                                "name": names.get(gid, gid), "green": g, "pcu": c}
                               for t, gid, g, c in reversed(sig_hist)],
        }

    def performance(self) -> dict | None:
        """Fixed-time vs adaptive control for the traffic measured in this video."""
        if self.status != "live" or self.video_time < 20:
            return None
        if self._perf and time.time() - self._perf_t < 30:
            return self._perf
        snap = self.snapshot()
        rates = [g["flow_per_min"] / 60.0 for g in snap["groups"]]
        if not any(rates):
            return None
        res = compare_controllers(rates, [len(g["lanes"]) for g in snap["groups"]], self._timing())
        res["groups"] = [g["name"] for g in snap["groups"]]
        res["rates_per_min"] = [round(r * 60, 1) for r in rates]
        self._perf, self._perf_t = res, time.time()
        return res


def _prep(frame):
    """Small videos are scaled up to 960 px wide so overlays stay readable.
    Detection runs at the model's own input size either way."""
    h, w = frame.shape[:2]
    if w < 960:
        frame = cv2.resize(frame, (960, int(round(h * 960 / w))), interpolation=cv2.INTER_LINEAR)
    return frame


def _fmt_t(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"
