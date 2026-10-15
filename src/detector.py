"""
Vehicle detection, counting, density estimation, adaptive signal timing and
incident (stalled vehicle) detection on a traffic video using YOLOv8.

Pipeline per frame (mirrors the block diagram in the report):
  preprocessing -> YOLO detection + tracking -> lane-wise counting ->
  density level -> adaptive green time -> incident check -> annotated output
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class ids that are vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {"car": (0, 200, 0), "motorcycle": (255, 160, 0),
                "bus": (0, 140, 255), "truck": (200, 0, 200)}
DENSITY_NAMES = ["LOW", "MEDIUM", "HIGH"]
DENSITY_COLORS = [(0, 200, 0), (0, 200, 255), (0, 0, 255)]


@dataclass
class Config:
    model_path: str = "models/yolov8s.pt"
    conf: float = 0.25
    imgsz: int = 960
    n_lanes: int = 2                 # frame split into equal vertical lanes
    low_max: int = 4                 # <= low_max vehicles in lane -> LOW
    medium_max: int = 9              # <= medium_max -> MEDIUM, else HIGH
    min_green: int = 10              # seconds
    max_green: int = 60
    sec_per_vehicle: float = 3.0
    stall_seconds: float = 6.0       # stationary this long -> incident alert
    stall_move_frac: float = 0.15    # max movement, as a fraction of the vehicle's box height
    stall_min_box: float = 0.07      # ignore vehicles smaller than this fraction of frame
                                     # height: too far away to judge motion reliably


def density_level(count: int, cfg: Config) -> int:
    if count <= cfg.low_max:
        return 0
    if count <= cfg.medium_max:
        return 1
    return 2


def green_time(count: int, cfg: Config) -> int:
    return int(np.clip(cfg.min_green + cfg.sec_per_vehicle * count,
                       cfg.min_green, cfg.max_green))


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Light denoise + contrast normalisation (CLAHE on luminance)."""
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class TrafficAnalyzer:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.model = YOLO(self.cfg.model_path)
        self.reset()

    def reset(self):
        self.tracks = defaultdict(lambda: deque(maxlen=300))  # id -> (frame, cx, cy)
        self.track_class = {}
        self.alerted = set()
        self.alerts = []
        self.unique_ids = set()
        self.frame_idx = 0

    # ------------------------------------------------------------------ core
    def process_frame(self, frame: np.ndarray, fps: float) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        h, w = frame.shape[:2]
        self._fps = fps
        t0 = time.perf_counter()
        res = self.model.track(preprocess(frame), persist=True, conf=cfg.conf,
                               imgsz=cfg.imgsz, classes=list(VEHICLE_CLASSES),
                               tracker="bytetrack.yaml", verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        lane_w = w / cfg.n_lanes
        lane_counts = [0] * cfg.n_lanes
        class_counts = {c: 0 for c in VEHICLE_CLASSES.values()}
        confs = []
        out = frame.copy()

        boxes = res.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for xyxy, cls, cf, tid in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(),
                                      boxes.conf.tolist(), ids):
            name = VEHICLE_CLASSES[cls]
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            lane = min(int(cx // lane_w), cfg.n_lanes - 1)
            lane_counts[lane] += 1
            class_counts[name] += 1
            confs.append(cf)
            color = CLASS_COLORS[name]
            stalled = False
            if tid is not None:
                self.unique_ids.add(tid)
                self.track_class[tid] = name
                self.tracks[tid].append((self.frame_idx, cx, cy))
                stalled = self._check_stall(tid, fps, y2 - y1, h)
            if stalled:
                color = (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {cf:.2f}" + (f" #{tid}" if tid is not None else "")
            if stalled:
                label = "STALLED " + label
            cv2.putText(out, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

        levels = [density_level(c, cfg) for c in lane_counts]
        greens = [green_time(c, cfg) for c in lane_counts]
        self._draw_overlay(out, lane_counts, levels, greens, lane_w)

        stats = {
            "frame": self.frame_idx,
            "time_s": round(self.frame_idx / fps, 2),
            "vehicles": sum(lane_counts),
            **{f"lane{i + 1}_count": c for i, c in enumerate(lane_counts)},
            **{f"lane{i + 1}_density": DENSITY_NAMES[l] for i, l in enumerate(levels)},
            **{f"lane{i + 1}_green_s": g for i, g in enumerate(greens)},
            **class_counts,
            "mean_conf": round(float(np.mean(confs)), 3) if confs else None,
            "unique_vehicles": len(self.unique_ids),
            "infer_ms": round(infer_ms, 1),
            "alerts": len(self.alerts),
        }
        self.frame_idx += 1
        return out, stats

    def _check_stall(self, tid: int, fps: float, box_h: float, frame_h: int) -> bool:
        hist = self.tracks[tid]
        need = int(self.cfg.stall_seconds * fps)
        if len(hist) < need or box_h < self.cfg.stall_min_box * frame_h:
            return False
        f0, x0, y0 = hist[-need]
        f1, x1, y1 = hist[-1]
        if f1 - f0 > need * 1.5:          # track had gaps, not reliable
            return False
        moved = np.hypot(x1 - x0, y1 - y0)
        stalled = moved < self.cfg.stall_move_frac * box_h
        if stalled and tid not in self.alerted:
            self.alerted.add(tid)
            self.alerts.append({"time_s": round(self.frame_idx / fps, 1), "track_id": tid,
                                "vehicle": self.track_class.get(tid, "vehicle"),
                                "type": "Stationary vehicle (possible accident / blockage)",
                                "x": int(x1), "y": int(y1)})
        return stalled

    def _draw_overlay(self, img, counts, levels, greens, lane_w):
        h, w = img.shape[:2]
        for i in range(1, len(counts)):
            x = int(i * lane_w)
            cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        # semi-transparent lane panels along the bottom edge
        shade = img.copy()
        for i in range(len(counts)):
            x = int(i * lane_w) + 8
            cv2.rectangle(shade, (x - 4, h - 74), (x + 200, h - 6), (0, 0, 0), -1)
        cv2.addWeighted(shade, 0.55, img, 0.45, 0, dst=img)
        for i, (c, l, g) in enumerate(zip(counts, levels, greens)):
            x = int(i * lane_w) + 8
            cv2.putText(img, f"Lane {i + 1}: {c} vehicles", (x, h - 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"Density: {DENSITY_NAMES[l]}", (x, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, DENSITY_COLORS[l], 1, cv2.LINE_AA)
            cv2.putText(img, f"Green time: {g}s", (x, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if self.alerts and self.frame_idx / self._fps - self.alerts[-1]["time_s"] < 5:
            a = self.alerts[-1]
            cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 180), -1)
            cv2.putText(img, f"ALERT t={a['time_s']}s: {a['type']} ({a['vehicle']} #{a['track_id']})",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------------------- helpers
    def frames(self, video_path: str, max_frames: int | None = None):
        """Yield (annotated_frame, stats) for each frame of a video."""
        self.reset()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and n >= max_frames):
                break
            yield self.process_frame(frame, fps)
            n += 1
        cap.release()

    def process_video(self, video_path: str, out_path: str | None = None,
                      max_frames: int | None = None, progress=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        writer, rows = None, []
        t0 = time.perf_counter()
        for out, stats in self.frames(video_path, max_frames):
            if out_path and writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if writer:
                writer.write(out)
            rows.append(stats)
            if progress:
                progress(len(rows), min(total, max_frames or total))
        if writer:
            writer.release()
        elapsed = time.perf_counter() - t0
        return rows, list(self.alerts), len(rows) / max(elapsed, 1e-6)
