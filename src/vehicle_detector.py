"""
Vehicle detection and classification (report section 4.5).

Vehicles are grouped the way Indian traffic rules and PCU tables group them,
by number of wheels:

    2-Wheeler      motorcycles, scooters, bicycles
    3-Wheeler      auto-rickshaws
    4-Wheeler      cars, jeeps, vans
    Heavy Vehicle  buses and trucks (6 or more wheels)

The default model is YOLOE (an open-vocabulary YOLO) with the vocabulary
"motorcycle, scooter, bicycle, auto rickshaw, car, bus, truck" built in, so it
recognises auto-rickshaws, which the standard COCO model cannot. If that model
file is missing, the COCO YOLOv8n model is used instead (no 3-wheeler class).
No training is needed for either.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from ultralytics import YOLO

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL = os.path.join(MODELS, "traffic_yoloe_s.pt")
FALLBACK_MODEL = os.path.join(MODELS, "yolov8n.pt")

CATEGORIES = ("2W", "3W", "4W", "HV")
CATEGORY_NAMES = {"2W": "2-Wheeler", "3W": "3-Wheeler", "4W": "4-Wheeler", "HV": "Heavy Vehicle"}
CATEGORY_HINT = {"2W": "bikes, scooters", "3W": "auto-rickshaws", "4W": "cars, vans",
                 "HV": "buses, trucks (6+ wheels)"}
# Passenger Car Units (IRC:106-1990): how much road space each type takes
PCU = {"2W": 0.5, "3W": 1.0, "4W": 1.0, "HV": 3.0}

# model class name -> category
NAME_TO_CATEGORY = {"motorcycle": "2W", "scooter": "2W", "bicycle": "2W", "auto rickshaw": "3W",
                    "car": "4W", "bus": "HV", "truck": "HV"}

# BGR box colours
CLASS_COLORS = {"2W": (0, 190, 255), "3W": (60, 220, 60), "4W": (255, 170, 40), "HV": (210, 80, 220)}


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Preprocessing (report section 4.4).

    A light Gaussian blur removes sensor noise, and CLAHE on the luminance
    channel evens out lighting so vehicles in shadow are still detected.
    """
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class VehicleDetector:
    """Wraps a YOLO model and returns categorised vehicle detections for a frame."""

    def __init__(self, model_path: str = DEFAULT_MODEL, conf: float = 0.3, imgsz: int = 640):
        if not os.path.exists(model_path):
            model_path = FALLBACK_MODEL if os.path.exists(FALLBACK_MODEL) else "yolov8n.pt"
        self.model = YOLO(model_path)
        self.model_name = os.path.basename(model_path)
        self.conf = conf
        self.imgsz = imgsz
        names = self.model.names
        self.class_map = {i: NAME_TO_CATEGORY[n] for i, n in names.items() if n in NAME_TO_CATEGORY}
        self.raw_names = names

    @property
    def supports_3w(self) -> bool:
        return "3W" in self.class_map.values()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return detections: class (category), kind (model label), confidence, bbox, center."""
        result = self.model(preprocess(frame), conf=self.conf, imgsz=self.imgsz,
                            classes=list(self.class_map), verbose=False)[0]
        detections = []
        for (x1, y1, x2, y2), cls, cf in zip(result.boxes.xyxy.tolist(),
                                             result.boxes.cls.int().tolist(),
                                             result.boxes.conf.tolist()):
            detections.append({
                "class": self.class_map[cls],
                "kind": self.raw_names[cls],
                "confidence": round(float(cf), 3),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            })
        return _suppress_overlaps(detections)

    @staticmethod
    def draw(frame: np.ndarray, detections: list[dict], flagged: set[int] | None = None) -> np.ndarray:
        """Draw boxes and labels. Indices in `flagged` are drawn red."""
        out = frame.copy()
        flagged = flagged or set()
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            color = (40, 40, 235) if i in flagged else CLASS_COLORS[d["class"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = d["class"]
            if d.get("track") is not None:
                label += f" #{d['track'].id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(y1, th + 6)
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
            cv2.putText(out, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return out


def _suppress_overlaps(dets: list[dict], thr: float = 0.7) -> list[dict]:
    """Open-vocabulary models sometimes label one vehicle twice (e.g. car and truck).
    Keep the more confident box when two boxes overlap heavily."""
    dets = sorted(dets, key=lambda d: -d["confidence"])
    keep = []
    for d in dets:
        a = d["bbox"]
        ok = True
        for k in keep:
            b = k["bbox"]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if small > 0 and inter / small > thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep
