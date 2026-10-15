"""
Vehicle detection and classification (report section 4.5).

Uses a pretrained YOLOv8 model (COCO weights, no training needed) to find
vehicles in a frame. Only the four COCO vehicle classes are kept:
car (2), motorcycle (3), bus (5) and truck (7).

Each detection is returned as a dict so the other modules (density, signal,
incident) can work without knowing anything about YOLO.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# BGR colours used for the boxes of each class
CLASS_COLORS = {
    "car": (80, 200, 60),
    "motorcycle": (0, 190, 255),
    "bus": (255, 140, 30),
    "truck": (200, 80, 220),
}

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "yolov8n.pt")


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
    """Wraps a YOLOv8 model and returns vehicle detections for a frame."""

    def __init__(self, model_path: str = DEFAULT_MODEL, conf: float = 0.3, imgsz: int = 640):
        # Ultralytics downloads the weights automatically if the file is missing
        self.model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return a list of detections: class, confidence, bbox (x1,y1,x2,y2), center."""
        result = self.model(preprocess(frame), conf=self.conf, imgsz=self.imgsz,
                            classes=list(VEHICLE_CLASSES), verbose=False)[0]
        detections = []
        for (x1, y1, x2, y2), cls, cf in zip(result.boxes.xyxy.tolist(),
                                             result.boxes.cls.int().tolist(),
                                             result.boxes.conf.tolist()):
            detections.append({
                "class": VEHICLE_CLASSES[cls],
                "confidence": round(float(cf), 3),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            })
        return detections

    @staticmethod
    def draw(frame: np.ndarray, detections: list[dict], flagged: set[int] | None = None) -> np.ndarray:
        """Draw boxes and 'class confidence' labels. Indices in `flagged` are drawn red."""
        out = frame.copy()
        flagged = flagged or set()
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            color = (40, 40, 235) if i in flagged else CLASS_COLORS[d["class"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d['class']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1, th + 6)
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
            cv2.putText(out, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def detect_and_draw(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Convenience: detections plus the annotated frame."""
        dets = self.detect(frame)
        return self.draw(frame, dets), dets
