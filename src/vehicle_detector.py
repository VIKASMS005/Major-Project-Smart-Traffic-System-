"""
Vehicle detection and classification (report section 4.5).

Vehicles are grouped the way Indian traffic rules and PCU tables group them,
by number of wheels:

    2-Wheeler      motorcycles, scooters, bicycles
    3-Wheeler      auto-rickshaws
    4-Wheeler      cars, jeeps, vans
    Heavy Vehicle  buses and trucks (6 or more wheels)

Two pretrained models work together (no training needed): COCO YOLOv8n finds
the vehicles, and YOLOE, an open-vocabulary YOLO with the vocabulary
"motorcycle, scooter, bicycle, auto rickshaw, car, bus, truck" built in,
recognises auto-rickshaws, which the COCO model cannot.
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


class _Model:
    def __init__(self, path):
        self.yolo = YOLO(path)
        self.name = os.path.basename(path)
        self.names = self.yolo.names
        self.class_map = {i: NAME_TO_CATEGORY[n] for i, n in self.names.items() if n in NAME_TO_CATEGORY}

    def __call__(self, img, conf, imgsz):
        r = self.yolo(img, conf=conf, imgsz=imgsz, classes=list(self.class_map), verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), cls, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(),
                                             r.boxes.conf.tolist()):
            out.append({"class": self.class_map[cls], "kind": self.names[cls], "confidence": round(float(cf), 3),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)), "center": ((x1 + x2) / 2, (y1 + y2) / 2)})
        return out


class VehicleDetector:
    """Two-model detector.

    * YOLOv8 (COCO) finds and boxes the vehicles. It has the best recall for cars,
      buses, trucks and motorcycles, but no auto-rickshaw class.
    * YOLOE (open vocabulary, Indian traffic words) looks at the same frame. Where it
      sees an auto-rickshaw (or a two-wheeler), that label replaces the COCO label,
      and autos COCO missed are added.

    If only one of the two model files exists, that model is used on its own.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, conf: float = 0.3, imgsz: int = 640,
                 base_path: str = FALLBACK_MODEL):
        self.base = _Model(base_path) if os.path.exists(base_path) else None
        self.vocab = _Model(model_path) if os.path.exists(model_path) and model_path != base_path else None
        if self.base is None and self.vocab is None:
            self.base = _Model("yolov8n.pt")            # Ultralytics downloads it
        self.model_name = " + ".join(m.name for m in (self.base, self.vocab) if m)
        self.conf = conf
        self.imgsz = imgsz
        self.vocab_conf = 0.15        # open-vocabulary scores run lower than COCO scores
        self.vocab_imgsz = 960

    @property
    def supports_3w(self) -> bool:
        return self.vocab is not None and "3W" in self.vocab.class_map.values()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return detections: class (category), kind (model label), confidence, bbox, center."""
        img = preprocess(frame)
        base = self.base(img, self.conf, self.imgsz) if self.base else []
        if self.vocab is None:
            return _suppress_overlaps(base)
        vocab = self.vocab(img, min(self.conf, self.vocab_conf), self.vocab_imgsz)
        if self.base is None:
            return _suppress_overlaps(vocab)
        special = [v for v in vocab if v["class"] in ("3W", "2W")]
        used = set()
        for d in base:
            best, bi = 0.45, None
            for i, v in enumerate(special):
                o = _iou(d["bbox"], v["bbox"])
                if o > best:
                    best, bi = o, i
            if bi is not None and not (special[bi]["class"] == "2W" and d["class"] == "HV"):
                used.add(bi)
                d["class"], d["kind"] = special[bi]["class"], special[bi]["kind"]
        extra = [v for i, v in enumerate(special) if i not in used and v["class"] == "3W"
                 and v["confidence"] >= 0.25 and all(_iou(v["bbox"], d["bbox"]) < 0.3 for d in base)]
        return _suppress_overlaps(base + extra)

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


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


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
