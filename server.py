"""
AI-Enabled Intelligent System for Smart Road Traffic Management - web application.

Run:   python server.py
Open:  http://localhost:5000
"""
import glob
import json
import os
import sys
import threading
import uuid

import cv2
import numpy as np
import pandas as pd
from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_from_directory)
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from simulation import (AdaptiveController, FixedTimeController,  # noqa: E402
                        QLearningController, simulate, train_rl)

DATA = os.path.join(ROOT, "data")
UPLOADS = os.path.join(ROOT, "uploads")
RESULTS = os.path.join(ROOT, "results")
MODELS = {"yolov8n.pt", "yolov8s.pt"}
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

_analyzers = {}          # (model, imgsz, conf, lanes, stall) -> TrafficAnalyzer
_sessions = {}           # session id -> live stats for the detection page
_agent = None
_lock = threading.Lock()


def get_analyzer(model, imgsz, conf, lanes, stall):
    from detector import Config, TrafficAnalyzer
    key = (model, imgsz, conf, lanes, stall)
    if key not in _analyzers:
        _analyzers[key] = TrafficAnalyzer(Config(
            model_path=os.path.join(ROOT, "models", model), imgsz=imgsz, conf=conf,
            n_lanes=lanes, stall_seconds=stall))
    return _analyzers[key]


def get_agent():
    global _agent
    if _agent is None:
        q_path = os.path.join(RESULTS, "signal", "q_table.npy")
        if os.path.exists(q_path):
            _agent = QLearningController()
            _agent.q = np.load(q_path)
            _agent.training = False
        else:
            _agent, _ = train_rl(episodes=200)
    return _agent


def video_path(name):
    name = secure_filename(name)
    for folder in (DATA, UPLOADS):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
    abort(404, "video not found")


# ------------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------- videos
@app.route("/api/videos")
def videos():
    files = [os.path.basename(p) for d in (DATA, UPLOADS)
             for p in sorted(glob.glob(os.path.join(d, "*")))
             if p.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    return jsonify(files)


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="no file"), 400
    name = secure_filename(f.filename)
    f.save(os.path.join(UPLOADS, name))
    return jsonify(name=name)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated frames; live numbers go to _sessions[sid]."""
    a = request.args
    sid = a.get("sid") or uuid.uuid4().hex
    path = video_path(a.get("video", ""))
    model = a.get("model", "yolov8n.pt")
    if model not in MODELS:
        abort(400)
    imgsz = int(a.get("imgsz", 640))
    conf = float(a.get("conf", 0.25))
    lanes = int(a.get("lanes", 2))
    stall = float(a.get("stall", 6))
    max_frames = int(a.get("max_frames", 0)) or None
    analyzer = get_analyzer(model, imgsz, conf, lanes, stall)
    total = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames or total)
    state = _sessions[sid] = {"done": False, "frame": 0, "total": limit, "stats": None,
                              "history": [], "alerts": [], "lanes": lanes}

    def generate():
        with _lock:   # one detection stream at a time keeps CPU usage sane
            for i, (frame, stats) in enumerate(analyzer.frames(path, max_frames)):
                state.update(frame=i + 1, stats=stats, alerts=list(analyzer.alerts))
                state["history"].append(stats)
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            state["done"] = True

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/live_stats")
def live_stats():
    s = _sessions.get(request.args.get("sid", ""))
    if not s:
        return jsonify(ready=False)
    hist = s["history"]
    step = max(1, len(hist) // 300)          # keep the chart payload small
    summary = None
    if hist:
        df = pd.DataFrame(hist)
        summary = {
            "frames": len(df),
            "unique_vehicles": int(df.unique_vehicles.iloc[-1]),
            "avg_infer_ms": round(float(df.infer_ms.mean()), 1),
            "class_totals": {c: int(df[c].sum()) for c in ["car", "motorcycle", "bus", "truck"]},
        }
    return jsonify(ready=True, done=s["done"], frame=s["frame"], total=s["total"],
                   lanes=s["lanes"], stats=s["stats"], alerts=s["alerts"], summary=summary,
                   history=[{k: h[k] for k in h if k == "time_s" or k.endswith("_count")}
                            for h in hist[::step]])


# ------------------------------------------------------------ simulation
@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    body = request.get_json(force=True) or {}
    scenario = body.get("scenario", "rush")
    if scenario not in ("low", "medium", "high", "rush"):
        abort(400)
    fixed = int(body.get("fixed_green", 30))
    seed = int(body.get("seed", 7))
    out, queues = [], {}
    for c in [FixedTimeController(fixed), AdaptiveController(), get_agent()]:
        sim = simulate(c, scenario, seed=seed)
        m = sim.metrics()
        out.append({"controller": c.name, **{k: round(float(v), 2) for k, v in m.items()}})
        q = np.array(sim.queue_log).sum(axis=1)[:3600]
        queues[c.name] = q[::30].tolist()          # one point every 30 s
    return jsonify(results=out, queues=queues, minutes=[i / 2 for i in range(len(next(iter(queues.values()))))])


# --------------------------------------------------------------- results
@app.route("/api/results")
def results():
    summ = os.path.join(RESULTS, "signal", "comparison_summary.csv")
    signal = pd.read_csv(summ).to_dict(orient="records") if os.path.exists(summ) else []
    detection = []
    for js in sorted(glob.glob(os.path.join(RESULTS, "detection", "*_summary.json"))):
        name = os.path.basename(js).replace("_summary.json", "")
        d = json.load(open(js))
        d["name"] = name
        d["video_url"] = f"/results/detection/{name}_annotated.mp4"
        d["charts"] = [f"/results/detection/{os.path.basename(p)}"
                       for p in sorted(glob.glob(os.path.join(RESULTS, "detection", f"{name}_*.png")))]
        alerts = os.path.join(RESULTS, "detection", f"{name}_alerts.csv")
        try:
            d["alerts"] = pd.read_csv(alerts).to_dict(orient="records")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            d["alerts"] = []
        detection.append(d)
    charts = [f"/results/signal/{os.path.basename(p)}"
              for p in sorted(glob.glob(os.path.join(RESULTS, "signal", "*.png")))]
    return jsonify(signal=signal, detection=detection, signal_charts=charts)


@app.route("/results/<path:p>")
def result_file(p):
    return send_from_directory(RESULTS, p)


if __name__ == "__main__":
    print("Smart Traffic Management running at http://localhost:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
