"""
SmartSignal: AI traffic control centre (web application).

Run:   python app.py
Open:  http://localhost:5000

Upload a traffic video. The system learns the lanes and traffic directions from
the vehicles in it, then detects, counts and classifies vehicles per lane,
measures density and runs adaptive signal control live.

Pages
  /upload      Upload a video (or pick one analysed before)
  /            Control Center: live annotated video, signals per approach/lane, alerts
  /analytics   Traffic trends, vehicle mix, flow per lane, adaptive vs fixed-time
  /signals     Signal timing rule and log of every green phase issued
  /incidents   Incident log with snapshots
  /settings    Density thresholds, signal timing, detection and incident settings
"""
import os
import re
import sys
import time

from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   send_from_directory)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from live_engine import DEFAULT_SETTINGS, TrafficEngine  # noqa: E402
from vehicle_detector import CATEGORY_HINT, CATEGORY_NAMES, PCU  # noqa: E402

UPLOAD_DIR = os.path.join(ROOT, "uploads")
SAMPLE_DIR = os.path.join(ROOT, "sample_videos")
INCIDENT_DIR = os.path.join(ROOT, "outputs", "incidents")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024      # 1 GB
engine = TrafficEngine(INCIDENT_DIR)


@app.context_processor
def inject_globals():
    return {"engine": engine, "categories": CATEGORY_NAMES, "category_hint": CATEGORY_HINT, "pcu": PCU}


def _needs_video(page):
    return engine.status == "empty" and page != "upload"


# ------------------------------------------------------------------ pages
def _page(template, page):
    if _needs_video(page):
        return redirect("/upload")
    return render_template(template, page=page)


@app.route("/")
def control_center():
    return _page("control_center.html", "control")


@app.route("/upload")
def upload_page():
    return render_template("upload.html", page="upload")


@app.route("/analytics")
def analytics_page():
    return _page("analytics.html", "analytics")


@app.route("/incidents")
def incidents_page():
    return _page("incidents.html", "incidents")


@app.route("/signals")
def signals_page():
    return _page("signals.html", "signals")


@app.route("/settings")
def settings_page():
    return render_template("settings.html", page="settings")


# ------------------------------------------------------------------ video
def _videos():
    out = []
    for folder, kind in ((UPLOAD_DIR, "upload"), (SAMPLE_DIR, "sample")):
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(VIDEO_EXT):
                p = os.path.join(folder, f)
                out.append({"name": f, "kind": kind, "size_mb": round(os.path.getsize(p) / 1e6, 1),
                            "modified": os.path.getmtime(p)})
    return sorted(out, key=lambda v: (v["kind"] != "upload", -v["modified"]))


@app.route("/api/videos")
def api_videos():
    return jsonify(_videos())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify({"error": "No file received."}), 400
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(f.filename))
    if not name.lower().endswith(VIDEO_EXT):
        return jsonify({"error": "Please upload a video file (mp4, avi, mov, mkv, webm)."}), 400
    path = os.path.join(UPLOAD_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{name}")
    f.save(path)
    engine.load(path, name)
    return jsonify({"ok": True, "name": name})


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """Analyse a video already on the server (a previous upload or a sample clip)."""
    data = request.get_json(force=True) or {}
    v = next((v for v in _videos() if v["name"] == data.get("name") and v["kind"] == data.get("kind")), None)
    if v is None:
        return jsonify({"error": "Video not found."}), 404
    folder = UPLOAD_DIR if v["kind"] == "upload" else SAMPLE_DIR
    display = re.sub(r"^\d{8}_\d{6}_", "", v["name"])
    engine.load(os.path.join(folder, v["name"]), display)
    return jsonify({"ok": True})


@app.route("/api/relearn", methods=["POST"])
def api_relearn():
    engine.relearn()
    return jsonify({"ok": True})


# ----------------------------------------------------------------- streams
@app.route("/stream")
def stream():
    def frames():
        last, sent = None, 0.0
        while True:
            jpg = engine.jpeg
            # new frames as they come; the same frame again every second once the
            # video has finished, so the browser keeps showing it
            if jpg is not None and (jpg is not last or time.time() - sent > 1):
                last, sent = jpg, time.time()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.04)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/layout.jpg")
def layout_image():
    if engine.layout_jpeg is None:
        abort(404)
    return Response(engine.layout_jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------- API
@app.route("/api/state")
def api_state():
    return jsonify(engine.snapshot())


@app.route("/api/analytics")
def api_analytics():
    return jsonify(engine.analytics())


@app.route("/api/incidents")
def api_incidents():
    return jsonify([{k: v for k, v in i.items() if not k.startswith("_")} for i in engine.incidents])


@app.route("/snapshots/<path:name>")
def snapshot(name):
    return send_from_directory(INCIDENT_DIR, name)


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        if data.get("reset"):
            data = DEFAULT_SETTINGS
        engine.update_settings(data)
    return jsonify(engine.settings)


@app.route("/api/performance")
def api_performance():
    """Average wait per vehicle with adaptive vs fixed-time control, for the measured traffic."""
    return jsonify(engine.performance())


if __name__ == "__main__":
    print("\n  SmartSignal running at  http://localhost:5000  (open it and upload a traffic video)\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True, debug=False)
