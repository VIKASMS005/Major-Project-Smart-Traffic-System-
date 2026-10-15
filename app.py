"""
Smart Traffic Control Center: web application.

Run:   python app.py
Open:  http://localhost:5000

Pages
  /            Control Center: live cameras, animated junction, signal timing, alerts
  /analytics   Traffic trends, vehicle mix, green-time allocation, performance vs fixed-time
  /incidents   Incident log with snapshots
  /signals     Signal timing log (every green phase the controller issued)
  /settings    Density thresholds, signal timing rules, incident sensitivity
"""
import os
import sys
import time

import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from live_engine import DEFAULT_SETTINGS, TrafficEngine  # noqa: E402
from simulation import (AdaptiveController, FixedTimeController,  # noqa: E402
                        QLearningController, simulate, train_rl)

VIDEO_DIR = os.path.join(ROOT, "sample_videos")
INCIDENT_DIR = os.path.join(ROOT, "outputs", "incidents")

app = Flask(__name__)
engine = TrafficEngine(VIDEO_DIR, INCIDENT_DIR)
_performance = None


@app.context_processor
def inject_globals():
    return {"junction_name": "Junction 01", "lanes": engine.lanes}


# ------------------------------------------------------------------ pages
@app.route("/")
def control_center():
    return render_template("control_center.html", page="control")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html", page="analytics")


@app.route("/incidents")
def incidents_page():
    return render_template("incidents.html", page="incidents")


@app.route("/signals")
def signals_page():
    return render_template("signals.html", page="signals")


@app.route("/settings")
def settings_page():
    return render_template("settings.html", page="settings")


# ----------------------------------------------------------------- streams
@app.route("/stream/<lane_id>")
def stream(lane_id):
    lane = next((l for l in engine.lanes if l.id == lane_id), None)
    if lane is None:
        abort(404)

    def frames():
        last = None
        while True:
            jpg = lane.jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.06)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
    """Waiting time with adaptive / AI control vs a fixed-time signal (simulated junction)."""
    global _performance
    if _performance is None:
        q_path = os.path.join(ROOT, "models", "signal_q_table.npy")
        if os.path.exists(q_path):
            agent = QLearningController()
            agent.q, agent.training = np.load(q_path), False
        else:
            agent, _ = train_rl(episodes=200)
        rows = []
        for sc in ["low", "medium", "high", "rush"]:
            row = {"scenario": sc}
            for key, c in [("fixed", FixedTimeController()), ("adaptive", AdaptiveController()), ("ai", agent)]:
                m = [simulate(c, sc, seed=s).metrics() for s in (11, 22, 33)]
                row[key] = round(float(np.mean([x["avg_wait_s"] for x in m])), 1)
            rows.append(row)
        _performance = rows
    return jsonify(_performance)


if __name__ == "__main__":
    engine.start()
    print("\n  Smart Traffic Control Center running at  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True, debug=False)
