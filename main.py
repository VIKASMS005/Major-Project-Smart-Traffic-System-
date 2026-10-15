"""
AI-Enabled Intelligent System for Smart Road Traffic Management: batch pipeline.

Processes one pre-recorded video per lane (sample_videos/lane1.mp4 ... lane4.mp4),
prints the adaptive signal schedule, saves annotated videos to outputs/ and opens
the junction simulator (outputs/traffic_light_simulator.html) in the browser.

    python main.py                 # all lanes found in sample_videos/
    python main.py --no-browser    # don't open the simulator at the end

For the live web control centre, run `python app.py` instead.
"""
import argparse
import glob
import os
import sys
import time
import webbrowser

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dashboard_generator import generate_dashboard  # noqa: E402
from density_estimator import classify_density, count_by_type  # noqa: E402
from incident_detector import IncidentDetector  # noqa: E402
from signal_optimizer import compute_signal_plan  # noqa: E402
from vehicle_detector import VehicleDetector  # noqa: E402

DIRECTIONS = ["North", "East", "South", "West"]
DETECT_EVERY = 5          # run YOLO on every 5th frame for CPU speed


def process_lane(detector: VehicleDetector, path: str, out_path: str) -> dict:
    """Run detection on one lane video and write the annotated copy."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    incidents = IncidentDetector()
    dets, flagged, counts, incident_events = [], set(), [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % DETECT_EVERY == 0:
            dets = detector.detect(frame)
            flagged, new = incidents.update(dets, frame_idx, h)
            incident_events += new
            counts.append((len(dets), count_by_type(dets)))
        out = detector.draw(frame, dets, flagged)
        density = classify_density(len(dets))
        cv2.rectangle(out, (0, 0), (300, 62), (20, 20, 20), -1)
        cv2.putText(out, f"Vehicles: {len(dets)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, f"Density: {density}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        if incidents.active():
            cv2.rectangle(out, (0, h - 44), (w, h), (40, 40, 200), -1)
            cv2.putText(out, "INCIDENT DETECTED", (12, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(out)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"\r    {frame_idx}/{total} frames", end="", flush=True)
    print(f"\r    {frame_idx}/{total} frames")
    cap.release()
    writer.release()

    # the lane's vehicle count = average vehicles in view over the clip (rounded)
    avg = round(sum(c for c, _ in counts) / max(len(counts), 1))
    by_type_avg = {k: round(sum(t[k] for _, t in counts) / max(len(counts), 1))
                   for k in ("car", "motorcycle", "bus", "truck")}
    return {"count": avg, "peak": max((c for c, _ in counts), default=0), "by_type": by_type_avg,
            "vehicles_counted": incidents.vehicles_counted, "incident": bool(incident_events),
            "incident_events": len(incident_events), "output": out_path}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=os.path.join(ROOT, "sample_videos"))
    ap.add_argument("--outputs", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    paths = [p for p in (os.path.join(args.videos, f"lane{i}.mp4") for i in range(1, 5)) if os.path.exists(p)]
    if not paths:
        sys.exit(f"No lane videos found. Put lane1.mp4 ... lane4.mp4 in {args.videos}")
    os.makedirs(args.outputs, exist_ok=True)

    print("\nLoading YOLOv8n ...")
    detector = VehicleDetector()
    results = []
    t0 = time.time()
    for path in paths:
        lane_no = int(os.path.basename(path)[4])
        print(f"\n[Lane {lane_no}] {os.path.basename(path)}")
        r = process_lane(detector, path, os.path.join(args.outputs, f"lane{lane_no}_annotated.mp4"))
        r.update(id=f"lane{lane_no}", name=f"Lane {lane_no}", direction=DIRECTIONS[(lane_no - 1) % 4],
                 density=classify_density(r["count"]))
        results.append(r)

    plan = compute_signal_plan({r["id"]: r["count"] for r in results})
    for r in results:
        r["green_time"] = plan[r["id"]]

    line = "-" * 118
    print(f"\n{line}\n  ADAPTIVE SIGNAL SCHEDULE   (green = 10 s + 2 s x vehicles, 10-60 s; incident lanes skipped)\n{line}")
    print(f"  {'Lane':<16}{'Vehicles':>9}{'Density':>10}{'Green':>8}  {'Incident':<10}{'Cars':>6}{'Bikes':>7}{'Buses':>7}"
          f"{'Trucks':>8}  Output video")
    print(line)
    for r in results:
        t = r["by_type"]
        green = "skip" if r["incident"] else f"{r['green_time']} s"
        print(f"  {r['name'] + ' (' + r['direction'] + ')':<16}{r['count']:>9}{r['density']:>10}{green:>8}  "
              f"{'YES' if r['incident'] else 'no':<10}{t['car']:>6}{t['motorcycle']:>7}{t['bus']:>7}{t['truck']:>8}  "
              f"{os.path.relpath(r['output'], ROOT)}")
    cycle = sum(r["green_time"] + 3 for r in results if not r["incident"])
    print(f"{line}\n  Full cycle: {cycle} s   ·   processed in {time.time() - t0:.0f} s\n")

    html = generate_dashboard([{k: r[k] for k in ("id", "name", "direction", "count", "density", "by_type",
                                                  "green_time", "incident")} for r in results],
                              os.path.join(args.outputs, "traffic_light_simulator.html"))
    print(f"Simulator saved to {os.path.relpath(html, ROOT)}")
    if not args.no_browser:
        webbrowser.open("file://" + os.path.abspath(html))


if __name__ == "__main__":
    main()
