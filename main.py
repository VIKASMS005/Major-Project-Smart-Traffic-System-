"""
AI-Enabled Intelligent System for Smart Road Traffic Management: batch pipeline.

Analyses one traffic video end to end, without the web server:

1. Learns the lanes and traffic directions from the vehicles in the video.
2. Detects, tracks and classifies every vehicle (2-wheeler, 3-wheeler,
   4-wheeler, heavy vehicle) and assigns it to a lane.
3. Prints a per-lane traffic report and the adaptive signal plan.
4. Saves the annotated video and incident snapshots to outputs/, and opens a
   self-contained signal simulator (outputs/traffic_light_simulator.html).

    python main.py --video path/to/traffic.mp4
    python main.py --video sample_videos/indian_junction_sample.mp4 --no-browser

For the live web control centre, run `python app.py` instead.
"""
import argparse
import os
import sys
import time
import webbrowser

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dashboard_generator import generate_dashboard  # noqa: E402
from density_estimator import classify_density, pcu_load  # noqa: E402
from live_engine import TrafficEngine  # noqa: E402
from signal_optimizer import compute_signal_plan  # noqa: E402
from simulation import compare_controllers  # noqa: E402
from vehicle_detector import CATEGORIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="traffic video to analyse")
    ap.add_argument("--outputs", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--learn-seconds", type=float, default=30, help="video seconds used to learn the lanes")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")
    os.makedirs(args.outputs, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_video = os.path.join(args.outputs, f"{stem}_annotated.mp4")

    print("\nLoading the detection model ...")
    engine = TrafficEngine(os.path.join(args.outputs, "incidents"), {"learn_seconds": args.learn_seconds},
                           realtime=False)
    print(f"Model: {engine.detector.model_name}")
    writer = {}

    def on_frame(img):
        if "w" not in writer:
            h, w = img.shape[:2]
            writer["w"] = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), engine.fps, (w, h))
        writer["w"].write(img)

    engine.on_frame = on_frame
    t0 = time.time()
    engine.load(args.video)
    stage = None
    while engine.thread.is_alive():
        s = engine.status
        if s != stage:
            print({"learning": "\n[1/2] Learning lanes from vehicle paths ...",
                   "live": "\n[2/2] Detecting, tracking and counting ..."}.get(s, ""))
            stage = s
        if s == "learning":
            print(f"\r      {engine.progress:5.1f}%", end="", flush=True)
        elif s == "live" and engine.total_frames:
            pct = 100 * engine.video_time / (engine.total_frames / engine.fps)
            print(f"\r      {pct:5.1f}%  ({engine.vehicles_counted} vehicles counted)", end="", flush=True)
        time.sleep(0.5)
    if "w" in writer:
        writer["w"].release()
    if engine.status == "error":
        sys.exit(f"\nError: {engine.message}")
    print()

    snap = engine.snapshot()
    duration = max(engine.video_time, 1e-6)
    # average load of each signal group over the whole video
    avg_pcu = {g["id"]: float(np.mean([h["groups"][g["id"]]["pcu"] for h in engine.history] or [0]))
               for g in snap["groups"]}
    avg_veh = {g["id"]: float(np.mean([h["groups"][g["id"]]["vehicles"] for h in engine.history] or [0]))
               for g in snap["groups"]}
    plan = compute_signal_plan({gid: round(v) for gid, v in avg_pcu.items()}, **engine._timing())
    incident_groups = {i["group"] for i in engine.incidents}
    st = engine.settings

    line = "-" * 112
    print(f"\n{line}\n  VIDEO: {os.path.basename(args.video)}   {snap['video']['width']}x{snap['video']['height']}, "
          f"{duration:.0f} s   ·   {len(snap['lanes'])} lanes in {snap['n_streams']} direction(s) detected\n{line}")
    print(f"  {'Lane':<6}{'Direction':<22}{'2W':>6}{'3W':>6}{'4W':>6}{'HV':>6}{'Total':>8}{'Flow/min':>10}")
    print(line)
    for l in snap["lanes"]:
        c = l["counted"]
        print(f"  {l['id']:<6}{l['stream']:<22}" + "".join(f"{c[k]:>6}" for k in CATEGORIES) +
              f"{l['counted_total']:>8}{l['counted_total'] * 60 / duration:>10.1f}")
    print(f"{line}\n  ADAPTIVE SIGNAL PLAN   (green = {st['base_time']} s + {st['per_vehicle_time']} s x PCU, "
          f"{st['min_green']}-{st['max_green']} s; groups with an incident are skipped)\n{line}")
    print(f"  {'Signal group':<22}{'Lanes':<14}{'Avg veh':>8}{'Avg PCU':>9}{'Density':>10}{'Green':>8}  Incident")
    print(line)
    results = []
    for g in snap["groups"]:
        dens = classify_density(avg_pcu[g["id"]], st["low_max"], st["medium_max"])
        inc = g["id"] in incident_groups
        print(f"  {g['name']:<22}{', '.join(g['lanes']):<14}{avg_veh[g['id']]:>8.1f}{avg_pcu[g['id']]:>9.1f}{dens:>10}"
              f"{('skip' if inc else str(plan[g['id']]) + ' s'):>8}  {'YES' if inc else 'no'}")
        by = {k: 0 for k in CATEGORIES}
        for lid in g["lanes"]:
            for k, n in engine.lanes[lid].counted.items():
                by[k] += n
        results.append({"id": g["id"], "name": g["name"].replace(" approach", ""), "direction": g["compass"],
                        "count": round(avg_veh[g["id"]]), "pcu": round(avg_pcu[g["id"]], 1), "density": dens,
                        "by_type": by, "green_time": plan[g["id"]], "incident": inc})
    cycle = sum(r["green_time"] + st["yellow_time"] + st["all_red_time"] for r in results if not r["incident"])
    print(f"{line}\n  Vehicles counted: {engine.vehicles_counted}  (" +
          ", ".join(f"{k} {v}" for k, v in snap["mix_counted"].items()) +
          f")   ·   PCU load of the mix: {pcu_load(snap['mix_counted'])}")
    print(f"  Incidents: {len(engine.incidents)}   ·   full signal cycle: {cycle} s   ·   processed in {time.time() - t0:.0f} s")
    print(f"  Detection: {engine.detector.model_name}, {engine.det_count} frames analysed, "
          f"{engine.infer_total_ms / max(engine.det_count, 1):.0f} ms per frame on average")
    print(f"  Annotated video: {os.path.relpath(out_video, ROOT)}")

    rates = [sum(engine.lanes[l].counted_total if hasattr(engine.lanes[l], "counted_total") else
                 sum(engine.lanes[l].counted.values()) for l in g["lanes"]) / duration for g in snap["groups"]]
    if any(rates):
        cmp = compare_controllers(rates, [len(g["lanes"]) for g in snap["groups"]], engine._timing())
        scaled = (f" (measured demand is above what any signal can clear, flow ratio {cmp['flow_ratio']}; "
                  f"scaled to {cmp['demand_scale'] * 100:.0f}% keeping the measured split)") if cmp["demand_scale"] < 1 else ""
        print(f"{line}\n  ADAPTIVE vs FIXED-TIME  ({cmp['duration_s'] // 60} min queue simulation, arrivals per min: " +
              ", ".join(f"{g['name']} {r * 60:.1f}" for g, r in zip(snap["groups"], rates)) + f"){scaled}")
        print(f"  Average wait per vehicle: fixed-time ({cmp['fixed_green_s']} s green each) {cmp['fixed_wait_s']} s, "
              f"adaptive {cmp['adaptive_wait_s']} s, reduction {cmp['reduction_pct']}%\n{line}\n")

    html = generate_dashboard(results, os.path.join(args.outputs, "traffic_light_simulator.html"),
                              yellow=st["yellow_time"], title=os.path.basename(args.video))
    print(f"Simulator saved to {os.path.relpath(html, ROOT)}")
    if not args.no_browser:
        webbrowser.open("file://" + os.path.abspath(html))


if __name__ == "__main__":
    main()
