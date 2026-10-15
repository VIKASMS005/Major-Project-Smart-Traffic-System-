"""
Generates every result used in the report and slides:

  results/detection/   annotated videos, per-frame CSVs, alerts, screenshots, charts
  results/signal/      fixed vs adaptive vs RL comparison tables and charts
  results/summary.md   all headline numbers in one place

Usage:  python run_experiments.py            (everything)
        python run_experiments.py --signal   (simulation only, no YOLO needed)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from simulation import (AdaptiveController, FixedTimeController,  # noqa: E402
                        simulate, train_rl)

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
SCENARIOS = ["low", "medium", "high", "rush"]
SEEDS = [11, 22, 33, 44, 55]
COLORS = {"Fixed-time": "#9aa0a6", "Adaptive (density)": "#1a73e8",
          "RL agent (Q-learning)": "#e8710a"}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def signal_experiments(lines):
    out = os.path.join(RES, "signal")
    os.makedirs(out, exist_ok=True)
    print("Training RL agent...")
    agent, hist = train_rl(episodes=400, log=print)
    np.save(os.path.join(out, "q_table.npy"), agent.q)
    hist = pd.DataFrame(hist)
    hist.to_csv(os.path.join(out, "rl_training.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for sc in SCENARIOS:
        h = hist[hist.scenario == sc]
        ax.plot(h.episode, h.avg_wait_s.rolling(10, min_periods=1).mean(), label=sc)
    ax.set(xlabel="Training episode", ylabel="Avg waiting time (s)",
           title="RL agent learning curve (10-episode moving average)")
    ax.set_yscale("log")
    ax.legend(title="Scenario")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_learning_curve.png"))
    plt.close(fig)

    controllers = [FixedTimeController(), AdaptiveController(), agent]
    rows = []
    for sc in SCENARIOS:
        for c in controllers:
            for s in SEEDS:
                m = simulate(c, sc, seed=s).metrics()
                rows.append({"scenario": sc, "controller": c.name, "seed": s, **m})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "comparison_raw.csv"), index=False)
    summ = df.groupby(["scenario", "controller"], sort=False).agg(
        avg_wait_s=("avg_wait_s", "mean"), avg_queue=("avg_queue", "mean"),
        max_queue=("max_queue", "mean"), throughput=("throughput", "mean"),
        arrived=("arrived", "mean")).reset_index()
    summ["served_pct"] = 100 * summ.throughput / summ.arrived
    summ = summ.round(2)
    summ.to_csv(os.path.join(out, "comparison_summary.csv"), index=False)

    # grouped bar charts
    for metric, label in [("avg_wait_s", "Average waiting time per vehicle (s)"),
                          ("avg_queue", "Average queue length (vehicles)")]:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(SCENARIOS))
        for i, c in enumerate(controllers):
            v = [summ[(summ.scenario == sc) & (summ.controller == c.name)][metric].item()
                 for sc in SCENARIOS]
            bars = ax.bar(x + (i - 1) * 0.27, v, 0.27, label=c.name, color=COLORS[c.name])
            ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        ax.set_xticks(x, [s.capitalize() for s in SCENARIOS])
        ax.set(xlabel="Traffic scenario", ylabel=label, title=label)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"compare_{metric}.png"))
        plt.close(fig)

    # queue length over time, rush hour
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for c in controllers:
        sim = simulate(c, "rush", seed=SEEDS[0])
        q = np.array(sim.queue_log).sum(axis=1)
        ax.plot(np.arange(len(q)) / 60, q, label=c.name, color=COLORS[c.name], lw=1.2)
    ax.set(xlabel="Time (minutes)", ylabel="Vehicles waiting (all lanes)",
           title="Queue build-up during 1 hour of rush-hour traffic")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "queue_over_time_rush.png"))
    plt.close(fig)

    lines.append("## Adaptive signal control (simulated 4-way junction, 1 hour, 5 seeds)\n")
    lines.append("| Scenario | Controller | Avg wait (s) | Avg queue | Max queue | Vehicles served % |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in summ.iterrows():
        lines.append(f"| {r.scenario} | {r.controller} | {r.avg_wait_s:.1f} | {r.avg_queue:.1f} "
                     f"| {r.max_queue:.0f} | {r.served_pct:.1f} |")
    lines.append("")
    for sc in SCENARIOS:
        f = summ[(summ.scenario == sc) & (summ.controller == "Fixed-time")].avg_wait_s.item()
        for name in ["Adaptive (density)", "RL agent (Q-learning)"]:
            v = summ[(summ.scenario == sc) & (summ.controller == name)].avg_wait_s.item()
            lines.append(f"- {sc}: {name} cuts average waiting time by {100 * (f - v) / f:.1f}% vs fixed-time")
    lines.append("")


def to_h264(src, dst):
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)
        os.remove(src)
    else:
        os.replace(src, dst)


def detection_experiments(lines, max_frames=None):
    import cv2
    from detector import DENSITY_NAMES, Config, TrafficAnalyzer
    out = os.path.join(RES, "detection")
    os.makedirs(out, exist_ok=True)
    analyzer = TrafficAnalyzer(Config(model_path=os.path.join(ROOT, "models", "yolov8s.pt")))
    lines.append("## Vehicle detection on sample traffic videos (YOLOv8s, 960 px, CPU)\n")
    lines.append("| Video | Frames | Processing FPS | Unique vehicles tracked | Avg vehicles/frame "
                 "| Peak vehicles/frame | Mean confidence | Incident alerts |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for vid in sorted(os.listdir(os.path.join(ROOT, "data"))):
        if not vid.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        name = os.path.splitext(vid)[0]
        print("Detecting on", vid)
        tmp = os.path.join(out, f"{name}_tmp.mp4")
        rows, alerts, fps = analyzer.process_video(os.path.join(ROOT, "data", vid), tmp, max_frames)
        to_h264(tmp, os.path.join(out, f"{name}_annotated.mp4"))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"{name}_frames.csv"), index=False)
        pd.DataFrame(alerts).to_csv(os.path.join(out, f"{name}_alerts.csv"), index=False)

        # screenshots at busiest frame and two other points
        cap = cv2.VideoCapture(os.path.join(out, f"{name}_annotated.mp4"))
        picks = sorted({int(df.vehicles.idxmax()), len(df) // 4, len(df) // 2})
        for k, fi in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(os.path.join(out, f"{name}_shot{k + 1}.jpg"), fr)
        cap.release()

        # charts
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for c in [c for c in df.columns if c.endswith("_count")]:
            ax.plot(df.time_s, df[c], label=c.replace("_count", "").replace("lane", "Lane "), lw=1)
        ax.set(xlabel="Video time (s)", ylabel="Vehicles detected",
               title=f"Vehicles per lane over time: {vid}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}_counts.png"))
        plt.close(fig)

        cls = df[["car", "motorcycle", "bus", "truck"]].sum()
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar(cls.index, cls.values, color=["#34a853", "#fbbc04", "#1a73e8", "#a142f4"])
        ax.set(ylabel="Detections (all frames)", title=f"Vehicle classes: {vid}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}_classes.png"))
        plt.close(fig)

        dens = pd.concat([df[c] for c in df.columns if c.endswith("_density")])
        dshare = (dens.value_counts(normalize=True) * 100).reindex(DENSITY_NAMES).fillna(0)
        lines.append(f"| {vid} | {len(df)} | {fps:.1f} | {df.unique_vehicles.iloc[-1]} "
                     f"| {df.vehicles.mean():.1f} | {df.vehicles.max()} "
                     f"| {df.mean_conf.mean():.2f} | {len(alerts)} |")
        json.dump({"video": vid, "frames": len(df), "fps": fps,
                   "density_share_pct": dshare.round(1).to_dict(),
                   "class_totals": cls.to_dict(),
                   "avg_infer_ms": float(df.infer_ms.mean()),
                   "avg_green_s": {c: float(df[c].mean()) for c in df.columns if c.endswith("_green_s")}},
                  open(os.path.join(out, f"{name}_summary.json"), "w"), indent=2, default=float)
    lines.append("")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", action="store_true", help="only run the signal simulation")
    ap.add_argument("--detect", action="store_true", help="only run video detection")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    lines = ["# Results summary\n"]
    if not a.detect:
        signal_experiments(lines)
    if not a.signal:
        detection_experiments(lines, a.max_frames)
    name = "summary.md" if not (a.signal or a.detect) else f"summary_{'signal' if a.signal else 'detect'}.md"
    open(os.path.join(RES, name), "w").write("\n".join(lines))
    print("\n".join(lines))
