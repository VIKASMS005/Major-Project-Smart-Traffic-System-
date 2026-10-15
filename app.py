"""
AI-Enabled Intelligent System for Smart Road Traffic Management - demo dashboard.

Run:  streamlit run app.py
"""
import glob
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from simulation import (AdaptiveController, FixedTimeController,  # noqa: E402
                        QLearningController, simulate, train_rl)

st.set_page_config(page_title="Smart Traffic Management", page_icon="🚦", layout="wide")
st.title("🚦 AI-Enabled Intelligent System for Smart Road Traffic Management")
st.caption("YOLOv8 vehicle detection · density estimation · adaptive signal control · "
           "reinforcement learning · incident detection")


@st.cache_resource
def load_analyzer(model, imgsz, conf, n_lanes, stall_s):
    from detector import Config, TrafficAnalyzer
    return TrafficAnalyzer(Config(model_path=os.path.join(ROOT, "models", model),
                                  imgsz=imgsz, conf=conf, n_lanes=n_lanes, stall_seconds=stall_s))


@st.cache_resource
def load_agent():
    q_path = os.path.join(ROOT, "results", "signal", "q_table.npy")
    agent = QLearningController()
    if os.path.exists(q_path):
        agent.q = np.load(q_path)
        agent.training = False
    else:
        agent, _ = train_rl(episodes=200)
    return agent


tab_live, tab_sim, tab_results, tab_about = st.tabs(
    ["📹 Live detection", "🚥 Signal control simulation", "📊 Results", "ℹ️ How it works"])

# ---------------------------------------------------------------- live tab
with tab_live:
    left, right = st.columns([1, 3])
    with left:
        samples = sorted(glob.glob(os.path.join(ROOT, "data", "*.mp4")))
        choice = st.selectbox("Traffic video", [os.path.basename(s) for s in samples] + ["Upload my own"])
        upload = None
        if choice == "Upload my own":
            upload = st.file_uploader("Video file", type=["mp4", "avi", "mov"])
        model = st.radio("YOLO model", ["yolov8n.pt", "yolov8s.pt"], index=0, horizontal=True,
                         help="n = fastest, s = more accurate on small / distant vehicles")
        imgsz = st.select_slider("Inference size (px)", [480, 640, 800, 960], value=640)
        conf = st.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
        n_lanes = st.radio("Lanes (frame split)", [1, 2, 3, 4], index=1, horizontal=True)
        stall_s = st.slider("Incident alert after vehicle stationary for (s)", 2.0, 20.0, 6.0, 1.0)
        max_frames = st.number_input("Max frames to process (0 = all)", 0, 100000, 300, 50)
        skip = st.slider("Show every Nth frame (faster)", 1, 5, 2)
        go = st.button("▶ Run detection", type="primary", use_container_width=True)
    with right:
        frame_box = st.empty()
        metric_box = st.empty()
        alert_box = st.empty()

    if go:
        if choice == "Upload my own":
            if upload is None:
                st.warning("Upload a video first.")
                st.stop()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.name)[1])
            tmp.write(upload.read())
            path = tmp.name
        else:
            path = os.path.join(ROOT, "data", choice)
        analyzer = load_analyzer(model, imgsz, conf, n_lanes, stall_s)
        rows = []
        prog = st.progress(0.0, "Processing...")
        total = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
        limit = max_frames or total
        for i, (frame, stats) in enumerate(analyzer.frames(path, max_frames or None)):
            rows.append(stats)
            if i % skip == 0:
                frame_box.image(frame[:, :, ::-1], use_container_width=True)
                cols = metric_box.columns(n_lanes + 2)
                cols[0].metric("Vehicles in frame", stats["vehicles"])
                cols[1].metric("Unique vehicles", stats["unique_vehicles"])
                for k in range(n_lanes):
                    cols[k + 2].metric(f"Lane {k + 1} · {stats[f'lane{k + 1}_density']}",
                                       f"{stats[f'lane{k + 1}_green_s']} s green",
                                       f"{stats[f'lane{k + 1}_count']} vehicles", delta_color="off")
                if analyzer.alerts:
                    a = analyzer.alerts[-1]
                    alert_box.error(f"⚠️ t={a['time_s']}s · {a['type']} · {a['vehicle']} #{a['track_id']}")
            prog.progress(min((i + 1) / limit, 1.0), f"Frame {i + 1}/{limit}")
        prog.empty()
        df = pd.DataFrame(rows)
        st.subheader("Session summary")
        c = st.columns(4)
        c[0].metric("Frames", len(df))
        c[1].metric("Unique vehicles", int(df.unique_vehicles.iloc[-1]) if len(df) else 0)
        c[2].metric("Avg inference", f"{df.infer_ms.mean():.0f} ms/frame")
        c[3].metric("Incident alerts", len(analyzer.alerts))
        st.line_chart(df.set_index("time_s")[[f"lane{k + 1}_count" for k in range(n_lanes)]])
        st.bar_chart(df[["car", "motorcycle", "bus", "truck"]].sum())
        if analyzer.alerts:
            st.dataframe(pd.DataFrame(analyzer.alerts), use_container_width=True)
        st.download_button("Download per-frame CSV", df.to_csv(index=False), "traffic_stats.csv")

# ----------------------------------------------------------------- sim tab
with tab_sim:
    st.markdown("A 4-way junction (North, South, East, West) is simulated second by second. "
                "The same arriving traffic is handled by three signal controllers.")
    c1, c2, c3 = st.columns(3)
    scenario = c1.selectbox("Traffic scenario", ["low", "medium", "high", "rush"], index=3)
    fixed_green = c2.slider("Fixed-time green per phase (s)", 10, 60, 30, 5)
    seed = c3.number_input("Random seed", 0, 9999, 7)
    if st.button("▶ Run simulation", type="primary"):
        controllers = [FixedTimeController(fixed_green), AdaptiveController(), load_agent()]
        res, queues = [], {}
        for c in controllers:
            sim = simulate(c, scenario, seed=seed)
            res.append({"Controller": c.name, **sim.metrics()})
            queues[c.name] = np.array(sim.queue_log).sum(axis=1)
        df = pd.DataFrame(res).set_index("Controller")
        base = df.loc["Fixed-time", "avg_wait_s"]
        cols = st.columns(3)
        for col, (name, r) in zip(cols, df.iterrows()):
            delta = None if name == "Fixed-time" else f"{100 * (r.avg_wait_s - base) / base:.0f}% vs fixed"
            col.metric(f"{name}: avg wait", f"{r.avg_wait_s:.1f} s", delta, delta_color="inverse")
        st.dataframe(df.round(2), use_container_width=True)
        qdf = pd.DataFrame(queues)
        qdf.index = qdf.index / 60
        qdf.index.name = "minute"
        st.markdown("**Vehicles waiting at the junction over time**")
        st.line_chart(qdf)

# ------------------------------------------------------------- results tab
with tab_results:
    rdir = os.path.join(ROOT, "results")
    summary = os.path.join(rdir, "summary.md")
    if not os.path.exists(summary):
        st.info("Run `python run_experiments.py` to generate the full results.")
    else:
        st.markdown(open(summary).read())
        st.subheader("Signal control charts")
        imgs = sorted(glob.glob(os.path.join(rdir, "signal", "*.png")))
        for i in range(0, len(imgs), 2):
            cols = st.columns(2)
            for col, p in zip(cols, imgs[i:i + 2]):
                col.image(p, use_container_width=True)
        st.subheader("Detection outputs")
        for vid in sorted(glob.glob(os.path.join(rdir, "detection", "*_annotated.mp4"))):
            name = os.path.basename(vid).replace("_annotated.mp4", "")
            st.markdown(f"**{name}**")
            cols = st.columns([3, 2])
            cols[0].video(vid)
            for p in sorted(glob.glob(os.path.join(rdir, "detection", f"{name}_*.png"))):
                cols[1].image(p, use_container_width=True)
            js = os.path.join(rdir, "detection", f"{name}_summary.json")
            if os.path.exists(js):
                with st.expander("Summary JSON"):
                    st.json(json.load(open(js)))

# --------------------------------------------------------------- about tab
with tab_about:
    st.markdown("""
**Pipeline** (same as the block diagram in the report)

1. **Data acquisition**: traffic camera / CCTV video or uploaded footage.
2. **Preprocessing**: Gaussian denoising and CLAHE contrast normalisation of each frame.
3. **Detection layer**: YOLOv8 (a CNN-based single-pass detector) finds cars, motorcycles, buses and trucks; ByteTrack gives each vehicle a persistent ID.
4. **Traffic analysis layer**: vehicles are counted per lane; each lane's density is classed LOW / MEDIUM / HIGH.
5. **Incident detection**: a tracked vehicle that stays almost still for longer than the threshold raises an alert (possible accident / breakdown / blockage).
6. **Signal optimisation layer**:
   * *Adaptive*: green time = min green + seconds-per-vehicle × queue, clipped to 10–60 s.
   * *Reinforcement learning*: a Q-learning agent observes the queue density on both approaches and picks the green duration that minimises total waiting time; it is trained over hundreds of simulated hours.
7. **Output layer**: annotated video, live metrics, alerts, CSV logs and comparison charts.
""")
