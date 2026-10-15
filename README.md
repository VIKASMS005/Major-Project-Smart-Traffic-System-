# AI-Enabled Intelligent System for Smart Road Traffic Management

Final-year project (BIS685), Dept. of ISE, Sir M. Visvesvaraya Institute of Technology.
Team: Sandesh K, Vinath J Shetty, Nishchith D N, Vikas M S. Guide: Ms. Manjula Maheshwari V.

A working prototype of the system described in the project report:

| Module (report section) | What the code does | File |
|---|---|---|
| Data acquisition (4.3) | Reads CCTV-style traffic video or an uploaded clip | `src/detector.py` |
| Preprocessing (4.4) | Gaussian denoise + CLAHE contrast normalisation | `src/detector.py` |
| Vehicle detection & classification (4.5) | YOLOv8 detects car / motorcycle / bus / truck, ByteTrack assigns IDs | `src/detector.py` |
| Traffic analysis & density (4.6) | Lane-wise counts, LOW / MEDIUM / HIGH density | `src/detector.py` |
| Incident detection (Objective 3) | Alert when a tracked vehicle stays stationary too long | `src/detector.py` |
| Signal optimisation (4.7) | Density-based adaptive timing + Q-learning RL agent, compared with fixed-time signals on a simulated 4-way junction | `src/simulation.py` |
| Output layer | Streamlit dashboard, annotated videos, CSV logs, charts | `app.py`, `run_experiments.py` |

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

YOLOv8 weights are in `models/` (`yolov8n.pt` fast, `yolov8s.pt` more accurate; Ultralytics downloads them if missing).
Sample videos are in `data/`.

## Run the demo dashboard

```bash
streamlit run app.py
```

Opens in the browser with four tabs: **Live detection**, **Signal control simulation**, **Results**, **How it works**.

## Regenerate all results (charts, tables, annotated videos)

```bash
python run_experiments.py              # everything
python run_experiments.py --signal     # only the signal-control simulation
python run_experiments.py --detect     # only video detection
```

Outputs go to `results/`; `results/summary.md` has every headline number.

## How the signal simulation works

* 4 approaches (N, S, E, W), 2 phases (NS green / EW green), 3 s yellow lost time per switch, 1-second steps.
* Vehicles arrive per lane as a Poisson process; four scenarios: low, medium, high, rush (heavy N-S corridor).
* A green lane discharges 0.5 vehicles/s (saturation flow 1800 veh/h).
* **Fixed-time**: 30 s green per phase. **Adaptive**: green = 10 s + 2.2 s × vehicles waiting per lane (10–60 s).
  **RL**: tabular Q-learning; state = (phase, NS density level, EW density level), action = green duration
  {10, 15, 20, 30, 45, 60 s}, reward = − waiting time accumulated, trained for 400 simulated hours.
* Metrics: average waiting time per vehicle, average / max queue, vehicles served. 5 random seeds per scenario.
