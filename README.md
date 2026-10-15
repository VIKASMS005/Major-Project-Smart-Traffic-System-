# AI-Enabled Intelligent System for Smart Road Traffic Management

Final-year project (BIS685), Dept. of ISE, Sir M. Visvesvaraya Institute of Technology.
Team: Sandesh K, Vinath J Shetty, Nishchith D N, Vikas M S. Guide: Ms. Manjula Maheshwari V.

**SmartSignal** is a traffic control system for a 4-way junction. Four camera feeds, one per approach,
are analysed with a pretrained YOLOv8 model. The system counts and classifies vehicles, estimates
congestion on each approach, gives each approach a green time that matches its traffic, and detects
stopped vehicles (possible accidents or blockages). An approach with an incident is skipped in the signal
cycle until it clears.

Everything runs on a normal laptop CPU. No GPU, camera hardware or model training is needed.

## Two ways to run it

### 1. Web control centre (live)

```bash
python app.py
```

Then open **http://localhost:5000**.

| Page | What it shows |
|---|---|
| **Control Center** | Four live camera feeds with detections, an animated top-down junction with working signals and countdown, the adaptive signal plan, and live incident alerts |
| **Analytics** | Vehicles per approach over time, vehicle mix, green time given to each approach, vehicles counted, and waiting time compared with a fixed-time signal |
| **Signal Timing** | The timing rule, the upcoming phases, and a log of every green phase issued |
| **Incidents** | Every incident with a snapshot, the vehicle type, how long it was stopped, and whether it is still active |
| **Settings** | Density thresholds, green-time rule, yellow and all-red times, detection confidence, and incident sensitivity. Changes apply live |

### 2. Batch pipeline (`main.py`)

```bash
python main.py
```

1. Processes `sample_videos/lane1.mp4` … `lane4.mp4` (any that exist), running detection on every 5th frame.
2. Writes annotated videos to `outputs/laneN_annotated.mp4`, with a vehicle count, the density level and an `INCIDENT DETECTED` banner.
3. Prints the adaptive signal schedule in the terminal: vehicles, density, green time, incident flag, a per-class breakdown and the output video path for each lane.
4. Generates `outputs/traffic_light_simulator.html`, a self-contained animated junction (no server or internet needed), and opens it in your browser.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/VIKASMS005/Major-Project-Smart-Traffic-System-.git
cd Major-Project-Smart-Traffic-System-
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

The first install downloads PyTorch, which takes a few minutes. The YOLOv8n weights are included in `models/`.

## Using your own videos

Put up to four clips named `lane1.mp4` … `lane4.mp4` in `sample_videos/`. Lane 1 is shown as the
North approach, Lane 2 as East, Lane 3 as South and Lane 4 as West. Good free sources of traffic footage:

* Pexels: https://www.pexels.com/search/videos/traffic/
* Pixabay: https://pixabay.com/videos/search/traffic/
* Intel IoT sample videos: https://github.com/intel-iot-devkit/sample-videos

Overhead or elevated camera angles work best.

## Code map (report methodology → code)

| Report chapter / section | File | What it does |
|---|---|---|
| 4.3 Data acquisition | `sample_videos/`, `src/live_engine.py`, `main.py` | Reads one video per approach, plays it in real time and loops it |
| 4.4 Data preprocessing | `src/vehicle_detector.py` → `preprocess()` | Gaussian denoising and CLAHE contrast normalisation |
| 4.5 Traffic detection | `src/vehicle_detector.py` | YOLOv8n (COCO) detects car, motorcycle, bus and truck, and draws labelled boxes |
| 4.6 Traffic analysis | `src/density_estimator.py` | Low / Medium / High density (≤5, ≤12, 13+) and counts per vehicle type |
| Objective 3: incident detection | `src/incident_detector.py` | Nearest-neighbour tracking. A vehicle stationary for 90+ frames is flagged |
| 4.7 Signal optimisation | `src/signal_optimizer.py` | `green = 10 s + 2 s × vehicles`, clamped to 10–60 s. Live signal state machine that skips blocked lanes |
| 4.7 (comparison) | `src/simulation.py` | Simulated junction comparing fixed-time, adaptive and reinforcement-learning (Q-learning) control |
| Output layer | `app.py`, `templates/`, `static/`, `src/dashboard_generator.py` | Web control centre and standalone simulator |

## Project structure

```
├── app.py                    # web control centre (Flask)
├── main.py                   # batch pipeline + standalone simulator
├── requirements.txt
├── models/                   # yolov8n.pt, trained Q-table for the RL comparison
├── sample_videos/            # lane1.mp4 ... lane4.mp4
├── outputs/                  # annotated videos, simulator HTML, incident snapshots
├── src/
│   ├── vehicle_detector.py
│   ├── density_estimator.py
│   ├── signal_optimizer.py
│   ├── incident_detector.py
│   ├── dashboard_generator.py
│   ├── live_engine.py
│   └── simulation.py
├── templates/                # web pages
└── static/                   # CSS, junction animation, Chart.js (bundled, works offline)
```

## Known limitations

* The incident rule is a heuristic, so a bus waiting at a bus stop is also reported as stationary.
  A real deployment would mask bus bays and parking areas out of the detection zone.
* Vehicle counts depend on camera angle. Very distant vehicles are small and are sometimes missed.
