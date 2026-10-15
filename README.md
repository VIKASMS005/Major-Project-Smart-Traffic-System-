# AI-Enabled Intelligent System for Smart Road Traffic Management

Final-year project (BIS685), Dept. of ISE, Sir M. Visvesvaraya Institute of Technology.
Team: Sandesh K, Vinath J Shetty, Nishchith D N, Vikas M S. Guide: Ms. Manjula Maheshwari V.

**SmartSignal** is an AI traffic control system that works from a single traffic video. Upload a video from
a CCTV or elevated camera and the system:

1. **Finds the lanes by itself.** It tracks the vehicles in the first seconds of the video and learns, from
   the paths they take, how many traffic directions there are and how many lanes each one has. No lanes
   are drawn by hand.
2. **Detects and classifies every vehicle** as a 2-wheeler, 3-wheeler (auto-rickshaw), 4-wheeler or
   heavy vehicle (6+ wheels: buses, trucks), and assigns it to its lane.
3. **Measures density per lane** in passenger car units (PCU), so a bus counts for more than a scooter,
   and counts traffic flow (vehicles per minute) in each lane.
4. **Controls the signals.** Each traffic direction (or each lane, if the video shows one direction) gets a
   green time that follows its measured load. A vehicle that stops while other traffic keeps moving is
   reported as an incident and its direction is skipped until it clears.
5. **Compares with a fixed-time signal** by simulating both controllers with the traffic measured in the
   video. When the measured demand is more than any signal could clear, it keeps the measured split between
   approaches and scales the total down so the comparison stays meaningful.

Everything runs on a normal laptop CPU. No GPU, camera hardware or model training is needed.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/VIKASMS005/Major-Project-Smart-Traffic-System-.git
cd Major-Project-Smart-Traffic-System-
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

The first install downloads PyTorch, which takes a few minutes. The detection models are included in `models/`.

## Run the web control centre

```bash
python app.py
```

Open **http://localhost:5000**, then upload a traffic video (or click *Analyse* on the sample clip).

| Page | What it shows |
|---|---|
| **Upload Video** | Drag-and-drop upload, and every video analysed before |
| **Control Center** | The live analysed video with the learned lanes, boxes and per-lane counts; one signal per approach (or lane) with countdown; a junction animation when approaches come from different sides; per-lane table; vehicle mix; live incident alerts |
| **Analytics** | Load over time per signal group, vehicle mix, flow per lane by vehicle type, average green time given, adaptive vs fixed-time waiting time, and the learned lane map |
| **Signal Timing** | The timing rule with the current settings, the PCU table, the phase order and a log of every green phase issued |
| **Incidents** | Every stationary-vehicle incident with a snapshot, lane, vehicle type and duration |
| **Settings** | Density thresholds, green-time rule, yellow and all-red times, detection confidence, incident sensitivity and lane-learning time. Changes apply live; *Re-learn lanes* runs lane discovery again |

While the lanes are being learned (the first 30 seconds of video), the page shows the vehicle paths being collected.
After that the video plays in real time and loops.

## Run the batch pipeline

```bash
python main.py --video sample_videos/indian_junction_sample.mp4
```

Processes the whole video and prints a per-lane report (2W / 3W / 4W / HV counts and flow) and the adaptive
signal plan. It saves `outputs/<video>_annotated.mp4` and incident snapshots, and opens
`outputs/traffic_light_simulator.html`, a self-contained signal simulator that needs no server or internet.

## Vehicle classes

| Class | Vehicles | PCU (IRC:106-1990) |
|---|---|---|
| 2-Wheeler | motorcycles, scooters, bicycles | 0.5 |
| 3-Wheeler | auto-rickshaws | 1.0 |
| 4-Wheeler | cars, jeeps, vans | 1.0 |
| Heavy Vehicle | buses and trucks (6 or more wheels) | 3.0 |

Two pretrained models work together, with no training:

* `models/yolov8n.pt` (COCO) finds and boxes the vehicles. It has the best recall for cars, buses, trucks
  and motorcycles, but no auto-rickshaw class.
* `models/traffic_yoloe_s.pt` is YOLOE, an open-vocabulary YOLO, with the vocabulary *motorcycle, scooter,
  bicycle, auto rickshaw, car, bus, truck* built in. Where it sees an auto-rickshaw, that label replaces the
  COCO label, and autos the COCO model missed are added.

## How lanes are found (`src/lane_detector.py`)

1. **Trajectories.** Each vehicle is tracked (`src/tracker.py`) and its ground-contact point (bottom centre of
   the box) is recorded over time.
2. **Directions.** Trajectories are grouped by their direction of motion. Each group is one traffic direction
   (approach). It is named by where traffic comes from, taking the top of the image as north.
3. **Lanes.** Inside a direction, trajectories are clustered by their sideways position. Distances are measured
   in vehicle widths at each depth of the image, so perspective (far vehicles look smaller) does not matter.
4. **Lane shapes.** Each cluster gives a smoothed centre line and a width profile, which become the lane
   polygon drawn on the video and used to assign vehicles to lanes.
5. **Waiting queues.** Traffic held at a red light hardly moves, so its lanes cannot be found from motion.
   Where many vehicles stand outside every lane found so far, that area becomes an approach of its own,
   split into lanes about 1.4 vehicle widths wide.

Lanes that no vehicle uses during the learning period are not found, so a longer learning time (Settings)
helps on quiet roads.

## Using your own videos

Any traffic video works, best from a fixed, elevated camera that sees several seconds of each vehicle's path.
Indian traffic footage shows the 2-wheeler and 3-wheeler classes best. Free sources:

* Pexels: https://www.pexels.com/search/videos/indian%20traffic/
* Pixabay: https://pixabay.com/videos/search/traffic%20india/

## Code map (report methodology → code)

| Report chapter / section | File | What it does |
|---|---|---|
| 4.3 Data acquisition | `app.py` (upload), `src/live_engine.py` | Receives the uploaded video and plays it in real time |
| 4.4 Data preprocessing | `src/vehicle_detector.py` → `preprocess()` | Gaussian denoising and CLAHE contrast normalisation |
| 4.5 Traffic detection | `src/vehicle_detector.py`, `src/tracker.py` | YOLOE detection, 2W/3W/4W/HV classes, multi-object tracking |
| 4.6 Traffic analysis | `src/lane_detector.py`, `src/density_estimator.py` | Automatic lane discovery; PCU density per lane (Low / Medium / High) |
| Objective 3: incident detection | `src/live_engine.py` → `_analyse()` | A vehicle that was driving, then stays stopped longer than the set time while traffic in its direction moves past |
| 4.7 Signal optimisation | `src/signal_optimizer.py` | `green = base + k × PCU`, clamped; live state machine that skips blocked groups |
| 4.7 (comparison) | `src/simulation.py` | Fixed-time vs adaptive at the measured arrival rates; also a 4-way junction model with a Q-learning agent |
| Output layer | `app.py`, `templates/`, `static/`, `src/dashboard_generator.py` | Web control centre and standalone simulator |

## Project structure

```
├── app.py                    # web control centre (Flask)
├── main.py                   # batch pipeline for one video + standalone simulator
├── requirements.txt
├── models/                   # traffic_yoloe_s.pt (default), yolov8n.pt (fallback), RL Q-table
├── sample_videos/            # indian_junction_sample.mp4 (Indian junction, autos, buses, bikes)
├── uploads/                  # videos uploaded through the website (created on first upload)
├── outputs/                  # annotated videos, simulator HTML, incident snapshots
├── src/
│   ├── vehicle_detector.py
│   ├── tracker.py
│   ├── lane_detector.py
│   ├── density_estimator.py
│   ├── signal_optimizer.py
│   ├── live_engine.py
│   ├── simulation.py
│   ├── incident_detector.py  # earlier stand-alone incident heuristic (kept for reference)
│   └── dashboard_generator.py
├── templates/                # web pages
└── static/                   # CSS, junction animation, Chart.js (bundled, works offline)
```

## Known limitations

* Detection takes about 100–150 ms per frame on a laptop CPU, so the live view analyses about 6–8 frames
  per second and skips the rest. Very fast or very small vehicles can be missed.
* Lane discovery needs vehicles to move through the view during the learning period. A parked or empty lane
  is not found, and a camera that pans or zooms breaks the learned layout.
* Approach names assume the top of the image is north.
* A bus standing at a bus stop while traffic passes it is reported as a stationary vehicle. A real
  deployment would mask bus bays and parking areas.
* The 3-wheeler class comes from an open-vocabulary model, not from training on Indian data, so its accuracy
  is lower than for cars.
