# Results summary

## Adaptive signal control (simulated 4-way junction, 1 hour, 5 seeds)

| Scenario | Controller | Avg wait (s) | Avg queue | Max queue | Vehicles served % |
|---|---|---|---|---|---|
| low | Fixed-time | 12.9 | 3.4 | 14 | 99.4 |
| low | Adaptive (density) | 7.8 | 2.1 | 10 | 99.6 |
| low | RL agent (Q-learning) | 7.4 | 2.0 | 9 | 99.7 |
| medium | Fixed-time | 16.6 | 8.0 | 26 | 99.4 |
| medium | Adaptive (density) | 10.9 | 5.3 | 18 | 99.7 |
| medium | RL agent (Q-learning) | 10.8 | 5.2 | 18 | 99.7 |
| high | Fixed-time | 61.5 | 41.7 | 77 | 97.9 |
| high | Adaptive (density) | 20.9 | 14.2 | 38 | 99.4 |
| high | RL agent (Q-learning) | 25.9 | 17.6 | 46 | 99.4 |
| rush | Fixed-time | 314.6 | 214.8 | 391 | 84.3 |
| rush | Adaptive (density) | 28.8 | 19.7 | 49 | 99.3 |
| rush | RL agent (Q-learning) | 18.8 | 12.8 | 37 | 99.7 |

- low: Adaptive (density) cuts average waiting time by 40.0% vs fixed-time
- low: RL agent (Q-learning) cuts average waiting time by 42.6% vs fixed-time
- medium: Adaptive (density) cuts average waiting time by 34.2% vs fixed-time
- medium: RL agent (Q-learning) cuts average waiting time by 34.5% vs fixed-time
- high: Adaptive (density) cuts average waiting time by 66.0% vs fixed-time
- high: RL agent (Q-learning) cuts average waiting time by 57.9% vs fixed-time
- rush: Adaptive (density) cuts average waiting time by 90.8% vs fixed-time
- rush: RL agent (Q-learning) cuts average waiting time by 94.0% vs fixed-time

## Vehicle detection on sample traffic videos (YOLOv8s, 960 px, CPU)

| Video | Frames | Processing FPS | Unique vehicles tracked | Avg vehicles/frame | Peak vehicles/frame | Mean confidence | Incident alerts |
|---|---|---|---|---|---|---|---|
| city_road.mp4 | 1190 | 6.3 | 143 | 10.9 | 23 | 0.64 | 2 |
| highway.mp4 | 984 | 5.6 | 157 | 8.9 | 16 | 0.68 | 0 |
