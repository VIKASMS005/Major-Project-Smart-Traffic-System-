"""
4-way intersection simulator used to compare signal control strategies.

Lanes: North, South, East, West. Two signal phases:
  phase 0 -> North-South green, phase 1 -> East-West green.
Each simulation step is 1 second. Vehicles arrive per lane following a
Poisson process whose rate varies over the day (peak / off-peak), and a
green lane discharges ~1 vehicle every SAT_HEADWAY seconds.

Controllers compared:
  * FixedTimeController   - conventional fixed cycle (baseline)
  * AdaptiveController    - green time proportional to detected queue (density)
  * QLearningController   - reinforcement learning agent (tabular Q-learning)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

LANES = ["North", "South", "East", "West"]
PHASE_LANES = {0: (0, 1), 1: (2, 3)}
SAT_FLOW = 0.5          # vehicles / second discharged per green lane
YELLOW = 3              # seconds lost on every phase switch
MIN_GREEN, MAX_GREEN = 10, 60


def arrival_rates(t: int, scenario: str) -> np.ndarray:
    """Arrival rate (veh/s) per lane at second t for a traffic scenario."""
    base = {
        "low":    np.array([0.08, 0.07, 0.06, 0.05]),
        "medium": np.array([0.16, 0.14, 0.10, 0.08]),
        "high":   np.array([0.24, 0.20, 0.14, 0.10]),
        # unbalanced: one heavy corridor, typical rush hour
        "rush":   np.array([0.30, 0.26, 0.07, 0.05]),
    }[scenario]
    # gentle oscillation so demand is not constant
    wave = 1 + 0.25 * np.sin(2 * np.pi * t / 900 + np.array([0, 0.5, 1.5, 2.0]))
    return base * wave


@dataclass
class Intersection:
    scenario: str = "medium"
    seed: int = 0
    queues: np.ndarray = field(default_factory=lambda: np.zeros(4))
    phase: int = 0
    t: int = 0
    total_wait: float = 0.0
    served: int = 0
    arrived: int = 0
    queue_log: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._carry = np.zeros(4)

    def _tick(self, green: bool):
        lam = arrival_rates(self.t, self.scenario)
        arrivals = self.rng.poisson(lam)
        self.queues += arrivals
        self.arrived += int(arrivals.sum())
        if green:
            for i in PHASE_LANES[self.phase]:
                self._carry[i] += SAT_FLOW
                out = min(int(self._carry[i]), int(self.queues[i]))
                self._carry[i] -= int(self._carry[i])
                self.queues[i] -= out
                self.served += out
        # every queued vehicle waits one more second
        self.total_wait += self.queues.sum()
        self.queue_log.append(self.queues.copy())
        self.t += 1

    def run_green(self, seconds: int):
        for _ in range(seconds):
            self._tick(green=True)

    def switch(self):
        for _ in range(YELLOW):
            self._tick(green=False)
        self.phase = 1 - self.phase

    def metrics(self) -> dict:
        q = np.array(self.queue_log)
        return {
            "avg_wait_s": self.total_wait / max(self.arrived, 1),
            "avg_queue": float(q.sum(axis=1).mean()),
            "max_queue": float(q.sum(axis=1).max()),
            "throughput": self.served,
            "arrived": self.arrived,
        }


class FixedTimeController:
    name = "Fixed-time"

    def __init__(self, green: int = 30):
        self.green = green

    def decide(self, sim: Intersection) -> int:
        return self.green


class AdaptiveController:
    """Density based: green time grows with the queue on the green approach."""
    name = "Adaptive (density)"

    def __init__(self, sec_per_vehicle: float = 2.2):
        self.k = sec_per_vehicle

    def decide(self, sim: Intersection) -> int:
        own = sum(sim.queues[i] for i in PHASE_LANES[sim.phase])
        return int(np.clip(MIN_GREEN + self.k * own / 2, MIN_GREEN, MAX_GREEN))


DENSITY_BINS = [3, 8, 15, 25]   # vehicle-count thresholds -> 5 RL density levels


def density_level(count: float) -> int:
    return int(np.searchsorted(DENSITY_BINS, count, side="right"))


class QLearningController:
    """Tabular Q-learning agent.

    State  : (phase, density level of NS approach, density level of EW approach)
    Action : green duration from GREEN_CHOICES for the current phase
    Reward : negative of total queue at the end of the action (less waiting = better)
    """
    name = "RL agent (Q-learning)"
    GREEN_CHOICES = [10, 15, 20, 30, 45, 60]

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, seed=0):
        self.q = np.zeros((2, 5, 5, len(self.GREEN_CHOICES)))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.random = random.Random(seed)
        self.training = True
        self._last = None

    def state(self, sim: Intersection):
        ns = sim.queues[0] + sim.queues[1]
        ew = sim.queues[2] + sim.queues[3]
        return (sim.phase, density_level(ns), density_level(ew))

    def decide(self, sim: Intersection) -> int:
        s = self.state(sim)
        if self.training and self.random.random() < self.epsilon:
            a = self.random.randrange(len(self.GREEN_CHOICES))
        else:
            a = int(np.argmax(self.q[s]))
        self._last = (s, a)
        return self.GREEN_CHOICES[a]

    def learn(self, sim: Intersection, reward: float):
        if not self.training or self._last is None:
            return
        s, a = self._last
        s2 = self.state(sim)
        target = reward + self.gamma * self.q[s2].max()
        self.q[s][a] += self.alpha * (target - self.q[s][a])


def simulate(controller, scenario="medium", duration=3600, seed=0) -> Intersection:
    sim = Intersection(scenario=scenario, seed=seed)
    while sim.t < duration:
        green = controller.decide(sim)
        before = sim.total_wait
        sim.run_green(green)
        sim.switch()
        if isinstance(controller, QLearningController):
            # reward: negative waiting accumulated during this action
            controller.learn(sim, -(sim.total_wait - before) / 100.0)
    return sim


def train_rl(episodes=300, duration=3600, scenarios=("low", "medium", "high", "rush"),
             seed=0, log=None):
    agent = QLearningController(seed=seed)
    history = []
    for ep in range(episodes):
        agent.epsilon = max(0.02, 0.3 * (1 - ep / (episodes * 0.8)))
        sc = scenarios[ep % len(scenarios)]
        sim = simulate(agent, sc, duration, seed=1000 + ep)
        history.append({"episode": ep + 1, "scenario": sc, **sim.metrics()})
        if log and (ep + 1) % 50 == 0:
            log(f"episode {ep + 1}: avg wait {history[-1]['avg_wait_s']:.1f}s ({sc})")
    agent.training = False
    return agent, history
