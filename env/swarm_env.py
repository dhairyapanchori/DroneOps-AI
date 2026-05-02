import numpy as np
from core.drone import Drone, WORLD_BOUND
from utils.config import *

OBSTACLE_RADIUS  = 2.0
TARGET_RADIUS    = 1.5
OBSTACLE_PENALTY = 0.4
TARGET_REWARD    = 2.0
ENERGY_WEIGHT    = 0.01
COORD_BONUS      = 0.15


class SwarmEnv:

    def __init__(self):
        self.drones          = [Drone(i) for i in range(NUM_DRONES)]
        self.t               = 0
        self.curriculum_ep   = 0   # set externally by trainer
        self.obstacles       = []
        self.targets         = np.zeros((3, 2))
        self.targets_reached = set()
        self.failed_ids      = set()

    # ── Curriculum helpers ────────────────────────────────────────────

    def _curriculum_phase(self):
        """
        Phase 0 (ep   0–74):  No obstacles, no failures, tight targets
        Phase 1 (ep  75–149): Add 2 obstacles, occasional failure
        Phase 2 (ep 150–224): 4 obstacles, wider targets, normal failures
        Phase 3 (ep 225+):    Full difficulty — 5 obstacles, 3 targets
        """
        ep = self.curriculum_ep
        if ep < 75:   return 0
        if ep < 150:  return 1
        if ep < 225:  return 2
        return 3

    def _n_obstacles(self):
        return [0, 2, 4, 5][self._curriculum_phase()]

    def _target_range(self):
        return [4, 5, 6, 7][self._curriculum_phase()]

    def _failure_prob(self):
        return [0.0, 0.01, 0.02, FAILURE_PROB][self._curriculum_phase()]

    def _max_failures(self):
        return [0, 1, 1, MAX_FAILURES][self._curriculum_phase()]

    def _n_targets(self):
        return [2, 2, 3, 3][self._curriculum_phase()]

    # ── Placement ─────────────────────────────────────────────────────

    def _place_obstacles(self):
        n = self._n_obstacles()
        if n == 0:
            return np.zeros((0, 2))
        obs, attempts = [], 0
        while len(obs) < n and attempts < 300:
            attempts += 1
            c = np.random.uniform(-8, 8, 2)
            if np.linalg.norm(c) < 4.0:
                continue
            if any(np.linalg.norm(c - o) < 3.5 for o in obs):
                continue
            obs.append(c)
        while len(obs) < n:
            obs.append(np.random.uniform(-8, 8, 2))
        return np.array(obs)

    def reset(self):
        for d in self.drones:
            d.reset()
        self.t               = 0
        self.obstacles       = self._place_obstacles()
        r                    = self._target_range()
        self.targets         = np.random.uniform(-r, r, (self._n_targets(), 2))
        self.targets_reached = set()
        self.failed_ids      = set()

        fp = self._failure_prob()
        mf = self._max_failures()
        if fp > 0:
            n_fail = min(np.random.binomial(NUM_DRONES, fp), mf)
            if n_fail > 0:
                for fid in np.random.choice(NUM_DRONES, n_fail, replace=False):
                    self.drones[int(fid)].alive = False
                    self.failed_ids.add(int(fid))

        return self.states()

    def active_drones(self):
        return [d for d in self.drones if d.alive]

    def neighbors(self, d):
        return [x for x in self.drones if x.id != d.id]

    def states(self):
        return np.array([
            d.state(self.neighbors(d), self.obstacles, self.targets)
            for d in self.drones
        ])

    def _reward(self, drone):
        if not drone.alive:
            return 0.0
        r = 0.0

        # 1. Inverse-distance proximity + first-touch bonus
        for ti, target in enumerate(self.targets):
            dist = np.linalg.norm(drone.pos - target)
            r   += 1.0 / (1.0 + dist)
            if dist < TARGET_RADIUS:
                key = (drone.id, ti)
                if key not in self.targets_reached:
                    r += TARGET_REWARD
                    self.targets_reached.add(key)

        # 2. Coordination bonus
        for target in self.targets:
            my_dist = np.linalg.norm(drone.pos - target)
            others  = [np.linalg.norm(d.pos - target)
                       for d in self.active_drones() if d.id != drone.id]
            if not others or my_dist < min(others):
                r += COORD_BONUS

        # 3. Obstacle avoidance
        for obs in self.obstacles:
            dist = np.linalg.norm(drone.pos - obs)
            if dist < OBSTACLE_RADIUS:
                r -= OBSTACLE_PENALTY * (1.0 - dist / OBSTACLE_RADIUS)

        # 4. Energy
        r += drone.energy * ENERGY_WEIGHT

        return float(np.clip(r, -3.0, 8.0))

    def step(self, actions):
        for d, a in zip(self.drones, actions):
            d.step(a)
            if not d.alive and d.id not in self.failed_ids:
                self.failed_ids.add(d.id)

        rewards = np.array([self._reward(d) for d in self.drones])
        self.t += 1
        done    = self.t >= MAX_STEPS
        return self.states(), rewards, done