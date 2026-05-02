import numpy as np

WORLD_BOUND = 12.0   # hard wall — drones cannot go beyond this
POS_SCALE   = WORLD_BOUND
VEL_SCALE   = 2.0
DIST_SCALE  = WORLD_BOUND * 2


class Drone:
    def __init__(self, i):
        self.id = i
        self.reset()

    def reset(self):
        self.pos    = np.random.uniform(-8, 8, 2)
        self.vel    = np.zeros(2)
        self.energy = 1.0
        self.alive  = True

    def step(self, action):
        if not self.alive:
            return

        # Update velocity and position
        self.vel = np.clip(self.vel + action[:2] * 0.1, -2.0, 2.0)
        self.pos = self.pos + self.vel

        # ── Hard boundary walls — bounce and clamp ─────────────────
        for axis in range(2):
            if self.pos[axis] > WORLD_BOUND:
                self.pos[axis] = WORLD_BOUND
                self.vel[axis] = -abs(self.vel[axis]) * 0.5   # damped bounce
            elif self.pos[axis] < -WORLD_BOUND:
                self.pos[axis] = -WORLD_BOUND
                self.vel[axis] =  abs(self.vel[axis]) * 0.5

        self.energy = max(0.0, self.energy - np.linalg.norm(action) * 0.001)
        if self.energy <= 0.0:
            self.alive = False

    def state(self, neighbors, obstacles=None, targets=None):
        if not self.alive:
            return np.zeros(16, dtype=np.float32)

        n_dist = []
        for n in neighbors:
            if n.alive:
                n_dist.append(min(np.linalg.norm(self.pos - n.pos) / DIST_SCALE, 1.0))
        while len(n_dist) < 5:
            n_dist.append(0.0)

        if obstacles is not None and len(obstacles) > 0:
            obs_dists       = [np.linalg.norm(self.pos - o) for o in obstacles]
            nearest_obs_idx = int(np.argmin(obs_dists))
            nearest_obs     = min(obs_dists[nearest_obs_idx] / DIST_SCALE, 1.0)
            obs_dir         = np.sign(obstacles[nearest_obs_idx] - self.pos)
        else:
            nearest_obs = 1.0
            obs_dir     = np.zeros(2)

        if targets is not None and len(targets) > 0:
            tgt_dists       = [np.linalg.norm(self.pos - t) for t in targets]
            nearest_tgt_idx = int(np.argmin(tgt_dists))
            nearest_tgt     = min(tgt_dists[nearest_tgt_idx] / DIST_SCALE, 1.0)
            tgt_dir         = np.sign(targets[nearest_tgt_idx] - self.pos)
        else:
            nearest_tgt = 1.0
            tgt_dir     = np.zeros(2)

        return np.array([
            self.pos[0] / POS_SCALE,
            self.pos[1] / POS_SCALE,
            self.vel[0] / VEL_SCALE,
            self.vel[1] / VEL_SCALE,
            self.energy,
            *n_dist[:5],
            nearest_obs,
            nearest_tgt,
            *tgt_dir,
            *obs_dir,
        ], dtype=np.float32)