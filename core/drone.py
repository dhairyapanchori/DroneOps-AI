"""
Single-drone model: point-mass kinematics, energy budget, and local sensing.

Design notes
------------
- The drone is a 2D point mass: action[:2] is acceleration, integrated into
  velocity then position each step. There is no drag or dt constant — the
  velocity clip (±2) is what bounds top speed.
- Energy drains in proportion to the *full* action norm each step and never
  recovers; a drone at zero energy is permanently dead. This is what makes
  wasteful control costly and enables the "energy crisis" failure mode.
- Observations are deliberately local and normalised to roughly [-1, 1]:
  a drone senses only its own kinematics/energy, distances to up to 5
  neighbours, and the distance + coarse direction (sign only) of the single
  nearest obstacle and target. The swarm-level picture is provided later by
  the GNN/Transformer fusion pipeline, not by the raw observation.
"""

import numpy as np

WORLD_BOUND = 12.0   # hard wall — drones cannot go beyond this
POS_SCALE   = WORLD_BOUND
VEL_SCALE   = 2.0
DIST_SCALE  = WORLD_BOUND * 2


class Drone:
    """A single drone: 2D point-mass kinematics, energy budget, local sensing."""

    def __init__(self, i):
        self.id = i
        self.reset()

    def reset(self):
        """Respawn at a random position with zero velocity and full energy."""
        self.pos    = np.random.uniform(-8, 8, 2)
        self.vel    = np.zeros(2)
        self.energy = 1.0
        self.alive  = True

    def step(self, action):
        """Apply one control step: accelerate, move, bounce off walls, drain energy.

        Only action[:2] drives motion; the full action norm drains energy.
        A drone whose energy reaches zero dies permanently.
        """
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
        """Build the 16-dim normalised observation vector.

        Layout: [pos(2), vel(2), energy(1), neighbor dists(5),
                 nearest obstacle dist(1), nearest target dist(1),
                 target direction(2), obstacle direction(2)].
        Dead drones observe all zeros.
        """
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