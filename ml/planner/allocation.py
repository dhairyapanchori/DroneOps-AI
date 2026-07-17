"""
Task allocation strategies — the extension point for Dynamic Task Allocation.

The MissionPlanner never decides *which drone services which objective*
itself; it delegates to a TaskAllocationStrategy (strategy pattern /
dependency inversion). This release ships one deterministic baseline,
NearestObjectiveAllocator. A future Dynamic Task Allocation release
(auction-based, Hungarian assignment, learned allocation, ...) implements
the same interface and is injected into MissionPlanner — no planner or
trainer changes required.

Contract notes for implementers
-------------------------------
- `assign` must be deterministic given its inputs (no RNG): the planner
  runs inside the training loop, and consuming random state there would
  silently change training trajectories.
- Return a mapping for every id in `alive_ids`; use None when a drone has
  no objective (e.g. everything is completed).
"""

from abc import ABC, abstractmethod

import numpy as np


class TaskAllocationStrategy(ABC):
    """Interface: decide which unfinished objective each alive drone services."""

    @abstractmethod
    def assign(self, objectives, drone_positions, alive_ids):
        """
        Args:
            objectives      : list[MissionObjective] — unfinished only.
            drone_positions : dict[int, np.ndarray] — id → world position (2,).
            alive_ids       : list[int] — ids of drones available for tasking.
        Returns:
            dict[int, int | None] — drone id → objective id (None = unassigned).
        """
        raise NotImplementedError


class NearestObjectiveAllocator(TaskAllocationStrategy):
    """Greedy static baseline: each drone services its nearest unfinished
    objective.

    Deterministic and O(drones × objectives). Multiple drones may share an
    objective — acceptable for a baseline, and exactly the redundancy the
    future Dynamic Task Allocation strategy exists to optimise away.
    """

    def assign(self, objectives, drone_positions, alive_ids):
        if not objectives:
            return {i: None for i in alive_ids}

        assignments = {}
        for i in alive_ids:
            dists = [np.linalg.norm(drone_positions[i] - o.position)
                     for o in objectives]
            assignments[i] = objectives[int(np.argmin(dists))].objective_id
        return assignments
