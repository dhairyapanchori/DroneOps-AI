"""
Mission-level state primitives for the hierarchical mission planner.

These are pure data structures — no planning logic lives here. The
MissionPlanner (ml/planner/mission_planner.py) mutates them; every other
consumer (trainer, dashboards, future task allocators) reads them.

Vocabulary
----------
MissionPhase     : the swarm-level operating mode (SEARCH → RESCUE → RETURN).
MissionObjective : one mission target with a monotonic status lifecycle
                   (PENDING → ENGAGED → COMPLETED).
DroneDirective   : the planner's current high-level order for one drone —
                   advisory today, the injection point for goal-conditioned
                   policies and Dynamic Task Allocation later.
ZoneGrid         : partition of the square world into n×n search zones,
                   used to spread drones out during the SEARCH phase.
CoordinationState: live snapshot of the Task Coordination Engine state —
                   mission progress, task counts, drone assignments, and
                   per-phase step counts.  Populated by TaskCoordinationEngine
                   and stored in MissionState.coordination each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    # Imported only for type annotations; avoids circular imports at runtime.
    # TaskCoordinationEngine → coordination_engine → mission_state is safe
    # because coordination_engine imports mission_state, not vice-versa.
    from ml.planner.coordination_engine import CoordinationState


class MissionPhase(Enum):
    """Swarm-level operating mode, decided once per step for the whole swarm."""
    IDLE   = auto()   # no mission running (pre-mission, or all drones lost)
    SEARCH = auto()   # no objective engaged yet — spread across zones
    RESCUE = auto()   # objectives engaged/completed — converge on assignments
    RETURN = auto()   # mission finished or energy-critical — regroup


class ObjectiveStatus(Enum):
    """Lifecycle of a mission objective. Transitions are monotonic:
    PENDING → ENGAGED → COMPLETED (never backwards)."""
    PENDING   = auto()
    ENGAGED   = auto()
    COMPLETED = auto()


@dataclass
class MissionObjective:
    """One mission target the swarm must reach."""
    objective_id : int
    position     : np.ndarray                                  # world coords (2,)
    status       : ObjectiveStatus = ObjectiveStatus.PENDING

    @property
    def finished(self):
        return self.status is ObjectiveStatus.COMPLETED


@dataclass
class DroneDirective:
    """The planner's current high-level order for a single drone.

    Advisory in this release: the SAC policy does not consume directives,
    so issuing them cannot change trained behaviour. They are the stable
    interface that Dynamic Task Allocation and goal-conditioned policies
    will drive in a future release.
    """
    drone_id     : int
    phase        : MissionPhase
    zone_id      : int | None = None    # search zone to cover (SEARCH)
    objective_id : int | None = None    # objective to service (RESCUE)


class ZoneGrid:
    """Partitions the square world [-bound, bound]² into an n×n grid of zones.

    Zone ids are row-major: zone 0 is the bottom-left cell. The grid is the
    planner's spatial vocabulary for area assignment — coarse on purpose,
    since fine-grained positioning is the SAC policy's job.
    """

    def __init__(self, world_bound, n=2):
        self.bound = float(world_bound)
        self.n     = int(n)
        self.size  = (2.0 * self.bound) / self.n   # side length of one zone

    @property
    def n_zones(self):
        return self.n * self.n

    def zone_of(self, pos):
        """Zone id containing world position `pos` (clamped to the grid)."""
        col = int(np.clip((pos[0] + self.bound) // self.size, 0, self.n - 1))
        row = int(np.clip((pos[1] + self.bound) // self.size, 0, self.n - 1))
        return row * self.n + col

    def center(self, zone_id):
        """World-coordinate centre of a zone, shape (2,)."""
        row, col = divmod(int(zone_id), self.n)
        cx = -self.bound + (col + 0.5) * self.size
        cy = -self.bound + (row + 0.5) * self.size
        return np.array([cx, cy])


@dataclass
class MissionState:
    """Complete snapshot of the mission at the current step.

    Owned and mutated by MissionPlanner; treat as read-only elsewhere.

    Fields
    ------
    phase       : Current swarm-level operating mode.
    step        : Step counter since begin_mission().
    objectives  : dict[objective_id, MissionObjective].
    directives  : dict[drone_id, DroneDirective] — advisory only.
    coordination: CoordinationState snapshot from TaskCoordinationEngine;
                  None until the first update() call after begin_mission().
    """
    phase        : MissionPhase = MissionPhase.IDLE
    step         : int = 0
    objectives   : dict[int, MissionObjective] = field(default_factory=dict)
    directives   : dict[int, DroneDirective]   = field(default_factory=dict)
    coordination : Optional["CoordinationState"] = field(default=None, repr=False)

    # ── Convenience queries ───────────────────────────────────────────

    def unfinished_objectives(self):
        return [o for o in self.objectives.values() if not o.finished]

    def completed_count(self):
        return sum(1 for o in self.objectives.values() if o.finished)

    @property
    def all_objectives_complete(self):
        return bool(self.objectives) and not self.unfinished_objectives()
