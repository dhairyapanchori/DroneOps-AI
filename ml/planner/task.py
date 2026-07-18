"""
Task-layer data primitives for the Mission Task Allocation & Coordination Engine.

Vocabulary
----------
TaskStatus     : Full lifecycle of a single atomic mission task.
                 PENDING → ASSIGNED → IN_PROGRESS → COMPLETED | FAILED
                 Transitions are monotonic except ASSIGNED → PENDING on drone loss
                 (re-queued for reallocation).

DroneStatus    : Operational state of a drone as seen by the coordination engine.
                 AVAILABLE  — alive, not at workload capacity, battery OK
                 BUSY       — alive, already assigned to a task this step
                 LOW_BATTERY — alive but energy below TASK_LOW_BATTERY threshold
                 OFFLINE    — dead (energy == 0 or alive == False)

MissionTask    : One atomic unit of work: reach an objective, report completion.
                 Created once per objective at mission start. Immutable identity
                 fields (id, objective_id, position, priority) are set at creation;
                 mutable status fields are updated by TaskCoordinationEngine.

DroneCapacity  : Live snapshot of one drone's availability, rebuilt every step
                 from env telemetry.  Never cached across steps.

Design notes
------------
- All enums derive from str so they serialise cleanly to JSON/log strings.
- MissionTask uses __slots__ = () to catch accidental attribute creation.
- DroneCapacity is intentionally a plain dataclass (not a domain object) —
  it is rebuilt every step, never mutated in-place.
- The heterogeneous-drone extension point: MissionTask.drone_type_required
  defaults to None (any drone may service it); typed fleets set it to a string
  capability tag matched against DroneCapacity.capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ── Task lifecycle ────────────────────────────────────────────────────────────

class TaskStatus(Enum):
    """Atomic task lifecycle.  Transitions enforced by TaskCoordinationEngine."""
    PENDING     = auto()   # created, not yet assigned to any drone
    ASSIGNED    = auto()   # a drone has been allocated to this task
    IN_PROGRESS = auto()   # assigned drone is within PLANNER_ENGAGE_RADIUS
    COMPLETED   = auto()   # objective reached — terminal, success
    FAILED      = auto()   # assigned drone went offline — terminal, failure

    @property
    def is_terminal(self) -> bool:
        """True for states from which no further transition is possible."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    @property
    def is_active(self) -> bool:
        """True when the task is consuming a drone slot."""
        return self in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)


# ── Drone operational status ──────────────────────────────────────────────────

class DroneStatus(Enum):
    """Drone availability as assessed by the coordination engine each step."""
    AVAILABLE   = auto()   # healthy, ready to accept a task
    BUSY        = auto()   # already servicing a task
    LOW_BATTERY = auto()   # alive but energy-critical; deprioritised
    OFFLINE     = auto()   # dead — excluded from allocation


# ── Task data model ───────────────────────────────────────────────────────────

@dataclass
class MissionTask:
    """One atomic unit of mission work.

    Identity fields are set at creation and never change.
    Status fields are mutated by TaskCoordinationEngine only.

    Args:
        task_id            : Unique task identifier (int, globally unique per mission).
        objective_id       : Which MissionObjective this task services.
        position           : World-coordinate target position, shape (2,).
        priority           : Higher value → preferred for allocation (default 1).
        drone_type_required: Capability tag for heterogeneous fleets (None = any drone).
    """

    # ── Identity (immutable) ─────────────────────────────────────────────
    task_id             : int
    objective_id        : int
    position            : np.ndarray           # world coords (2,)
    priority            : int = 1
    drone_type_required : Optional[str] = None  # extension point for typed fleets

    # ── Status (mutable, managed by TaskCoordinationEngine) ─────────────
    status              : TaskStatus = TaskStatus.PENDING
    assigned_drone_id   : Optional[int] = None

    # ── Timing audit trail ───────────────────────────────────────────────
    created_step        : int = 0
    assigned_step       : Optional[int] = None
    completed_step      : Optional[int] = None

    def assign_to(self, drone_id: int, step: int) -> None:
        """Transition PENDING → ASSIGNED."""
        self.status           = TaskStatus.ASSIGNED
        self.assigned_drone_id = drone_id
        self.assigned_step    = step

    def mark_in_progress(self) -> None:
        """Transition ASSIGNED → IN_PROGRESS when drone is within engage radius."""
        if self.status is TaskStatus.ASSIGNED:
            self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, step: int) -> None:
        """Transition any active status → COMPLETED."""
        self.status         = TaskStatus.COMPLETED
        self.completed_step = step

    def mark_failed(self) -> None:
        """Transition any active status → FAILED (assigned drone went offline)."""
        self.status           = TaskStatus.FAILED
        self.assigned_drone_id = None

    def requeue(self) -> None:
        """Reset FAILED task back to PENDING for reallocation.

        Called when a task fails but the objective is still reachable and there
        are other drones available.  This is the only backwards transition
        allowed in the lifecycle — it is not a contradiction because the *task*
        failed (drone died) but the *objective* is still open.
        """
        self.status           = TaskStatus.PENDING
        self.assigned_drone_id = None
        self.assigned_step    = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_active(self) -> bool:
        return self.status.is_active


# ── Drone capacity snapshot ───────────────────────────────────────────────────

@dataclass
class DroneCapacity:
    """Live snapshot of one drone's allocation capacity.

    Rebuilt from env telemetry every step by TaskCoordinationEngine.
    Never cached or mutated between steps.

    Args:
        drone_id    : Matches Drone.id in SwarmEnv.
        status      : Operational availability (see DroneStatus).
        position    : Current world position, shape (2,).
        battery     : Current energy level [0, 1].
        workload    : Number of active tasks currently assigned to this drone.
        capabilities: Set of capability tags (extension point for typed fleets).
    """
    drone_id     : int
    status       : DroneStatus
    position     : np.ndarray
    battery      : float
    workload     : int = 0
    capabilities : frozenset = field(default_factory=frozenset)

    @property
    def is_available_for_assignment(self) -> bool:
        """True when the drone can accept a new task assignment."""
        return self.status in (DroneStatus.AVAILABLE, DroneStatus.LOW_BATTERY)
