"""
Task Coordination Engine — the central subsystem for Feature 2.

Position in the architecture
-----------------------------
    MissionPlanner owns and drives TaskCoordinationEngine each step:

        MissionPlanner.begin_mission()
            └─► TaskCoordinationEngine.register_tasks(objectives)

        MissionPlanner.update()
            └─► TaskCoordinationEngine.update(env, phase)
                    ├─ _refresh_drone_capacities(env)
                    ├─ _advance_task_lifecycles(env, phase)
                    └─ _run_allocation(step)

    The engine writes its results into self.coordination_state, which the
    planner then stores in MissionState.coordination.

Responsibilities
----------------
1. Task Registry  — one MissionTask per objective, created at mission start.
2. Lifecycle      — monotonic PENDING → ASSIGNED → IN_PROGRESS → COMPLETED | FAILED
                    (plus FAILED → PENDING requeue when a drone dies but the
                     objective remains reachable).
3. Drone Tracking — builds DroneCapacity snapshots every step from live telemetry.
4. Allocation     — delegates to a pluggable TaskAllocationStrategy; commits
                    assignments; respects battery and workload constraints.
5. State API      — exposes coordination_state (CoordinationState) and
                    task_summary() dict for the trainer and dashboard.

Design notes
------------
- Fully deterministic: no RNG; safe to run inside the training loop.
- Single Responsibility: the engine manages tasks. The planner manages phases
  and objectives. The allocator decides assignments. Each is independently
  testable and replaceable.
- The allocation strategy is constructor-injected (Dependency Inversion).
  Swap MultiCriteriaAllocator for an auction, Hungarian, or learned strategy
  without changing this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ml.planner.allocation_engine import MultiCriteriaAllocator, TaskAllocationStrategy
from ml.planner.mission_state import MissionObjective, MissionPhase
from ml.planner.task import DroneCapacity, DroneStatus, MissionTask, TaskStatus
from utils.config import (PLANNER_ENGAGE_RADIUS, TASK_ALLOC_W_BATT,
                          TASK_ALLOC_W_DIST, TASK_ALLOC_W_LOAD,
                          TASK_LOW_BATTERY)


# ── Coordination state snapshot ───────────────────────────────────────────────

@dataclass
class CoordinationState:
    """Immutable snapshot of coordination state at the current step.

    Produced by TaskCoordinationEngine.update() and stored in
    MissionState.coordination.  Consumers (trainer, dashboard, logger)
    should treat this as read-only.

    Fields
    ------
    mission_progress  : fraction of tasks completed [0.0, 1.0].
    task_counts       : dict mapping TaskStatus.name → count for all statuses.
    drone_assignments : dict[drone_id, task_id | None] — current assignment.
    active_drones     : number of alive drones this step.
    total_drones      : total drone count in the swarm.
    phase_step_counts : dict[phase_name, step_count] — steps spent per phase.
    """
    mission_progress  : float
    task_counts       : dict[str, int]
    drone_assignments : dict[int, Optional[int]]
    active_drones     : int
    total_drones      : int
    phase_step_counts : dict[str, int]


# ── Task Coordination Engine ──────────────────────────────────────────────────

class TaskCoordinationEngine:
    """Manages the full task lifecycle and coordinates drone assignments.

    This is the primary new subsystem introduced in Feature 2.

    Args:
        allocator  : TaskAllocationStrategy implementation.
                     Defaults to MultiCriteriaAllocator (distance + battery +
                     workload weighted scoring).
                     Inject a different strategy here to replace the allocation
                     algorithm without changing any other code.
    """

    def __init__(self, allocator: Optional[TaskAllocationStrategy] = None):
        self._allocator : TaskAllocationStrategy = allocator or MultiCriteriaAllocator(
            w_dist=TASK_ALLOC_W_DIST,
            w_batt=TASK_ALLOC_W_BATT,
            w_load=TASK_ALLOC_W_LOAD,
        )

        # Task registry: task_id → MissionTask
        self._tasks      : dict[int, MissionTask] = {}
        # Reverse index: drone_id → task_id (None if unassigned)
        self._drone_task : dict[int, Optional[int]] = {}
        # Live drone capacities (rebuilt each step)
        self._capacities : dict[int, DroneCapacity] = {}
        # Phase step counter (mirrors planner's _phase_steps)
        self._phase_steps: dict[str, int] = {p.name: 0 for p in MissionPhase}

        self.coordination_state: Optional[CoordinationState] = None

    # ── Mission lifecycle API ─────────────────────────────────────────────

    def register_tasks(
        self,
        objectives : dict[int, MissionObjective],
        start_step : int = 0,
    ) -> None:
        """Create one MissionTask per objective.  Called at mission start.

        Resets all internal state so the engine is ready for a new mission.
        """
        self._tasks.clear()
        self._drone_task.clear()
        self._capacities.clear()
        self._phase_steps = {p.name: 0 for p in MissionPhase}
        self.coordination_state = None

        for obj_id, obj in objectives.items():
            task = MissionTask(
                task_id      = obj_id,      # 1:1 mapping for now
                objective_id = obj_id,
                position     = obj.position.copy(),
                priority     = 1,           # uniform priority baseline
                created_step = start_step,
            )
            self._tasks[task.task_id] = task

    def update(self, env, phase: MissionPhase) -> CoordinationState:
        """Advance task lifecycle and run allocation for the current step.

        Call once per step, after MissionPlanner._refresh_objectives().

        Args:
            env   : SwarmEnv — provides live drone telemetry.
            phase : Current MissionPhase from MissionPlanner.

        Returns:
            Updated CoordinationState (also stored in self.coordination_state).
        """
        step = getattr(env, 't', 0)
        self._phase_steps[phase.name] = self._phase_steps.get(phase.name, 0) + 1

        self._refresh_drone_capacities(env)
        self._advance_task_lifecycles(env, phase, step)
        self._run_allocation(step)
        self.coordination_state = self._build_coordination_state(env)
        return self.coordination_state

    # ── Serialisation API ─────────────────────────────────────────────────

    def task_summary(self) -> dict:
        """Serialisable end-of-mission task summary for logging / metrics."""
        counts = self._count_by_status()
        total  = len(self._tasks)
        return {
            "total_tasks"        : total,
            "tasks_completed"    : counts.get(TaskStatus.COMPLETED.name, 0),
            "tasks_failed"       : counts.get(TaskStatus.FAILED.name, 0),
            "tasks_pending"      : counts.get(TaskStatus.PENDING.name, 0),
            "mission_progress"   : self._mission_progress(),
            "allocator"          : repr(self._allocator),
        }

    # ── Internal: drone capacity refresh ─────────────────────────────────

    def _refresh_drone_capacities(self, env) -> None:
        """Rebuild DroneCapacity snapshots from live env telemetry."""
        # Count active tasks per drone
        task_count: dict[int, int] = {}
        for task in self._tasks.values():
            if task.is_active and task.assigned_drone_id is not None:
                did = task.assigned_drone_id
                task_count[did] = task_count.get(did, 0) + 1

        self._capacities = {}
        for drone in env.drones:
            if not drone.alive:
                status = DroneStatus.OFFLINE
            elif drone.energy < TASK_LOW_BATTERY:
                status = DroneStatus.LOW_BATTERY
            elif task_count.get(drone.id, 0) > 0:
                status = DroneStatus.BUSY
            else:
                status = DroneStatus.AVAILABLE

            self._capacities[drone.id] = DroneCapacity(
                drone_id = drone.id,
                status   = status,
                position = drone.pos.copy(),
                battery  = float(drone.energy),
                workload = task_count.get(drone.id, 0),
            )

        # Sync drone_task index
        self._drone_task = {d.id: None for d in env.drones}
        for task in self._tasks.values():
            if task.is_active and task.assigned_drone_id is not None:
                self._drone_task[task.assigned_drone_id] = task.task_id

    # ── Internal: task lifecycle advancement ─────────────────────────────

    def _advance_task_lifecycles(self, env, phase: MissionPhase, step: int) -> None:
        """Advance task statuses based on env telemetry.

        Rules (in priority order):
        1. Completed in env → COMPLETED (terminal).
        2. Assigned drone offline → FAILED; requeue if objective still open.
        3. Assigned drone within engage radius → IN_PROGRESS.
        4. RETURN / IDLE phase: non-terminal tasks are left as-is (no realloc).
        """
        completed_obj_ids = {ti for (_, ti) in env.targets_reached}
        alive_drone_ids   = {d.id for d in env.drones if d.alive}

        for task in self._tasks.values():
            if task.is_terminal:
                continue

            # Rule 1: objective completed in env
            if task.objective_id in completed_obj_ids:
                task.mark_completed(step)
                continue

            # Rule 2: assigned drone went offline
            if (task.assigned_drone_id is not None
                    and task.assigned_drone_id not in alive_drone_ids):
                task.mark_failed()
                # Requeue if there are still drones alive to service the task
                if alive_drone_ids:
                    task.requeue()
                continue

            # Rule 3: assigned drone within engage radius → IN_PROGRESS
            if task.status is TaskStatus.ASSIGNED and task.assigned_drone_id is not None:
                cap = self._capacities.get(task.assigned_drone_id)
                if cap is not None:
                    dist = float(np.linalg.norm(cap.position - task.position))
                    if dist < PLANNER_ENGAGE_RADIUS:
                        task.mark_in_progress()

    # ── Internal: allocation ──────────────────────────────────────────────

    def _run_allocation(self, step: int) -> None:
        """Assign PENDING tasks to available drones via the pluggable strategy.

        Only runs when there are PENDING tasks and AVAILABLE/LOW_BATTERY drones.
        Already-active tasks are not reallocated unless they were requeued.
        """
        pending_tasks = [t for t in self._tasks.values()
                         if t.status is TaskStatus.PENDING]
        if not pending_tasks:
            return

        available_caps = {
            did: cap for did, cap in self._capacities.items()
            if cap.is_available_for_assignment
        }
        if not available_caps:
            return

        assignments = self._allocator.allocate(pending_tasks, available_caps)

        for task in pending_tasks:
            drone_id = assignments.get(task.task_id)
            if drone_id is not None:
                task.assign_to(drone_id, step)
                # Update capacity workload so subsequent loop iterations see
                # the new workload (important if the same allocator is called
                # again within the same step in tests/future extensions)
                if drone_id in self._capacities:
                    self._capacities[drone_id].workload += 1
                    if self._capacities[drone_id].workload > 0:
                        self._capacities[drone_id].status = DroneStatus.BUSY

    # ── Internal: state construction ──────────────────────────────────────

    def _build_coordination_state(self, env) -> CoordinationState:
        """Construct the read-only CoordinationState snapshot for this step."""
        # Build drone → task assignment map (for all drones)
        drone_assignments: dict[int, Optional[int]] = {}
        for drone in env.drones:
            drone_assignments[drone.id] = self._drone_task.get(drone.id)

        active_count = sum(1 for d in env.drones if d.alive)

        return CoordinationState(
            mission_progress  = self._mission_progress(),
            task_counts       = self._count_by_status(),
            drone_assignments = drone_assignments,
            active_drones     = active_count,
            total_drones      = len(env.drones),
            phase_step_counts = dict(self._phase_steps),
        )

    # ── Internal: helpers ─────────────────────────────────────────────────

    def _mission_progress(self) -> float:
        """Fraction of tasks in COMPLETED state [0.0, 1.0]."""
        if not self._tasks:
            return 0.0
        completed = sum(1 for t in self._tasks.values()
                        if t.status is TaskStatus.COMPLETED)
        return completed / len(self._tasks)

    def _count_by_status(self) -> dict[str, int]:
        """Count tasks per TaskStatus, keyed by status name string."""
        counts: dict[str, int] = {s.name: 0 for s in TaskStatus}
        for task in self._tasks.values():
            counts[task.status.name] += 1
        return counts
