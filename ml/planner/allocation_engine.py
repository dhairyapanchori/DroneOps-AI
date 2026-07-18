"""
Task-layer allocation strategies for the Task Allocation & Coordination Engine.

Design
------
TaskAllocationStrategy is a runtime-checkable Protocol — any object that
implements `allocate(tasks, capacities)` with the correct signature is a valid
strategy.  No inheritance required.  This satisfies the Dependency Inversion
principle: TaskCoordinationEngine depends on the abstract Protocol, not on any
concrete allocator.

Shipped strategies
------------------
MultiCriteriaAllocator  — deterministic weighted-score greedy baseline.
    Scores every (task, drone) pair using three criteria:
        proximity  : 1 / (1 + euclidean distance)   favours closer drones
        battery    : normalised energy level         favours healthier drones
        workload   : current task count              penalises busy drones
    Assigns greedily: picks the highest-scoring unmatched (task, drone) pair,
    commits it, updates the drone's workload, repeats until all tasks or all
    drones are exhausted.  Runs in O(T × D) time — fine for swarm sizes
    used in this project.

    Weights are constructor-injected from config constants, so tuning is a
    one-file edit.  Passing a custom weight triple is sufficient to express
    "distance-only", "battery-first", or any other policy without subclassing.

Extension points
----------------
Future algorithms (auction-based, Hungarian assignment, learned allocation)
implement the same `allocate` signature and are injected into
TaskCoordinationEngine via its constructor — no other code changes required.

Contract for implementers
-------------------------
- `allocate` MUST be deterministic given its inputs (no RNG).
- Return a mapping for every task_id in `tasks`; use None when the task cannot
  be assigned this step (no suitable drone available).
- Do NOT mutate the MissionTask or DroneCapacity objects — the engine does
  that after receiving the returned mapping.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ml.planner.task import DroneCapacity, DroneStatus, MissionTask


# ── Strategy interface ────────────────────────────────────────────────────────

@runtime_checkable
class TaskAllocationStrategy(Protocol):
    """Protocol: assign tasks to drones.

    Args:
        tasks      : List of MissionTask objects that need (re)assignment.
                     Only non-terminal tasks are passed; already-active tasks
                     are included only when the engine decides reallocation is
                     warranted (e.g. assigned drone went offline).
        capacities : dict[drone_id, DroneCapacity] — live snapshot of every
                     drone's availability.  OFFLINE drones are included so
                     strategies can choose to skip them explicitly.

    Returns:
        dict[task_id, drone_id | None]
            task_id → drone_id  : assign this task to this drone.
            task_id → None      : no suitable drone found this step (task stays
                                  PENDING until the next allocation cycle).
    """

    def allocate(
        self,
        tasks      : list[MissionTask],
        capacities : dict[int, DroneCapacity],
    ) -> dict[int, int | None]:
        ...


# ── Multi-criteria greedy allocator ──────────────────────────────────────────

class MultiCriteriaAllocator:
    """Deterministic weighted-score greedy task allocator.

    Scoring function per (task t, drone d) pair:

        score(t, d) = w_dist × prox(t, d)
                    + w_batt × d.battery
                    − w_load × d.workload
                    + w_prio × t.priority

    where prox(t, d) = 1 / (1 + ||t.position − d.position||₂).

    LOW_BATTERY drones are not excluded — they receive a battery score
    penalty naturally.  OFFLINE drones are always skipped.

    Assignment is greedy: the globally highest-scoring (task, drone) pair is
    committed first, the drone's workload is incremented, and the process
    repeats.  This does not guarantee a globally optimal assignment (that would
    require the Hungarian algorithm), but it is deterministic, fast, and
    suitable as the production baseline.  It is also easily replaceable.

    Args:
        w_dist : Weight for proximity score  (default from config).
        w_batt : Weight for battery score    (default from config).
        w_load : Weight for workload penalty (default from config).
        w_prio : Weight for task priority    (default 0.1).
    """

    def __init__(
        self,
        w_dist : float = 1.0,
        w_batt : float = 0.5,
        w_load : float = 0.3,
        w_prio : float = 0.1,
    ):
        self.w_dist = float(w_dist)
        self.w_batt = float(w_batt)
        self.w_load = float(w_load)
        self.w_prio = float(w_prio)

    # ── Public API ────────────────────────────────────────────────────────

    def allocate(
        self,
        tasks      : list[MissionTask],
        capacities : dict[int, DroneCapacity],
    ) -> dict[int, int | None]:
        """Return task_id → drone_id | None for every task in `tasks`."""
        if not tasks:
            return {}

        # Filter drones that can receive work
        eligible = {
            did: cap for did, cap in capacities.items()
            if cap.status is not DroneStatus.OFFLINE
        }

        if not eligible:
            return {t.task_id: None for t in tasks}

        # Build mutable workload tracker (so greedy loop reflects assignments)
        workload = {did: cap.workload for did, cap in eligible.items()}

        # Pre-compute score matrix  shape: (n_tasks, n_drones)
        drone_ids = list(eligible.keys())
        n_t, n_d  = len(tasks), len(drone_ids)

        scores = np.zeros((n_t, n_d), dtype=np.float64)
        for ti, task in enumerate(tasks):
            for di, did in enumerate(drone_ids):
                cap  = eligible[did]
                dist = float(np.linalg.norm(task.position - cap.position))
                prox = 1.0 / (1.0 + dist)
                scores[ti, di] = (
                    self.w_dist * prox
                    + self.w_batt * cap.battery
                    - self.w_load * workload[did]
                    + self.w_prio * task.priority
                )

        # Greedy assignment: pick max (task, drone) pair, commit, repeat
        result        : dict[int, int | None] = {t.task_id: None for t in tasks}
        assigned_tasks = set()
        assigned_drones = set()

        # Sort (task_idx, drone_idx) pairs by descending score
        flat_indices = np.argsort(-scores, axis=None)   # descending

        for flat_idx in flat_indices:
            ti = int(flat_idx // n_d)
            di = int(flat_idx  % n_d)

            if ti in assigned_tasks or di in assigned_drones:
                continue

            task_id  = tasks[ti].task_id
            drone_id = drone_ids[di]

            # Check drone type compatibility (heterogeneous fleet extension)
            task = tasks[ti]
            if (task.drone_type_required is not None
                    and task.drone_type_required not in eligible[drone_id].capabilities):
                continue

            result[task_id] = drone_id
            workload[drone_id] += 1
            assigned_tasks.add(ti)
            assigned_drones.add(di)

            if len(assigned_tasks) == n_t:
                break   # all tasks assigned

        return result

    def __repr__(self) -> str:
        return (
            f"MultiCriteriaAllocator("
            f"w_dist={self.w_dist}, w_batt={self.w_batt}, "
            f"w_load={self.w_load}, w_prio={self.w_prio})"
        )
