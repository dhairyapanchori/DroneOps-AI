"""
Hierarchical Mission Planner — the mission-command layer of the stack.

Position in the architecture
----------------------------
    SwarmEnv ──► MissionPlanner ──► MetaAdapter ─► GNN/Transformer ─► Actor
                 (mission command)   (motion-control pipeline, unchanged)

The planner sits ABOVE the learned control pipeline: once per environment
step it reads swarm telemetry (positions, energy, alive flags, targets
reached), refreshes mission state, derives the swarm-level MissionPhase,
and issues one DroneDirective per drone via its TaskAllocationStrategy
and ZoneGrid.

Directives are ADVISORY in this release. The SAC actor was trained on raw
fused observations, so feeding it directives would invalidate the shipped
checkpoints; instead the planner exposes a stable read API (`state`,
`mission_summary`) consumed by the trainer, metrics, and dashboard.
Goal-conditioned control and Dynamic Task Allocation plug in later by
(a) implementing TaskAllocationStrategy and (b) conditioning the policy
on `state.directives` — both without changing this class's interface.

Phase policy (evaluated every step, in priority order)
------------------------------------------------------
    IDLE    no drone alive, or no mission started
    RETURN  every objective completed, OR mean alive energy below
            PLANNER_RETURN_ENERGY (energy reserve for the trip home)
    RESCUE  at least one objective engaged or completed
    SEARCH  otherwise — spread across ZoneGrid zones round-robin
RETURN is absorbing for the remainder of the mission: once the swarm is
sent home it is not re-tasked, matching real airborne-operations doctrine.
"""

import numpy as np

from core.drone import WORLD_BOUND
from ml.planner.allocation import NearestObjectiveAllocator
from ml.planner.coordination_engine import TaskCoordinationEngine
from ml.planner.mission_state import (DroneDirective, MissionObjective,
                                      MissionPhase, MissionState,
                                      ObjectiveStatus, ZoneGrid)
from utils.config import (PLANNER_ENGAGE_RADIUS, PLANNER_RETURN_ENERGY,
                          PLANNER_ZONE_GRID)


class MissionPlanner:
    """Tracks mission state and issues per-drone directives each step.

    Now also owns a TaskCoordinationEngine that manages the full task
    lifecycle (PENDING → ASSIGNED → IN_PROGRESS → COMPLETED | FAILED),
    tracks drone availability and battery constraints, and delegates
    multi-criteria task assignment to a pluggable TaskAllocationStrategy.

    Stateless with respect to RNG (fully deterministic given env telemetry),
    so running it inside the training loop cannot alter training trajectories.

    Args:
        allocator  : TaskAllocationStrategy for RESCUE-phase objective tasking.
                     Defaults to the greedy NearestObjectiveAllocator;
                     inject a Dynamic Task Allocation strategy here later.
        zone_grid  : ZoneGrid for SEARCH-phase area assignment.
                     Defaults to PLANNER_ZONE_GRID × PLANNER_ZONE_GRID zones.
        task_engine: TaskCoordinationEngine instance.  Defaults to a fresh
                     engine with MultiCriteriaAllocator weights from config.
                     Inject a custom engine (with a different allocator) for
                     experiments or future algorithm comparisons.
    """

    def __init__(self, allocator=None, zone_grid=None, task_engine=None):
        self.allocator   = allocator or NearestObjectiveAllocator()
        self.zones       = zone_grid or ZoneGrid(WORLD_BOUND, PLANNER_ZONE_GRID)
        self.state       = MissionState()
        self._engine     = task_engine or TaskCoordinationEngine()
        self._phase_steps = {p: 0 for p in MissionPhase}   # step count per phase

    # ── Mission lifecycle ─────────────────────────────────────────────

    def begin_mission(self, env):
        """Start a new mission from a freshly reset environment.

        Builds one MissionObjective per environment target, registers tasks
        with the coordination engine, and issues the opening SEARCH directives.
        Call immediately after `env.reset()`.
        """
        objectives = {
            ti: MissionObjective(objective_id=ti, position=np.array(tgt, dtype=float))
            for ti, tgt in enumerate(env.targets)
        }
        self.state = MissionState(phase=MissionPhase.SEARCH,
                                  objectives=objectives)
        self._phase_steps = {p: 0 for p in MissionPhase}
        # Register tasks with the coordination engine (resets engine state too)
        self._engine.register_tasks(objectives, start_step=0)
        self._issue_directives(env)
        return self.state

    def update(self, env):
        """Advance mission state by one step. Call after each `env.step()`."""
        self.state.step += 1
        self._refresh_objectives(env)
        self.state.phase = self._derive_phase(env)
        self._phase_steps[self.state.phase] += 1
        self._issue_directives(env)
        # Update the coordination engine and store its snapshot in MissionState
        coord = self._engine.update(env, self.state.phase)
        self.state.coordination = coord
        return self.state

    def mission_summary(self):
        """End-of-mission report for logging and dashboards."""
        task_info = self._engine.task_summary()
        return {
            "final_phase"          : self.state.phase.name,
            "objectives_completed" : self.state.completed_count(),
            "objectives_total"     : len(self.state.objectives),
            "steps"                : self.state.step,
            "phase_steps"          : {p.name: n for p, n in self._phase_steps.items()
                                      if n > 0},
            # Task-layer coordination summary (Feature 2)
            "mission_progress"     : task_info["mission_progress"],
            "tasks_completed"      : task_info["tasks_completed"],
            "tasks_failed"         : task_info["tasks_failed"],
            "tasks_pending"        : task_info["tasks_pending"],
            "total_tasks"          : task_info["total_tasks"],
        }

    # ── Internal: state refresh ───────────────────────────────────────

    def _refresh_objectives(self, env):
        """Advance objective statuses from env telemetry (monotonic only)."""
        completed_ids = {ti for (_, ti) in env.targets_reached}
        alive         = [d for d in env.drones if d.alive]

        for obj in self.state.objectives.values():
            if obj.status is ObjectiveStatus.COMPLETED:
                continue
            if obj.objective_id in completed_ids:
                obj.status = ObjectiveStatus.COMPLETED
            elif obj.status is ObjectiveStatus.PENDING and any(
                    np.linalg.norm(d.pos - obj.position) < PLANNER_ENGAGE_RADIUS
                    for d in alive):
                obj.status = ObjectiveStatus.ENGAGED

    def _derive_phase(self, env):
        alive = [d for d in env.drones if d.alive]
        if not alive:
            return MissionPhase.IDLE
        if self.state.phase is MissionPhase.RETURN:      # absorbing
            return MissionPhase.RETURN
        if self.state.all_objectives_complete:
            return MissionPhase.RETURN
        if np.mean([d.energy for d in alive]) < PLANNER_RETURN_ENERGY:
            return MissionPhase.RETURN
        if any(o.status is not ObjectiveStatus.PENDING
               for o in self.state.objectives.values()):
            return MissionPhase.RESCUE
        return MissionPhase.SEARCH

    # ── Internal: directive issue ─────────────────────────────────────

    def _issue_directives(self, env):
        phase     = self.state.phase
        alive_ids = [d.id for d in env.drones if d.alive]
        positions = {d.id: d.pos for d in env.drones}
        directives = {}

        if phase is MissionPhase.RESCUE:
            tasking = self.allocator.assign(self.state.unfinished_objectives(),
                                            positions, alive_ids)
        else:
            tasking = {}

        for rank, d in enumerate(env.drones):
            if not d.alive:
                directives[d.id] = DroneDirective(d.id, MissionPhase.IDLE)
            elif phase is MissionPhase.SEARCH:
                # Round-robin zone coverage — spreads the swarm before
                # any objective has been located.
                zone = rank % self.zones.n_zones
                directives[d.id] = DroneDirective(d.id, phase, zone_id=zone)
            elif phase is MissionPhase.RESCUE:
                directives[d.id] = DroneDirective(d.id, phase,
                                                  objective_id=tasking.get(d.id))
            else:   # RETURN / IDLE — regroup toward the central zone
                home = self.zones.zone_of(np.zeros(2))
                directives[d.id] = DroneDirective(d.id, phase, zone_id=home)

        self.state.directives = directives
