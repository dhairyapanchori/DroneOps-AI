"""
ml.planner — Hierarchical Mission Planner and Task Coordination Engine.

Public API
----------
MissionPlanner          : Top-level mission command layer.
TaskCoordinationEngine  : Task lifecycle management and drone assignment.
MultiCriteriaAllocator  : Default multi-criteria weighted-score allocator.
TaskAllocationStrategy  : Protocol — implement to plug in a custom strategy.
MissionState            : Mission snapshot (phase, objectives, directives, coordination).
CoordinationState       : Task/drone coordination snapshot.
MissionPhase            : Swarm-level operating mode enum.
ObjectiveStatus         : Objective lifecycle enum.
TaskStatus              : Task lifecycle enum.
DroneStatus             : Drone operational status enum.
MissionTask             : Atomic task data model.
DroneCapacity           : Live drone capacity snapshot.
DroneDirective          : Per-drone high-level order.
MissionObjective        : Per-objective tracking.
ZoneGrid                : Spatial search-zone grid.
"""

from ml.planner.mission_planner import MissionPlanner
from ml.planner.mission_state import (
    MissionPhase,
    MissionState,
    MissionObjective,
    ObjectiveStatus,
    DroneDirective,
    ZoneGrid,
)
from ml.planner.coordination_engine import TaskCoordinationEngine, CoordinationState
from ml.planner.allocation_engine import TaskAllocationStrategy, MultiCriteriaAllocator
from ml.planner.task import MissionTask, TaskStatus, DroneStatus, DroneCapacity

__all__ = [
    # Core planner
    "MissionPlanner",
    "MissionState",
    "MissionPhase",
    "MissionObjective",
    "ObjectiveStatus",
    "DroneDirective",
    "ZoneGrid",
    # Coordination engine
    "TaskCoordinationEngine",
    "CoordinationState",
    # Allocation
    "TaskAllocationStrategy",
    "MultiCriteriaAllocator",
    # Task primitives
    "MissionTask",
    "TaskStatus",
    "DroneStatus",
    "DroneCapacity",
]
