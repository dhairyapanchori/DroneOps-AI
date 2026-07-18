"""
Central hyperparameter configuration.

Everything tunable lives here so experiments are a one-file edit.
FUSED_DIM must stay consistent with the fusion pipeline:
STATE_DIM (MetaAdapter, passthrough shape) + 64 (GNN) + 64 (Transformer).
"""

# ── Core dimensions ───────────────────────────────────────────────────
NUM_DRONES = 6
STATE_DIM  = 16
ACTION_DIM = 4
FUSED_DIM  = 144   # 16 + 64 + 64

# ── SAC hyperparameters ───────────────────────────────────────────────
LR_ACTOR  = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA  = 1e-4   # slower — prevents alpha decaying to zero too fast
GAMMA     = 0.99
TAU       = 0.005

BUFFER_SIZE  = 200000
BATCH        = 256
WARMUP_STEPS = 1000

# SAC target entropy: -action_dim (standard heuristic)
TARGET_ENTROPY = -float(ACTION_DIM)

# Reward scaling — keeps Q-values in [-10, +10] range
# Raw rewards clip to [-3, 8]; divide by REWARD_SCALE before storing
REWARD_SCALE = 5.0

MAX_EPISODES = 500   # more episodes for SAC to converge
MAX_STEPS    = 150

# ── Drone failure simulation ──────────────────────────────────────────
FAILURE_PROB  = 0.03
MAX_FAILURES  = 1

# ── Evolution engine ──────────────────────────────────────────────────
EVOLVE_EVERY  = 50
EVOLVE_POP    = 8
EVOLVE_SIGMA  = 0.02

# ── Hierarchical mission planner ──────────────────────────────────────
PLANNER_ZONE_GRID     = 2      # world divided into a 2×2 grid of search zones
PLANNER_ENGAGE_RADIUS = 3.0    # objective counts as ENGAGED within this range
PLANNER_RETURN_ENERGY = 0.25   # mean swarm energy that triggers RETURN phase

# ── Task Allocation & Coordination Engine (Feature 2) ─────────────────
# MultiCriteriaAllocator scoring weights — adjust to bias the allocator.
# score(task, drone) = w_dist × proximity + w_batt × battery − w_load × workload
TASK_ALLOC_W_DIST  = 1.0   # proximity weight  (1 / (1 + distance))
TASK_ALLOC_W_BATT  = 0.5   # battery weight    (favours healthier drones)
TASK_ALLOC_W_LOAD  = 0.3   # workload penalty  (discourages busy drones)
# Drone health thresholds
TASK_LOW_BATTERY   = 0.20  # energy below this → DroneStatus.LOW_BATTERY