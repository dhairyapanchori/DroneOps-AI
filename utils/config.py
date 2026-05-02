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

# ── Coordination reward ───────────────────────────────────────────────
COORDINATION_BONUS = 0.15

# ── Evolution engine ──────────────────────────────────────────────────
EVOLVE_EVERY  = 50
EVOLVE_POP    = 8
EVOLVE_SIGMA  = 0.02

# ── Meta adapter ─────────────────────────────────────────────────────
META_LR = 3e-4

# ── Backwards compat ─────────────────────────────────────────────────
LR = LR_ACTOR
NOISE_SIGMA = 0.0   # SAC doesn't use OU noise
NOISE_DECAY = 1.0