"""
swarm_dashboard_demo.py
================================================================================
Standalone demo dashboard -- 7 rotating scenes, loops forever.
Zero changes to existing code. Loads same .pth files as original.

Scenes:
  1. Open Field        -- baseline, no obstacles
  2. Ring Gauntlet     -- 6 obstacles in ring, targets inside
  3. Cross Formation   -- 4 obstacles cross, targets in corners
  4. Hex Grid          -- 7 obstacles hexagonal, targets between nodes
  5. Energy Crisis     -- sparse obstacles, drones at 25% energy
  6. Binomial Failure  -- pre-killed drones, survivors cover targets
  7. Star Pattern      -- star layout, target focus buttons active

Run:
  python swarm_dashboard_demo.py
================================================================================
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import torch

from env.swarm_env import (SwarmEnv, OBSTACLE_RADIUS, TARGET_RADIUS,
                           OBSTACLE_PENALTY)
from ml.marl.actor import Actor
from ml.gnn.swarm_gnn import SwarmGNN
from ml.transformer.mission_transformer import MissionTransformer
from ml.meta.meta_adapter import MetaAdapter
from utils.config import (NUM_DRONES, STATE_DIM, FUSED_DIM,
                           ACTION_DIM, MAX_STEPS)

# ── Scene definitions ─────────────────────────────────────────────────────────

SCENES = [
    {
        "id": 1,
        "name": "Open Field",
        "subtitle": "Baseline — No Obstacles",
        "description": (
            "No obstacles. Targets placed randomly across the map.\n"
            "All 6 drones start at full energy.\n\n"
            "Watch how drones naturally spread to cover\n"
            "different targets using learned coordination.\n"
            "Green lines show communication links between drones."
        ),
        "obstacles": np.zeros((0, 2)),
        "targets": None,        # None = random placement
        "energy_start": 1.0,
        "pre_kill": 0,
        "focus_scene": False,
        "random_scene": False,
    },
    {
        "id": 2,
        "name": "Ring Gauntlet",
        "subtitle": "6 Obstacles in a Ring — Targets Inside",
        "description": (
            "6 obstacles form a ring around the centre.\n"
            "3 targets are placed inside the ring.\n\n"
            "Drones must find and navigate through the gaps\n"
            "to reach targets. Watch for the red penalty glow\n"
            "when drones graze obstacle edges.\n"
            "penalty: value shows the exact reward deducted."
        ),
        "obstacles": np.array([
            [ 6.0,  0.0], [-6.0,  0.0],
            [ 3.0,  5.2], [-3.0,  5.2],
            [ 3.0, -5.2], [-3.0, -5.2],
        ]),
        "targets": np.array([
            [ 0.0,  2.5], [-2.2, -1.5], [ 2.2, -1.5]
        ]),
        "energy_start": 1.0,
        "pre_kill": 0,
        "focus_scene": False,
        "random_scene": False,
    },
    {
        "id": 3,
        "name": "Cross Formation",
        "subtitle": "4 Obstacles in a Cross — Targets in Corners",
        "description": (
            "4 obstacles arranged in a cross block the map centre.\n"
            "3 targets are placed in separate quadrant corners.\n\n"
            "Drones must navigate around the cross arms\n"
            "and spread across quadrants to cover all targets.\n"
            "Watch coordination as drones pick separate routes."
        ),
        "obstacles": np.array([
            [ 0.0,  5.5], [ 0.0, -5.5],
            [ 5.5,  0.0], [-5.5,  0.0],
        ]),
        "targets": np.array([
            [ 6.0,  6.0], [-6.0,  6.0], [ 6.0, -6.0]
        ]),
        "energy_start": 1.0,
        "pre_kill": 0,
        "focus_scene": False,
        "random_scene": False,
    },
    {
        "id": 4,
        "name": "Hex Grid",
        "subtitle": "7 Obstacles — Hexagonal Pattern",
        "description": (
            "7 obstacles in a hexagonal grid pattern.\n"
            "Targets placed between grid nodes.\n\n"
            "The hex grid forces drones to weave between\n"
            "columns of obstacles. Visually shows how the\n"
            "learned policy navigates structured environments.\n"
            "Watch the penalty glow when drones cut corners."
        ),
        "obstacles": np.array([
            [ 0.0,  0.0],
            [ 5.0,  0.0], [-5.0,  0.0],
            [ 2.5,  4.3], [-2.5,  4.3],
            [ 2.5, -4.3], [-2.5, -4.3],
        ]),
        "targets": np.array([
            [ 7.5,  4.3], [-7.5,  4.3], [ 0.0, -8.0]
        ]),
        "energy_start": 1.0,
        "pre_kill": 0,
        "focus_scene": False,
        "random_scene": False,
    },
    {
        "id": 5,
        "name": "Energy Crisis",
        "subtitle": "Drones Start at 25% Energy",
        "description": (
            "All 6 drones start with only 25% energy remaining.\n"
            "Obstacles are sparse — energy is the real threat.\n\n"
            "Watch drones run out of energy mid-episode and\n"
            "go offline one by one. Surviving drones automatically\n"
            "redistribute to cover remaining targets.\n"
            "Energy bars drain visibly on the right panel."
        ),
        "obstacles": np.array([
            [ 5.0,  5.0], [-5.0, -5.0], [ 5.0, -5.0]
        ]),
        "targets": None,
        "energy_start": 0.25,
        "pre_kill": 0,
        "focus_scene": False,
        "random_scene": False,
    },
    {
        "id": 6,
        "name": "Binomial Failure",
        "subtitle": "Pre-Episode Drone Failures — Survivors Must Adapt",
        "description": (
            "2-3 drones are killed before the episode starts.\n"
            "This simulates real-world pre-flight failures.\n\n"
            "The surviving drones must cover all 3 targets\n"
            "with a reduced team. Watch how the swarm adapts —\n"
            "survivors spread further and work harder to\n"
            "compensate for lost teammates.\n"
            "Failed drones shown as grey X marks."
        ),
        "obstacles": np.array([
            [ 6.0,  0.0], [-6.0,  0.0],
            [ 3.0,  5.2], [-3.0,  5.2],
            [ 3.0, -5.2],
        ]),
        "targets": np.array([
            [ 0.0,  5.0], [-4.5, -3.0], [ 4.5, -3.0]
        ]),
        "energy_start": 1.0,
        "pre_kill": 3,          # exactly 3 drones pre-killed
        "focus_scene": False,
        "random_scene": False,
    },
    {
        "id": 7,
        "name": "Star Pattern",
        "subtitle": "Target Focus Active -- Click T0 / T1 / T2",
        "description": (
            "6 obstacles in a star/snowflake arrangement.\n"
            "3 targets placed at the star tips, well separated.\n\n"
            "USE THE BUTTONS: Click T0, T1, or T2 to focus\n"
            "the swarm on a specific target. The selected target\n"
            "pulses gold and drones prioritise it.\n"
            "Click CLEAR to restore normal behaviour.\n"
            "This scene is designed to make focus effect obvious."
        ),
        "obstacles": np.array([
            [ 0.0,  5.5], [ 0.0, -5.5],
            [ 4.8,  2.8], [-4.8,  2.8],
            [ 4.8, -2.8], [-4.8, -2.8],
        ]),
        "targets": np.array([
            [ 0.0,  8.5], [-7.5, -4.5], [ 7.5, -4.5]
        ]),
        "energy_start": 1.0,
        "pre_kill": 0,
        "focus_scene": True,
        "random_scene": False,
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
class SwarmDashboardDemo:

    def __init__(self,
                 actor_path="actor_trained.pth",
                 gnn_path="gnn_trained.pth",
                 trans_path="trans_trained.pth",
                 meta_path="meta_trained.pth"):

        self.env   = SwarmEnv()
        self.actor = Actor(FUSED_DIM, ACTION_DIM)
        self.gnn   = SwarmGNN(STATE_DIM)
        self.trans = MissionTransformer(STATE_DIM)
        self.meta  = MetaAdapter(STATE_DIM)

        loaded = []
        for path, net, name in [
            (actor_path, self.actor, "actor"),
            (gnn_path,   self.gnn,   "gnn"),
            (trans_path, self.trans,  "transformer"),
            (meta_path,  self.meta,   "meta"),
        ]:
            if path and os.path.exists(path):
                net.load_state_dict(torch.load(path, weights_only=True))
                loaded.append(name)
        print(f"[Demo] Loaded: {', '.join(loaded) if loaded else 'none'}")
        for net in [self.actor, self.gnn, self.trans, self.meta]:
            net.eval()

        # State
        self.scene_idx       = 0
        self.episode_num     = 0
        self.reward_history  = []
        self.episode_rewards = []
        self.selected_target = -1      # -1 = no focus
        self.pulse_tick      = 0       # for target pulse animation

        # Transient visual elements cleared each frame
        self._penalty_patches  = []
        self._penalty_texts    = []
        self._obs_flash        = []
        self._comm_lines       = []
        self._quivers          = []
        self._drone_labels     = []
        self._obs_rings        = []
        self._tgt_rings        = []
        self._pulse_ring       = None

        self._build_figure()

    # ── Figure ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor="#0d0d0d")
        self.fig.canvas.manager.set_window_title(
            "DroneOps AI -- Demo Dashboard")

        gs = GridSpec(4, 3, figure=self.fig,
                      left=0.04, right=0.97, top=0.94, bottom=0.10,
                      wspace=0.35, hspace=0.55)

        self.ax_main   = self.fig.add_subplot(gs[:, :2])
        self.ax_reward = self.fig.add_subplot(gs[0, 2])
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        self.ax_info   = self.fig.add_subplot(gs[2, 2])
        self.ax_btn    = self.fig.add_subplot(gs[3, 2])

        for ax in [self.ax_main, self.ax_reward,
                   self.ax_energy, self.ax_info, self.ax_btn]:
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")

        plt.suptitle(
            "[DRONEOPS] DroneOps AI -- SAC + GNN + Transformer + Evolution | Demo Mode",
            color="#00ff88", fontsize=9, fontweight="bold")

        self._setup_main_panel()
        self._setup_reward_panel()
        self._setup_energy_panel()
        self._setup_info_panel()
        self._setup_buttons()

    def _setup_main_panel(self):
        ax = self.ax_main
        ax.set_xlim(-13, 13); ax.set_ylim(-13, 13)
        ax.set_aspect("equal")
        ax.set_title("Swarm Environment", color="#aaaaaa", fontsize=9)
        ax.grid(True, color="#1a1a1a", linewidth=0.5)
        ax.axhline(0, color="#222222", lw=0.8)
        ax.axvline(0, color="#222222", lw=0.8)

        self.drone_scatter  = ax.scatter([], [], s=90,  c="#00ff88", zorder=6,
                                          label="Active drones")
        self.dead_scatter   = ax.scatter([], [], s=90,  c="#444444",
                                          marker="x", zorder=6, label="Failed drones")
        self.obs_scatter    = ax.scatter([], [], s=130, c="#ff4444",
                                          marker="X", zorder=4, label="Obstacles")
        self.tgt_scatter    = ax.scatter([], [], s=160, c="#4488ff",
                                          marker="*", zorder=5, label="Targets")
        ax.legend(loc="upper right", fontsize=6,
                  facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#cccccc")

        # Splash overlay text (hidden until needed)
        self.splash_text = ax.text(
            0, 0, "", color="white", fontsize=9,
            va="center", ha="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#0d0d0d",
                      edgecolor="#00ff88", linewidth=2, alpha=0.95),
            zorder=20, visible=False
        )

    def _setup_reward_panel(self):
        self.ax_reward.set_title("Reward / Step", color="#aaaaaa", fontsize=8)
        self.reward_line, = self.ax_reward.plot([], [], color="#00ff88", lw=1)

    def _setup_energy_panel(self):
        self.ax_energy.set_title("Drone Energy", color="#aaaaaa", fontsize=8)
        self.ax_energy.set_ylim(0, 1.05)
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, NUM_DRONES))
        self.energy_bars = self.ax_energy.bar(
            range(NUM_DRONES), [1.0]*NUM_DRONES, color=colors, width=0.6)
        self.ax_energy.set_xticks(range(NUM_DRONES))
        self.ax_energy.set_xticklabels(
            [f"D{i}" for i in range(NUM_DRONES)], fontsize=6, color="#888888")

    def _setup_info_panel(self):
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.05, 0.97, "",
            transform=self.ax_info.transAxes,
            color="#cccccc", fontsize=7.5,
            va="top", fontfamily="monospace")

    def _setup_buttons(self):
        self.ax_btn.axis("off")
        # Four buttons: T0 T1 T2 CLEAR
        # Placed manually using figure add_axes
        btn_y  = 0.04
        btn_h  = 0.055
        btn_w  = 0.055
        gap    = 0.005
        starts = [0.545, 0.605, 0.665, 0.740]
        labels = ["T0", "T1", "T2", "CLEAR"]
        colors = ["#1a3a5c", "#1a3a5c", "#1a3a5c", "#3a1a1a"]

        self.btn_axes   = []
        self.btn_objs   = []
        for i, (x, lbl, col) in enumerate(zip(starts, labels, colors)):
            ax  = self.fig.add_axes([x, btn_y, btn_w, btn_h])
            btn = Button(ax, lbl,
                         color=col, hovercolor="#2a5a8c" if i < 3 else "#6a2a2a")
            btn.label.set_color("white")
            btn.label.set_fontsize(8)
            self.btn_axes.append(ax)
            self.btn_objs.append(btn)

        self.btn_objs[0].on_clicked(lambda e: self._set_focus(0))
        self.btn_objs[1].on_clicked(lambda e: self._set_focus(1))
        self.btn_objs[2].on_clicked(lambda e: self._set_focus(2))
        self.btn_objs[3].on_clicked(lambda e: self._set_focus(-1))

        # Label above buttons
        self.fig.text(0.645, 0.105, "TARGET FOCUS", color="#888888",
                      fontsize=7, ha="center", fontfamily="monospace")

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(self, s_t):
        adapted = self.meta(s_t)
        adj     = torch.ones(NUM_DRONES, NUM_DRONES)
        gnn_ctx = self.gnn(adapted, adj)
        mission = self.trans(adapted)
        return torch.cat([adapted, gnn_ctx, mission], dim=-1)

    def _get_actions(self, s):
        with torch.no_grad():
            s_t   = torch.FloatTensor(s)
            fused = self._fuse(s_t)
            if hasattr(self.actor, "deterministic"):
                actions = self.actor.deterministic(fused)
                # Fallback to stochastic sampling if the policy mean has collapsed
                # to near-zero (untrained or poorly-converged checkpoint).
                if actions.abs().mean() < 0.05:
                    actions = self.actor.sample(fused)[0]
                return actions.numpy()
            return self.actor(fused).numpy()
        with torch.no_grad():
            s_t   = torch.FloatTensor(s)
            fused = self._fuse(s_t)
            if hasattr(self.actor, "deterministic"):
                return self.actor.deterministic(fused).numpy()
            return self.actor(fused).numpy()

    # ── Focus ─────────────────────────────────────────────────────────────────

    def _set_focus(self, idx):
        self.selected_target = idx
        # Update button colours to show which is active
        focus_col = "#00884a"
        base_cols = ["#1a3a5c", "#1a3a5c", "#1a3a5c", "#3a1a1a"]
        for i, btn in enumerate(self.btn_objs):
            if i < 3:
                btn.ax.set_facecolor(focus_col if i == idx else base_cols[i])
            else:
                btn.ax.set_facecolor("#00442a" if idx == -1 else base_cols[3])
        self.fig.canvas.draw_idle()

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _load_scene(self, scene):
        """Apply scene config to env before episode."""
        self.env.curriculum_ep = 225   # Phase 3 always

        # Reset env normally first to get drone objects initialised
        self.env.reset()

        # Override obstacles
        self.env.obstacles = scene["obstacles"].copy() \
            if len(scene["obstacles"]) > 0 else np.zeros((0, 2))

        # Override targets
        if scene["targets"] is not None:
            self.env.targets = scene["targets"].copy()
        else:
            # Random targets clear of obstacles
            r = 7
            tgts, attempts = [], 0
            while len(tgts) < 3 and attempts < 500:
                attempts += 1
                c = np.random.uniform(-r, r, 2)
                if len(self.env.obstacles) > 0:
                    if any(np.linalg.norm(c - o) < OBSTACLE_RADIUS + 2.0
                           for o in self.env.obstacles):
                        continue
                if any(np.linalg.norm(c - t) < 4.0 for t in tgts):
                    continue
                tgts.append(c)
            while len(tgts) < 3:
                tgts.append(np.random.uniform(-6, 6, 2))
            self.env.targets = np.array(tgts)

        # Spawn drones from bottom-left corner in a tight cluster
        corner      = np.array([-9.0, -9.0])
        min_obs     = OBSTACLE_RADIUS + 1.5
        placed      = []
        for d in self.env.drones:
            attempts = 0
            while attempts < 400:
                attempts += 1
                offset = np.random.uniform(0, 3.5, 2)
                c      = corner + offset
                c      = np.clip(c, -11, 11)
                if len(self.env.obstacles) > 0:
                    if any(np.linalg.norm(c - o) < min_obs
                           for o in self.env.obstacles):
                        continue
                if any(np.linalg.norm(c - p) < 1.8 for p in placed):
                    continue
                d.pos = c.astype(float)
                placed.append(c)
                break
            else:
                d.pos = np.array([-8.0 + len(placed)*1.5, -8.0])
                placed.append(d.pos.copy())

        # Apply energy start
        for d in self.env.drones:
            d.energy = scene["energy_start"]

        # Pre-kill drones (binomial scene)
        n_kill = scene["pre_kill"]
        if n_kill > 0:
            kill_ids = np.random.choice(NUM_DRONES, n_kill, replace=False)
            for kid in kill_ids:
                self.env.drones[int(kid)].alive  = False
                self.env.drones[int(kid)].energy = 0.0
                self.env.failed_ids.add(int(kid))

        return self.env.states()

    # ── Splash screen ─────────────────────────────────────────────────────────

    def _show_splash(self, scene, duration=4.0):
        text = (
            f"{'='*46}\n"
            f"  SCENE {scene['id']} -- {scene['name']}\n"
            f"  {scene['subtitle']}\n"
            f"{'-'*46}\n\n"
            f"  [X] Red X    -- Obstacle (penalty radius 2.0)\n"
            f"  [*] Blue *   -- Target   (reach radius 1.5)\n"
            f"  [o] Green o  -- Active drone\n"
            f"  [x] Grey x   -- Failed drone\n\n"
            f"{self._wrap(scene['description'], 44)}\n\n"
            f"  Starting in {duration:.0f} seconds...\n"
            f"{'='*46}"
        )
        self.splash_text.set_text(text)
        self.splash_text.set_visible(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Countdown
        steps  = 20
        dt     = duration / steps
        for i in range(steps):
            remaining = duration - (i + 1) * dt
            lines     = text.split("\n")
            lines[-2] = f"  Starting in {max(0, remaining):.1f} seconds..."
            self.splash_text.set_text("\n".join(lines))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(dt)

        self.splash_text.set_visible(False)

    def _wrap(self, text, width):
        """Indent multiline description for splash."""
        lines = text.split("\n")
        out   = []
        for line in lines:
            if len(line) <= width:
                out.append(f"  {line}")
            else:
                words, cur = line.split(), ""
                for w in words:
                    if len(cur) + len(w) + 1 <= width:
                        cur = f"{cur} {w}" if cur else w
                    else:
                        out.append(f"  {cur}")
                        cur = w
                if cur:
                    out.append(f"  {cur}")
        return "\n".join(out)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _clear_transients(self):
        for p in self._penalty_patches: p.remove()
        for t in self._penalty_texts:   t.remove()
        for f in self._obs_flash:       f.remove()
        for l in self._comm_lines:      l.remove()
        for q in self._quivers:         q.remove()
        for l in self._drone_labels:    l.remove()
        self._penalty_patches = []
        self._penalty_texts   = []
        self._obs_flash       = []
        self._comm_lines      = []
        self._quivers         = []
        self._drone_labels    = []

    def _draw_static_rings(self):
        for r in self._obs_rings + self._tgt_rings:
            r.remove()
        self._obs_rings = []
        self._tgt_rings = []

        for obs in self.env.obstacles:
            c = Circle(obs, OBSTACLE_RADIUS,
                       color="#ff4444", fill=True, alpha=0.10, zorder=2)
            self.ax_main.add_patch(c)
            self._obs_rings.append(c)
            c2 = Circle(obs, OBSTACLE_RADIUS,
                        color="#ff4444", fill=False,
                        linewidth=1.5, alpha=0.5, zorder=3)
            self.ax_main.add_patch(c2)
            self._obs_rings.append(c2)

        for ti, tgt in enumerate(self.env.targets):
            color = "#4488ff"
            c = Circle(tgt, TARGET_RADIUS,
                       color=color, fill=True, alpha=0.15, zorder=2)
            self.ax_main.add_patch(c)
            self._tgt_rings.append(c)

    def _draw_penalty_glow(self, drone):
        """Draw red glow + penalty text when drone is inside obstacle zone."""
        if not drone.alive or len(self.env.obstacles) == 0:
            return
        dists = [np.linalg.norm(drone.pos - o) for o in self.env.obstacles]
        min_dist = min(dists)
        if min_dist < OBSTACLE_RADIUS:
            penetration = 1.0 - min_dist / OBSTACLE_RADIUS
            penalty_val = OBSTACLE_PENALTY * penetration

            glow_r = 0.6 + penetration * 0.8
            glow   = Circle(drone.pos, glow_r,
                            color="#ff2200", fill=True,
                            alpha=0.55 + penetration * 0.2, zorder=8)
            self.ax_main.add_patch(glow)
            self._penalty_patches.append(glow)

            obs_idx  = int(np.argmin(dists))
            obs_pos  = self.env.obstacles[obs_idx]
            flash    = Circle(obs_pos, OBSTACLE_RADIUS,
                              color="#ff4444", fill=True,
                              alpha=0.30 + penetration * 0.25, zorder=2)
            self.ax_main.add_patch(flash)
            self._obs_flash.append(flash)

            txt = self.ax_main.text(
                drone.pos[0] + 0.6, drone.pos[1] + 1.0,
                f"penalty: -{penalty_val:.2f}",
                color="#ff6666", fontsize=6.5, zorder=9,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#1a0000", alpha=0.75,
                          edgecolor="none"))
            self._penalty_texts.append(txt)

    def _draw_pulse_ring(self):
        """Pulsing gold ring on selected target (Scene 7)."""
        if self._pulse_ring is not None:
            try:
                self._pulse_ring.remove()
            except Exception:
                pass
            self._pulse_ring = None

        if (self.selected_target < 0 or
                self.selected_target >= len(self.env.targets)):
            return

        self.pulse_tick += 1
        pulse_r = TARGET_RADIUS + 0.4 + 0.35 * np.sin(self.pulse_tick * 0.25)
        alpha   = 0.55 + 0.35 * np.sin(self.pulse_tick * 0.25)
        tgt_pos = self.env.targets[self.selected_target]
        ring    = Circle(tgt_pos, pulse_r,
                         color="#ffd700", fill=False,
                         linewidth=2.5, alpha=alpha, zorder=7)
        self.ax_main.add_patch(ring)
        self._pulse_ring = ring

    def _draw_comm_lines(self, positions, alive_mask, threshold=8.0):
        for i in range(NUM_DRONES):
            for j in range(i + 1, NUM_DRONES):
                if not (alive_mask[i] and alive_mask[j]):
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold:
                    alpha = max(0.05, 1.0 - dist / threshold) * 0.5
                    ln, = self.ax_main.plot(
                        [positions[i][0], positions[j][0]],
                        [positions[i][1], positions[j][1]],
                        color="#00ff88", lw=0.4, alpha=alpha)
                    self._comm_lines.append(ln)

    def _draw_quivers(self):
        for d in self.env.drones:
            if not d.alive:
                continue
            q = self.ax_main.quiver(
                d.pos[0], d.pos[1], d.vel[0], d.vel[1],
                color="#ffdd44", scale=8, width=0.004, alpha=0.7)
            self._quivers.append(q)

    def _draw_labels(self, positions, alive_mask):
        for i, pos in enumerate(positions):
            color = "#88ffcc" if alive_mask[i] else "#555555"
            label = f"D{i}" if alive_mask[i] else f"D{i}✗"
            l = self.ax_main.text(
                pos[0] + 0.4, pos[1] + 0.4, label,
                color=color, fontsize=6, zorder=7)
            self._drone_labels.append(l)

    # ── Frame update ──────────────────────────────────────────────────────────

    def _update_frame(self, step_reward, step, scene):
        self._clear_transients()

        positions  = np.array([d.pos    for d in self.env.drones])
        energies   = np.array([d.energy for d in self.env.drones])
        alive_mask = np.array([d.alive  for d in self.env.drones])

        active_pos = positions[alive_mask]
        dead_pos   = positions[~alive_mask]

        if len(active_pos) > 0:
            active_colors = plt.cm.RdYlGn(
                np.clip(energies[alive_mask], 0, 1))
            self.drone_scatter.set_offsets(active_pos)
            self.drone_scatter.set_color(active_colors)
        else:
            self.drone_scatter.set_offsets(np.empty((0, 2)))

        self.dead_scatter.set_offsets(
            dead_pos if len(dead_pos) > 0 else np.empty((0, 2)))

        tgt_colors = []
        for ti in range(len(self.env.targets)):
            hit = any(k[1] == ti for k in self.env.targets_reached)
            if ti == self.selected_target and scene["focus_scene"]:
                tgt_colors.append("#ffd700")
            elif hit:
                tgt_colors.append("#ffd700")
            else:
                tgt_colors.append("#4488ff")

        if len(self.env.targets) > 0:
            self.tgt_scatter.set_offsets(
                np.array(self.env.targets).reshape(-1, 2))
            self.tgt_scatter.set_color(tgt_colors)

        if len(self.env.obstacles) > 0:
            self.obs_scatter.set_offsets(
                np.array(self.env.obstacles).reshape(-1, 2))
        else:
            self.obs_scatter.set_offsets(np.empty((0, 2)))

        for d in self.env.drones:
            self._draw_penalty_glow(d)

        if scene["focus_scene"]:
            self._draw_pulse_ring()

        self._draw_comm_lines(positions, alive_mask)
        self._draw_quivers()
        self._draw_labels(positions, alive_mask)

        self.reward_history.append(step_reward)
        self.reward_line.set_data(
            range(len(self.reward_history)), self.reward_history)
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()

        for i, (bar, e) in enumerate(zip(self.energy_bars, energies)):
            if alive_mask[i]:
                bar.set_height(max(0, e))
                bar.set_color(plt.cm.RdYlGn(np.clip(e, 0, 1)))
            else:
                bar.set_height(0.02)
                bar.set_color("#333333")

        n_active = int(alive_mask.sum())
        n_failed = NUM_DRONES - n_active
        n_tgts   = len(self.env.targets_reached)
        covered  = len(set(ti for (_, ti) in self.env.targets_reached))
        coord    = covered / max(1, len(self.env.targets))

        avg_to_tgt = np.mean([
            min(np.linalg.norm(d.pos - t) for t in self.env.targets)
            for d in self.env.drones if d.alive
        ]) if n_active > 0 and len(self.env.targets) > 0 else 99.0

        focus_str = (f"Focus     : T{self.selected_target}\n"
                     if self.selected_target >= 0 else "")

        info = (
            f"Scene {scene['id']}: {scene['name']}\n"
            f"Episode   : {self.episode_num}\n"
            f"Step      : {step}/{MAX_STEPS}\n"
            f"────────────────────\n"
            f"Active    : {n_active}/{NUM_DRONES}\n"
            f"Failed    : {n_failed}\n"
            f"Targets   : {n_tgts}\n"
            f"Coord     : {coord:.2f}\n"
            f"Avg→Tgt   : {avg_to_tgt:.2f}\n"
            f"Reward    : {step_reward:+.2f}\n"
            f"────────────────────\n"
            f"{focus_str}"
        )
        if self.episode_rewards:
            info += f"Best Ep   : {max(self.episode_rewards):.1f}\n"
            info += f"Last Ep   : {self.episode_rewards[-1]:.1f}\n"

        self.info_text.set_text(info)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, pause=0.04):
        plt.ion()
        plt.show(block=False)
        plt.show()

        while True:                              # loop forever
            scene = SCENES[self.scene_idx % len(SCENES)]

            # Reset focus when leaving Scene 7
            if not scene["focus_scene"]:
                self.selected_target = -1
                self._set_focus(-1)

            # Show/hide focus buttons
            for i, btn in enumerate(self.btn_objs[:3]):
                btn.ax.set_visible(scene["focus_scene"])
            self.btn_objs[3].ax.set_visible(scene["focus_scene"])

            # Splash screen
            self._show_splash(scene, duration=4.0)

            # Load scene into env
            s = self._load_scene(scene)

            # Draw static rings for this episode
            self._draw_static_rings()

            self.reward_history = []
            ep_r  = 0.0
            done  = False
            step  = 0
            self.episode_num += 1

            while not done:
                actions = self._get_actions(s)
                s, rewards, done = self.env.step(actions)
                step_reward      = rewards.mean()
                ep_r            += step_reward
                step            += 1
                self._update_frame(step_reward, step, scene)
                plt.pause(pause)

            self.episode_rewards.append(ep_r)
            print(f"[MODEL ] Scene {scene['id']} | Ep {self.episode_num} | "
                  f"Reward={ep_r:.1f}  "
                  f"Targets={len(self.env.targets_reached)}  "
                  f"Failed={len(self.env.failed_ids)}")

            # Advance to next scene
            self.scene_idx += 1

        plt.ioff()
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dashboard = SwarmDashboardDemo(
        actor_path="actor_trained.pth",
        gnn_path="gnn_trained.pth",
        trans_path="trans_trained.pth",
        meta_path="meta_trained.pth",
    )
    dashboard.run(pause=0.04)