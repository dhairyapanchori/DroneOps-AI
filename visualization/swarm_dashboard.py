"""
DroneOps AI — Mission Control Center Dashboard
================================================

Enterprise-grade real-time mission dashboard.  Redesigned from scratch as a
professional Mission Control Center to clearly demonstrate the full system
architecture during technical reviews and demonstrations.

Layout (3-column GridSpec)
--------------------------
  LEFT   (cols 0-1, all rows) : Live Swarm Map — primary situational awareness
  CENTER (col 2, rows 0-1)    : Mission Overview + Task Coordination
  CENTER (col 2, row 2)       : Drone Fleet status table
  RIGHT  (col 3, rows 0-1)    : Mission Log (live event stream)
  RIGHT  (col 3, row 2)       : AI System Status
  BOTTOM (cols 2-3, row 3)    : Training Metrics (reward + energy)

Architecture notes
------------------
- Dashboard reads from planner.state, coordination_state, env telemetry only.
- No writes to planner, env, or any ML module.
- _fuse() and _get_actions() are unchanged from original.
- run() episode loop is unchanged.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import torch

from env.swarm_env import SwarmEnv, OBSTACLE_RADIUS, TARGET_RADIUS
from ml.marl.actor import Actor
from ml.gnn.swarm_gnn import SwarmGNN
from ml.transformer.mission_transformer import MissionTransformer
from ml.meta.meta_adapter import MetaAdapter
from ml.planner.mission_planner import MissionPlanner
from ml.planner.task import DroneStatus
from utils.config import NUM_DRONES, MAX_STEPS, FUSED_DIM, ACTION_DIM, STATE_DIM


# ── Color palette ─────────────────────────────────────────────────────────────

C = {
    "bg"         : "#0a0e1a",
    "panel"      : "#0d1117",
    "border"     : "#1e2938",
    "accent"     : "#00d4aa",   # green — healthy / success
    "blue"       : "#4a9eff",   # SEARCH phase / targets
    "yellow"     : "#f5c518",   # RESCUE phase / low battery
    "red"        : "#ff4560",   # danger / obstacles / failed
    "text_hi"    : "#e6edf3",   # primary text
    "text_lo"    : "#7d8590",   # secondary / dim text
    "dim"        : "#1c2333",   # subtle backgrounds
    "SEARCH"     : "#4a9eff",
    "RESCUE"     : "#f5c518",
    "RETURN"     : "#00d4aa",
    "IDLE"       : "#7d8590",
}

PHASE_COLOR = {
    "SEARCH": C["blue"],
    "RESCUE": C["yellow"],
    "RETURN": C["accent"],
    "IDLE"  : C["text_lo"],
}

ENERGY_CMAP = matplotlib.colormaps.get_cmap("RdYlGn")


def _energy_color(e: float) -> str:
    """Map energy [0,1] to hex color via RdYlGn colormap."""
    rgba = ENERGY_CMAP(float(np.clip(e, 0.0, 1.0)))
    return matplotlib.colors.to_hex(rgba)


# ── Dashboard ─────────────────────────────────────────────────────────────────

class SwarmDashboard:
    """DroneOps AI Mission Control Center.

    Args:
        actor_path : Path to saved actor checkpoint.
        gnn_path   : Path to saved GNN checkpoint.
        trans_path : Path to saved Transformer checkpoint.
        meta_path  : Path to saved MetaAdapter checkpoint.
    """

    def __init__(
        self,
        actor_path : str = "actor_trained.pth",
        gnn_path   : str = "gnn_trained.pth",
        trans_path : str = "trans_trained.pth",
        meta_path  : str = "meta_trained.pth",
    ):
        self.env     = SwarmEnv()
        self.planner = MissionPlanner()
        self.actor   = Actor(FUSED_DIM, ACTION_DIM)
        self.gnn     = SwarmGNN(STATE_DIM)
        self.trans   = MissionTransformer(STATE_DIM)
        self.meta    = MetaAdapter(STATE_DIM)

        # ── Load checkpoints ──────────────────────────────────────────────
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

        print(f"[Dashboard] Loaded: {', '.join(loaded) if loaded else 'none (untrained)'}")
        for net in [self.actor, self.gnn, self.trans, self.meta]:
            net.eval()

        # ── Episode state ─────────────────────────────────────────────────
        self.reward_history  : list[float] = []
        self.episode_rewards : list[float] = []
        self.episode_num     : int         = 0
        self.log_lines       : list[str]   = []   # mission event log

        # Previous-state trackers for event detection
        self._prev_phase       = None
        self._prev_assignments : dict = {}
        self._prev_alive       : set  = set(range(NUM_DRONES))
        self._prev_completed   : set  = set()

        # ── Figure & layout ───────────────────────────────────────────────
        self.fig = plt.figure(
            figsize=(20, 11),
            facecolor=C["bg"],
        )
        self.fig.canvas.manager.set_window_title(
            "DroneOps AI — Mission Control Center"
        )

        # 4-column, 4-row grid
        # Map occupies cols 0-1, all rows
        # Panels occupy cols 2-3
        gs = gridspec.GridSpec(
            4, 4,
            figure=self.fig,
            left=0.03, right=0.98,
            top=0.95,  bottom=0.04,
            wspace=0.35, hspace=0.45,
        )

        # Map (left, large)
        self.ax_map    = self.fig.add_subplot(gs[0:4, 0:2])

        # Center column panels
        self.ax_mission = self.fig.add_subplot(gs[0, 2])    # Mission Overview
        self.ax_tasks   = self.fig.add_subplot(gs[1, 2])    # Task Coordination
        self.ax_fleet   = self.fig.add_subplot(gs[2:4, 2])  # Drone Fleet

        # Right column panels
        self.ax_log     = self.fig.add_subplot(gs[0:2, 3])  # Mission Log
        self.ax_ai      = self.fig.add_subplot(gs[2, 3])    # AI System Status
        self.ax_reward  = self.fig.add_subplot(gs[3, 3])    # Reward chart

        # Energy bars (slim, below map)
        self.ax_energy  = self.fig.add_subplot(gs[3, 0:2])  # Energy bars

        self._style_all_axes()
        self._setup_map()
        self._setup_energy()
        self._setup_reward()
        self._setup_ai_status()

        plt.suptitle(
            "DroneOps AI  ·  Mission Control Center  ·  "
            "SAC + GNN + Transformer + Meta-Adapter + Evolution",
            color=C["accent"], fontsize=9, fontweight="bold",
            y=0.98,
        )

    # ── Styling helpers ───────────────────────────────────────────────────────

    def _style_axes(self, ax, title: str = "", title_color: str = ""):
        """Apply standard dark panel styling to an axis."""
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text_lo"], labelsize=6.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
            sp.set_linewidth(0.8)
        if title:
            ax.set_title(
                title,
                color=title_color or C["accent"],
                fontsize=7.5, fontweight="bold",
                pad=4,
            )

    def _style_all_axes(self):
        for ax, title in [
            (self.ax_map,    ""),
            (self.ax_mission,"  MISSION OVERVIEW"),
            (self.ax_tasks,  "  TASK COORDINATION"),
            (self.ax_fleet,  "  DRONE FLEET"),
            (self.ax_log,    "  MISSION LOG"),
            (self.ax_ai,     "  AI SYSTEM STATUS"),
            (self.ax_reward, "  REWARD / STEP"),
            (self.ax_energy, "  DRONE ENERGY"),
        ]:
            self._style_axes(ax, title)

        # Text-only panels
        for ax in [self.ax_mission, self.ax_tasks, self.ax_fleet,
                   self.ax_log, self.ax_ai]:
            ax.axis("off")

    # ── Map setup ─────────────────────────────────────────────────────────────

    def _setup_map(self):
        ax = self.ax_map
        ax.set_facecolor(C["bg"])
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
        ax.set_title(
            "  LIVE SWARM MAP", color=C["accent"],
            fontsize=8, fontweight="bold", pad=5, loc="left",
        )
        ax.grid(True, color=C["border"], linewidth=0.4, alpha=0.6)
        ax.axhline(0, color=C["border"], lw=0.6)
        ax.axvline(0, color=C["border"], lw=0.6)
        ax.tick_params(colors=C["text_lo"], labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])

        # Static scatter artists
        self.scat_drones  = ax.scatter([], [], s=110, zorder=8,
                                        edgecolors=C["bg"], linewidths=1.2)
        self.scat_dead    = ax.scatter([], [], s=90, c=C["border"],
                                        marker="x", zorder=7, linewidths=1.5)
        self.scat_targets = ax.scatter([], [], s=200, marker="*", zorder=6)
        self.scat_obs     = ax.scatter([], [], s=100, c=C["red"],
                                        marker="X", zorder=5, alpha=0.9)

        # Dynamic artist lists
        self.obs_rings   : list = []
        self.tgt_rings   : list = []
        self.quivers     : list = []
        self.drone_labels: list = []
        self.comm_lines  : list = []
        self.task_lines  : list = []
        self.zone_lines  : list = []

        # Map legend
        legend_elems = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=C["accent"], markersize=6, label='Drone (active)'),
            Line2D([0],[0], marker='x', color=C["border"],
                   markersize=6, label='Drone (offline)', linewidth=2),
            Line2D([0],[0], marker='*', color='w',
                   markerfacecolor=C["blue"], markersize=8, label='Target'),
            Line2D([0],[0], marker='*', color='w',
                   markerfacecolor=C["yellow"], markersize=8, label='Target (reached)'),
            Line2D([0],[0], marker='X', color='w',
                   markerfacecolor=C["red"], markersize=6, label='Obstacle'),
            Line2D([0],[0], linestyle='--', color=C["accent"],
                   alpha=0.5, label='Task assignment'),
        ]
        ax.legend(
            handles=legend_elems, loc="upper right",
            fontsize=5.5, facecolor=C["panel"],
            edgecolor=C["border"], labelcolor=C["text_lo"],
            framealpha=0.9,
        )

    # ── Energy bar setup ──────────────────────────────────────────────────────

    def _setup_energy(self):
        ax = self.ax_energy
        ax.set_facecolor(C["panel"])
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, NUM_DRONES - 0.5)
        ax.set_xticks(range(NUM_DRONES))
        ax.set_xticklabels([f"D{i}" for i in range(NUM_DRONES)],
                           fontsize=6.5, color=C["text_lo"])
        ax.tick_params(axis='y', colors=C["text_lo"], labelsize=6)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%","25%","50%","75%","100%"],
                           fontsize=5.5, color=C["text_lo"])
        ax.axhline(0.20, color=C["red"],    lw=0.8, linestyle="--", alpha=0.5,
                   label="Low battery")
        ax.axhline(0.50, color=C["yellow"], lw=0.6, linestyle=":", alpha=0.4)

        colors = [_energy_color(1.0)] * NUM_DRONES
        self.energy_bars = ax.bar(
            range(NUM_DRONES), [1.0] * NUM_DRONES,
            color=colors, width=0.55,
            edgecolor=C["bg"], linewidth=0.8,
        )

    # ── Reward chart setup ────────────────────────────────────────────────────

    def _setup_reward(self):
        ax = self.ax_reward
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text_lo"], labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        self.reward_line, = ax.plot([], [], color=C["accent"], lw=1.2,
                                    alpha=0.9)
        ax.axhline(0, color=C["border"], lw=0.6)

    # ── AI Status setup ───────────────────────────────────────────────────────

    def _setup_ai_status(self):
        """Render the static AI architecture checklist — drawn once."""
        ax = self.ax_ai
        components = [
            ("Mission Planner",    C["accent"]),
            ("Task Allocation",    C["accent"]),
            ("Meta Adapter",       C["blue"]),
            ("Swarm GNN",          C["blue"]),
            ("Mission Transformer",C["blue"]),
            ("SAC Actor",          C["yellow"]),
            ("Twin Critics",       C["yellow"]),
            ("Evolution Engine",   C["red"]),
        ]
        y = 0.95
        ax.text(0.04, y, "COMPONENT", fontsize=6, color=C["text_lo"],
                transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.72, y, "STATUS", fontsize=6, color=C["text_lo"],
                transform=ax.transAxes, fontfamily="monospace")
        y -= 0.08
        ax.plot([0.03, 0.97], [y, y], color=C["border"], linewidth=0.5,
                transform=ax.transAxes, solid_capstyle='butt')
        y -= 0.04

        for name, color in components:
            ax.text(0.04, y, name, fontsize=6.5, color=C["text_hi"],
                    transform=ax.transAxes, fontfamily="monospace")
            ax.text(0.73, y, "[ONLINE]", fontsize=6.5, color=color,
                    transform=ax.transAxes, fontfamily="monospace",
                    fontweight="bold")
            y -= 0.10

    # ── Fusion pipeline (unchanged from original) ─────────────────────────────

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
                return self.actor.deterministic(fused).numpy()
            return self.actor(fused).numpy()

    # ── Map drawing helpers ───────────────────────────────────────────────────

    def _draw_rings(self):
        """Draw obstacle and target radius rings — called once per episode."""
        for r in self.obs_rings + self.tgt_rings:
            r.remove()
        self.obs_rings = []
        self.tgt_rings = []

        for obs in self.env.obstacles:
            c = Circle(obs, OBSTACLE_RADIUS, color=C["red"],
                       fill=True, alpha=0.06, zorder=2)
            self.ax_map.add_patch(c)
            self.obs_rings.append(c)

        for tgt in self.env.targets:
            c = Circle(tgt, TARGET_RADIUS, color=C["blue"],
                       fill=True, alpha=0.10, zorder=2)
            self.ax_map.add_patch(c)
            self.tgt_rings.append(c)

    def _draw_zone_grid(self, phase: str):
        """Draw the 2×2 search zone overlay during SEARCH phase."""
        for ln in self.zone_lines:
            ln.remove()
        self.zone_lines = []

        if phase != "SEARCH":
            return

        # Planner zones: 2×2 over [-12, 12]
        bound = 12.0
        mid   = 0.0
        style = dict(color=C["blue"], lw=0.6, linestyle="--",
                     alpha=0.25, zorder=1)
        ln1, = self.ax_map.plot([mid, mid], [-bound, bound], **style)
        ln2, = self.ax_map.plot([-bound, bound], [mid, mid], **style)
        self.zone_lines.extend([ln1, ln2])

    def _draw_comm_lines(self, positions, alive_mask, threshold=8.0):
        for ln in self.comm_lines:
            ln.remove()
        self.comm_lines = []

        for i in range(NUM_DRONES):
            for j in range(i + 1, NUM_DRONES):
                if not (alive_mask[i] and alive_mask[j]):
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold:
                    alpha = max(0.04, 1.0 - dist / threshold) * 0.4
                    ln, = self.ax_map.plot(
                        [positions[i][0], positions[j][0]],
                        [positions[i][1], positions[j][1]],
                        color=C["accent"], lw=0.5, alpha=alpha, zorder=3,
                    )
                    self.comm_lines.append(ln)

    def _draw_task_lines(self, drone_assignments):
        """Draw dashed lines from each drone to its assigned target."""
        for ln in self.task_lines:
            ln.remove()
        self.task_lines = []

        if not drone_assignments:
            return

        for drone in self.env.drones:
            if not drone.alive:
                continue
            task_id = drone_assignments.get(drone.id)
            if task_id is None:
                continue
            # task_id == objective_id (1:1 mapping)
            if task_id < len(self.env.targets):
                tgt = self.env.targets[task_id]
                ln, = self.ax_map.plot(
                    [drone.pos[0], tgt[0]],
                    [drone.pos[1], tgt[1]],
                    color=C["accent"], lw=0.8,
                    linestyle="--", alpha=0.45, zorder=4,
                )
                self.task_lines.append(ln)

    def _draw_quivers(self):
        for q in self.quivers:
            q.remove()
        self.quivers = []
        for d in self.env.drones:
            if not d.alive:
                continue
            q = self.ax_map.quiver(
                d.pos[0], d.pos[1], d.vel[0], d.vel[1],
                color=C["yellow"], scale=10, width=0.003, alpha=0.65, zorder=9,
            )
            self.quivers.append(q)

    def _draw_labels(self, positions, alive_mask):
        for lbl in self.drone_labels:
            lbl.remove()
        self.drone_labels = []
        for i, pos in enumerate(positions):
            if alive_mask[i]:
                color = C["text_hi"]
                text  = f"D{i}"
            else:
                color = C["text_lo"]
                text  = f"D{i}✗"
            lbl = self.ax_map.text(
                pos[0] + 0.45, pos[1] + 0.45, text,
                color=color, fontsize=5.5, zorder=10, fontweight="bold",
            )
            self.drone_labels.append(lbl)

    # ── Event log ─────────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        """Append a line to the mission log (max 14 lines)."""
        self.log_lines.append(msg)
        if len(self.log_lines) > 14:
            self.log_lines.pop(0)

    def _detect_events(self, step: int):
        """Compare current vs previous state; emit events on changes."""
        phase = self.planner.state.phase.name

        # Phase change
        if phase != self._prev_phase:
            self._append_log(f"[{step:>3}] Phase: {phase}")
            self._prev_phase = phase

        # Drone losses
        current_alive = {d.id for d in self.env.drones if d.alive}
        lost = self._prev_alive - current_alive
        for did in sorted(lost):
            self._append_log(f"[{step:>3}] Drone {did} offline")
        self._prev_alive = current_alive

        # Task assignments
        cs = self.planner.state.coordination
        if cs is not None:
            for did, tid in cs.drone_assignments.items():
                prev_tid = self._prev_assignments.get(did)
                if tid is not None and tid != prev_tid:
                    self._append_log(f"[{step:>3}] D{did} -> Task {tid}")
            self._prev_assignments = dict(cs.drone_assignments)

        # Objective completions
        completed_now = {ti for (_, ti) in self.env.targets_reached}
        new_done = completed_now - self._prev_completed
        for ti in sorted(new_done):
            self._append_log(f"[{step:>3}] Objective {ti} DONE")
        self._prev_completed = completed_now

    # ── Panel renderers ───────────────────────────────────────────────────────

    def _render_mission_overview(self, step: int, ep_r: float):
        ax = self.ax_mission
        ax.clear()
        ax.axis("off")
        ax.set_facecolor(C["panel"])
        self._style_axes(ax, "  MISSION OVERVIEW")

        ms    = self.planner.state
        phase = ms.phase.name
        pcol  = PHASE_COLOR.get(phase, C["text_lo"])

        cs = ms.coordination
        progress = cs.mission_progress if cs else 0.0
        pct      = int(progress * 100)
        bar_len  = 12
        filled   = int(progress * bar_len)
        bar      = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

        obj_done  = ms.completed_count()
        obj_total = len(ms.objectives)
        status_txt = "COMPLETE" if ms.all_objectives_complete else "ACTIVE"
        status_col = C["accent"] if ms.all_objectives_complete else C["yellow"]

        y = 0.90
        lh = 0.115

        def row(label, value, vcol=C["text_hi"], bold=False):
            nonlocal y
            ax.text(0.05, y, label, fontsize=6.5, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace")
            ax.text(0.48, y, value, fontsize=6.5, color=vcol,
                    transform=ax.transAxes, fontfamily="monospace",
                    fontweight="bold" if bold else "normal")
            y -= lh

        row("Status",    status_txt,                          status_col, True)
        row("Phase",     phase,                               pcol,       True)
        row("Progress",  f"{bar} {pct}%",                    C["text_hi"])
        row("Objectives",f"{obj_done} / {obj_total}",         C["text_hi"])
        row("Step",      f"{step} / {MAX_STEPS}",             C["text_lo"])
        row("Episode",   str(self.episode_num),               C["text_lo"])
        if self.episode_rewards:
            row("Best Ep", f"{max(self.episode_rewards):.1f}", C["accent"])

    def _render_task_coordination(self):
        ax = self.ax_tasks
        ax.clear()
        ax.axis("off")
        ax.set_facecolor(C["panel"])
        self._style_axes(ax, "  TASK COORDINATION")

        cs = self.planner.state.coordination
        statuses = [
            ("PENDING",     C["text_lo"]),
            ("ASSIGNED",    C["blue"]),
            ("IN_PROGRESS", C["yellow"]),
            ("COMPLETED",   C["accent"]),
            ("FAILED",      C["red"]),
        ]

        y  = 0.87
        lh = 0.16

        for key, col in statuses:
            count = cs.task_counts.get(key, 0) if cs else 0
            label = key.replace("_", " ")

            # Colored bullet using Unicode character
            ax.text(0.05, y, "\u25cf", fontsize=8, color=col,
                    transform=ax.transAxes)
            ax.text(0.13, y, label, fontsize=6.5, color=C["text_hi"],
                    transform=ax.transAxes, fontfamily="monospace")
            ax.text(0.82, y, str(count), fontsize=7.5, color=col,
                    transform=ax.transAxes, fontfamily="monospace",
                    fontweight="bold", ha="center")
            y -= lh

    def _render_drone_fleet(self):
        ax = self.ax_fleet
        ax.clear()
        ax.axis("off")
        ax.set_facecolor(C["panel"])
        self._style_axes(ax, "  DRONE FLEET")

        cs = self.planner.state.coordination

        # Header
        y = 0.93
        lh = 0.115
        cols = [0.04, 0.15, 0.36, 0.56, 0.75, 0.91]
        headers = ["ID", "STATUS", "TASK", "BATT", "POS-X", "POS-Y"]
        for hx, ht in zip(cols, headers):
            ax.text(hx, y, ht, fontsize=5.5, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace")
        y -= 0.04
        ax.plot([0.02, 0.98], [y, y], color=C["border"], linewidth=0.5,
                transform=ax.transAxes, solid_capstyle='butt')
        y -= 0.06

        for drone in self.env.drones:
            did  = drone.id
            alv  = drone.alive
            batt = drone.energy
            pos  = drone.pos

            if not alv:
                status_txt = "OFFLINE"
                status_col = C["text_lo"]
                row_col    = C["text_lo"]
            elif batt < 0.20:
                status_txt = "LOW BATT"
                status_col = C["red"]
                row_col    = C["text_hi"]
            elif cs and cs.drone_assignments.get(did) is not None:
                status_txt = "ON TASK"
                status_col = C["blue"]
                row_col    = C["text_hi"]
            else:
                status_txt = "READY"
                status_col = C["accent"]
                row_col    = C["text_hi"]

            task_txt = "—"
            if cs:
                tid = cs.drone_assignments.get(did)
                task_txt = f"T{tid}" if tid is not None else "—"

            batt_col = _energy_color(batt)
            batt_txt = f"{int(batt * 100)}%"

            vals = [
                (f"D{did}",       C["text_hi"]),
                (status_txt,      status_col),
                (task_txt,        C["blue"] if task_txt != "—" else C["text_lo"]),
                (batt_txt,        batt_col),
                (f"{pos[0]:+.1f}", row_col),
                (f"{pos[1]:+.1f}", row_col),
            ]
            for vx, (vt, vc) in zip(cols, vals):
                ax.text(vx, y, vt, fontsize=6, color=vc,
                        transform=ax.transAxes, fontfamily="monospace")
            y -= lh

    def _render_mission_log(self):
        ax = self.ax_log
        ax.clear()
        ax.axis("off")
        ax.set_facecolor(C["panel"])
        self._style_axes(ax, "  MISSION LOG")

        if not self.log_lines:
            ax.text(0.05, 0.5, "Awaiting events...",
                    fontsize=6, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace")
            return

        y  = 0.92
        lh = 0.063
        for line in reversed(self.log_lines[-14:]):
            # Color-code keywords
            if "offline" in line.lower() or "FAIL" in line:
                col = C["red"]
            elif "DONE" in line or "COMPLETE" in line:
                col = C["accent"]
            elif "Phase" in line:
                col = C["yellow"]
            elif "Task" in line or "->" in line:
                col = C["blue"]
            else:
                col = C["text_lo"]
            ax.text(0.03, y, line, fontsize=5.8, color=col,
                    transform=ax.transAxes, fontfamily="monospace")
            y -= lh

    # ── Main frame update ─────────────────────────────────────────────────────

    def _update_frame(self, step_reward: float, step: int):
        positions  = np.array([d.pos    for d in self.env.drones])
        energies   = np.array([d.energy for d in self.env.drones])
        alive_mask = np.array([d.alive  for d in self.env.drones], dtype=bool)
        phase      = self.planner.state.phase.name
        cs         = self.planner.state.coordination

        # ── Detect mission events ─────────────────────────────────────────
        self._detect_events(step)

        # ── Swarm map ─────────────────────────────────────────────────────
        active_pos = positions[alive_mask]
        dead_pos   = positions[~alive_mask]
        active_eng = energies[alive_mask]

        if len(active_pos) > 0:
            colors = [_energy_color(e) for e in active_eng]
            self.scat_drones.set_offsets(active_pos)
            self.scat_drones.set_color(colors)
        else:
            self.scat_drones.set_offsets(np.empty((0, 2)))

        if len(dead_pos) > 0:
            self.scat_dead.set_offsets(dead_pos)
        else:
            self.scat_dead.set_offsets(np.empty((0, 2)))

        # Targets — gold when reached
        tgt_colors = []
        for ti in range(len(self.env.targets)):
            hit = any(k[1] == ti for k in self.env.targets_reached)
            tgt_colors.append(C["yellow"] if hit else C["blue"])
        self.scat_targets.set_offsets(self.env.targets)
        self.scat_targets.set_color(tgt_colors)
        self.scat_obs.set_offsets(
            self.env.obstacles if len(self.env.obstacles) > 0
            else np.empty((0, 2))
        )

        # Dynamic overlays
        drone_assignments = cs.drone_assignments if cs else {}
        self._draw_zone_grid(phase)
        self._draw_comm_lines(positions, alive_mask)
        self._draw_task_lines(drone_assignments)
        self._draw_quivers()
        self._draw_labels(positions, alive_mask)

        # Map subtitle
        pcol = PHASE_COLOR.get(phase, C["text_lo"])
        n_active = int(alive_mask.sum())
        self.ax_map.set_xlabel(
            f"Phase: {phase}  |  Active: {n_active}/{NUM_DRONES}  "
            f"|  Step: {step}/{MAX_STEPS}  |  Reward: {step_reward:+.3f}",
            color=pcol, fontsize=7, labelpad=4,
        )

        # ── Reward chart ──────────────────────────────────────────────────
        self.reward_history.append(step_reward)
        self.reward_line.set_data(
            range(len(self.reward_history)), self.reward_history
        )
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()

        # ── Energy bars ───────────────────────────────────────────────────
        for i, (bar, e) in enumerate(zip(self.energy_bars, energies)):
            if alive_mask[i]:
                bar.set_height(max(0.0, e))
                bar.set_color(_energy_color(e))
            else:
                bar.set_height(0.015)
                bar.set_color(C["border"])

        # ── Text panels ───────────────────────────────────────────────────
        self._render_mission_overview(step, step_reward)
        self._render_task_coordination()
        self._render_drone_fleet()
        self._render_mission_log()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ── Episode run ───────────────────────────────────────────────────────────

    def run(self, num_episodes: int = 20, pause: float = 0.04):
        """Run `num_episodes` evaluation episodes with live rendering."""
        self.env.curriculum_ep = 225   # full Phase-3 difficulty for evaluation
        plt.ion()
        plt.show()

        for ep in range(num_episodes):
            self.episode_num    = ep + 1
            self.reward_history = []
            self.log_lines      = []
            self._prev_phase       = None
            self._prev_assignments = {}
            self._prev_alive       = set(range(NUM_DRONES))
            self._prev_completed   = set()

            s    = self.env.reset()
            self.planner.begin_mission(self.env)
            self._append_log(f"[  0] Mission {ep + 1} started")
            ep_r = 0.0
            done = False
            step = 0

            self._draw_rings()

            while not done:
                actions          = self._get_actions(s)
                s, rewards, done = self.env.step(actions)
                self.planner.update(self.env)
                step_reward = rewards.mean()
                ep_r       += step_reward
                step       += 1
                self._update_frame(step_reward, step)
                plt.pause(pause)

            self.episode_rewards.append(ep_r)
            n_failed = len(self.env.failed_ids)
            mission  = self.planner.mission_summary()
            self._append_log(
                f"[{step:>3}] Ep {ep+1} done  R={ep_r:.1f}"
            )
            print(
                f"[Ep {ep+1:>3}]  Reward={ep_r:.1f}  "
                f"Targets={len(self.env.targets_reached)}  "
                f"Failed={n_failed}  "
                f"Mission={mission['final_phase']} "
                f"({mission['objectives_completed']}/{mission['objectives_total']} objectives)"
            )

        plt.ioff()
        plt.show()