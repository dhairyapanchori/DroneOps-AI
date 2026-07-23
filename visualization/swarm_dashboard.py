"""
DroneOps AI - Mission Control Center Dashboard
================================================

Enterprise-grade real-time mission dashboard. Redesigned from scratch as a
professional Mission Control Center to clearly demonstrate the full system
architecture during technical reviews and demonstrations.

Layout (6x6 GridSpec)
--------------------------
  TOP BAR      (Row 0, Cols 0-5)   : 5 discrete cards (Mission Status, Phase, Objectives, Active Drones, Avg Battery)
  MIDDLE LEFT  (Rows 1-4, Cols 0-3): Large Live Swarm Map
  MIDDLE RIGHT (Rows 1-2, Cols 4-5): Mission Summary
  MIDDLE RIGHT (Row 3, Cols 4-5)   : Drone Fleet
  MIDDLE RIGHT (Row 4, Cols 4-5)   : Task Status / Task List
  BOTTOM LEFT  (Row 5, Cols 0-3)   : Battery Levels (bar chart)
  BOTTOM RIGHT (Row 5, Cols 4-5)   : Live Mission Log

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
from matplotlib.patches import Circle
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


# === Typography ===
FS_KPI_LBL = 8
FS_KPI_VAL = 14
FS_HDR     = 9.5
FS_TBL     = 9.5
FS_LOG     = 9.5

# === Color palette ===

C = {
    "bg"         : "#0a0e1a",
    "panel"      : "#0d1117",
    "border"     : "#1e2938",
    "accent"     : "#00d4aa",   # green - healthy / success
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


# === Dashboard ===

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
        # 1. Models & Env
        self.env     = SwarmEnv()
        self.planner = MissionPlanner()

        def load_ckpt(path):
            p = os.path.join(os.path.dirname(__file__), "..", "checkpoints", path)
            return torch.load(p, map_location="cpu", weights_only=True) if os.path.exists(p) else None

        self.actor = Actor(FUSED_DIM, ACTION_DIM)
        if ckpt := load_ckpt(actor_path): self.actor.load_state_dict(ckpt)
        self.actor.eval()

        self.gnn = SwarmGNN(STATE_DIM)
        if ckpt := load_ckpt(gnn_path): self.gnn.load_state_dict(ckpt)
        self.gnn.eval()

        self.trans = MissionTransformer(STATE_DIM)
        if ckpt := load_ckpt(trans_path): self.trans.load_state_dict(ckpt)
        self.trans.eval()

        self.meta = MetaAdapter(STATE_DIM)
        if ckpt := load_ckpt(meta_path): self.meta.load_state_dict(ckpt)
        self.meta.eval()

        # Data tracking
        self.reward_history  : list[float] = []
        self.episode_rewards : list[float] = []
        self.episode_num     : int         = 0
        self.log_lines       : list[str]   = []   # mission event log

        # Previous-state trackers for event detection
        self._prev_phase       = None
        self._prev_assignments : dict = {}
        self._prev_alive       : set  = set(range(NUM_DRONES))
        self._prev_completed   : set  = set()

        # Figure & layout
        self.fig = plt.figure(
            figsize=(20, 11),
            facecolor=C["bg"],
        )
        self.fig.canvas.manager.set_window_title(
            "DroneOps AI - Mission Control Center"
        )

        gs = gridspec.GridSpec(
            6, 6,
            figure=self.fig,
            left=0.03, right=0.97,
            top=0.92, bottom=0.04,
            wspace=0.4, hspace=0.5,
        )

        # Top Bar (Rows 0, Cols 0-5): 5 discrete cards
        top_gs = gs[0, 0:6].subgridspec(1, 5, wspace=0.3)
        self.ax_top = [self.fig.add_subplot(top_gs[0, i]) for i in range(5)]

        # Middle Left: Map (Rows 1-4, Cols 0-3)
        self.ax_map = self.fig.add_subplot(gs[1:5, 0:4])

        # Middle Right Panels
        self.ax_mission = self.fig.add_subplot(gs[1:3, 4:6])  # Mission Summary
        self.ax_fleet   = self.fig.add_subplot(gs[3, 4:6])    # Drone Fleet
        self.ax_tasks   = self.fig.add_subplot(gs[4, 4:6])    # Task Status / Task List

        # Bottom Panels
        self.ax_energy  = self.fig.add_subplot(gs[5, 0:4])    # Battery Levels
        self.ax_log     = self.fig.add_subplot(gs[5, 4:6])    # Live Mission Log

        self._style_all_axes()
        self._setup_map()
        self._setup_energy()

        plt.suptitle(
            "DroneOps AI - Mission Control Center - SAC + GNN + Transformer + Meta-Adapter + Evolution",
            color=C["accent"], fontsize=9, fontweight="bold",
            y=0.98,
        )

    # === Styling helpers ===

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
                fontsize=FS_HDR, fontweight="bold",
                pad=8, loc="left"
            )

    def _style_all_axes(self):
        # Top cards
        for ax in self.ax_top:
            self._style_axes(ax, "")

        # Main map
        self._style_axes(self.ax_map, "")

        # Right/Bottom panels
        for ax, title in [
            (self.ax_mission, "  MISSION SUMMARY"),
            (self.ax_fleet,   "  DRONE FLEET"),
            (self.ax_tasks,   "  TASK STATUS"),
            (self.ax_log,     "  LIVE MISSION LOG"),
            (self.ax_energy,  "  BATTERY LEVELS"),
        ]:
            self._style_axes(ax, title)

        # Text-only panels (hide ticks but keep borders for consistent cards)
        for ax in self.ax_top + [self.ax_mission, self.ax_fleet, self.ax_tasks, self.ax_log]:
            ax.set_xticks([])
            ax.set_yticks([])

    # === Map setup ===

    def _setup_map(self):
        ax = self.ax_map
        ax.set_facecolor(C["bg"])
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
        ax.set_title(
            "  LIVE SWARM MAP", color=C["accent"],
            fontsize=FS_HDR, fontweight="bold", pad=8, loc="left",
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
            fontsize=FS_TBL, facecolor=C["panel"],
            edgecolor=C["border"], labelcolor=C["text_lo"],
            framealpha=0.9,
        )

    # === Energy bar setup ===

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
            color=colors, width=0.6, alpha=0.9
        )
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_edgecolor(C["border"])
        ax.axhline(0, color=C["border"], lw=0.6)

    # === Fusion pipeline (unchanged from original) ===

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

    # === Map drawing helpers ===

    def _draw_rings(self):
        """Draw obstacle and target radius rings - called once per episode."""
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
        """Draw the 2A-2 search zone overlay during SEARCH phase."""
        for ln in self.zone_lines:
            ln.remove()
        self.zone_lines = []

        if phase != "SEARCH":
            return

        # Planner zones: 2A-2 over [-12, 12]
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
                text  = f"D{i}[X]"
            lbl = self.ax_map.text(
                pos[0] + 0.45, pos[1] + 0.45, text,
                color=color, fontsize=5.5, zorder=10, fontweight="bold",
            )
            self.drone_labels.append(lbl)

    # === Event log ===

    def _format_time(self, step: int) -> str:
        """Format step as MM:SS (assuming 1 step = 1 second for demo timeline)."""
        return f"{step // 60:02d}:{step % 60:02d}"

    def _append_log(self, msg: str):
        """Append a line to the mission log (max 14 lines)."""
        self.log_lines.append(msg)
        if len(self.log_lines) > 14:
            self.log_lines.pop(0)

    def _detect_events(self, step: int):
        """Compare current vs previous state; emit events on changes."""
        ms = self.planner.state
        cs = ms.coordination

        # Phase change
        curr_phase = ms.phase.name
        if curr_phase != self._prev_phase:
            if self._prev_phase is not None:
                self._append_log(f"{self._format_time(step)} Phase shift: {curr_phase}")
            self._prev_phase = curr_phase

        # Task assignments
        if cs:
            curr_assignments = cs.drone_assignments
            for did, tid in curr_assignments.items():
                if tid is not None and self._prev_assignments.get(did) != tid:
                    self._append_log(f"{self._format_time(step)} Drone {did} -> Task {tid}")
            self._prev_assignments = dict(curr_assignments)

        # Drone deaths
        curr_alive = {d.id for d in self.env.drones if d.alive}
        died = self._prev_alive - curr_alive
        for did in died:
            self._append_log(f"{self._format_time(step)} Drone {did} offline!")
        self._prev_alive = curr_alive

        # Task completion
        curr_completed = set(k[1] for k in self.env.targets_reached)
        new_comp = curr_completed - self._prev_completed
        for tid in new_comp:
            self._append_log(f"{self._format_time(step)} Task {tid} COMPLETED")
        self._prev_completed = curr_completed

    # === Render functions ===

    def _render_top_cards(self, step: int):
        ms = self.planner.state
        phase = ms.phase.name
        pcol = PHASE_COLOR.get(phase, C["text_lo"])
        status_txt = "COMPLETE" if ms.all_objectives_complete else "ACTIVE"

        alive_count = sum(1 for d in self.env.drones if d.alive)
        avg_batt = np.mean([d.energy for d in self.env.drones]) * 100

        obj_done = ms.completed_count()
        obj_total = len(ms.objectives)

        cards = [
            ("MISSION STATUS", status_txt, C["text_hi"]),
            ("CURRENT PHASE", phase, pcol),
            ("OBJECTIVES", f"{obj_done} / {obj_total}", C["text_hi"]),
            ("ACTIVE DRONES", f"{alive_count} / {NUM_DRONES}", C["text_hi"] if alive_count == NUM_DRONES else C["yellow"]),
            ("AVG BATTERY", f"{avg_batt:.1f}%", _energy_color(avg_batt/100)),
        ]

        for ax, (title, value, color) in zip(self.ax_top, cards):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            self._style_axes(ax, "")
            
            # Title
            ax.text(0.1, 0.60, title, fontsize=FS_KPI_LBL, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace", fontweight="bold")
            # Value
            ax.text(0.1, 0.20, value, fontsize=FS_KPI_VAL, color=color,
                    transform=ax.transAxes, fontfamily="monospace", fontweight="bold")

    def _render_mission_summary(self):
        ax = self.ax_mission
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_axes(ax, "  MISSION SUMMARY")
        
        ms = self.planner.state
        cs = ms.coordination
        progress = cs.mission_progress if cs else 0.0
        pct = int(progress * 100)
        bar_len = 14
        filled = int(progress * bar_len)
        bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

        y = 0.75
        lh = 0.20
        ax.text(0.05, y, f"Progress: {bar} {pct}%", fontsize=FS_TBL, color=C["accent"],
                transform=ax.transAxes, fontfamily="monospace")
        y -= lh
        ax.text(0.05, y, f"Targets Reached: {ms.completed_count()} / {len(ms.objectives)}", fontsize=FS_TBL, color=C["text_hi"],
                transform=ax.transAxes, fontfamily="monospace")
        y -= lh
        
        if cs:
            ax.text(0.05, y, f"Active Coordination: True", fontsize=FS_TBL, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace")
            y -= lh
            ax.text(0.05, y, f"Total Tasks: {sum(cs.task_counts.values())}", fontsize=FS_TBL, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace")

    def _render_task_coordination(self):
        ax = self.ax_tasks
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_axes(ax, "  TASK STATUS")

        cs = self.planner.state.coordination
        statuses = [
            ("PENDING",     C["text_lo"]),
            ("ASSIGNED",    C["blue"]),
            ("IN_PROGRESS", C["yellow"]),
            ("COMPLETED",   C["accent"]),
            ("FAILED",      C["red"]),
        ]

        y  = 0.82
        lh = 0.18

        for key, col in statuses:
            count = cs.task_counts.get(key, 0) if cs else 0
            ax.text(0.08, y, f"{key:11s} : {count}", fontsize=FS_TBL, color=col,
                    transform=ax.transAxes, fontfamily="monospace")
            y -= lh

    def _render_drone_fleet(self):
        ax = self.ax_fleet
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_axes(ax, "  DRONE FLEET")

        cs = self.planner.state.coordination

        # Header
        y = 0.85
        lh = 0.16
        cols = [0.05, 0.22, 0.48, 0.75]
        headers = ["ID", "STATUS", "TASK", "BATT"]
        for hx, ht in zip(cols, headers):
            ax.text(hx, y, ht, fontsize=FS_TBL, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace", fontweight="bold", va="center")
        y -= 0.05
        ax.plot([0.02, 0.98], [y, y], color=C["border"], linewidth=0.5,
                transform=ax.transAxes, solid_capstyle='butt')
        
        y -= 0.10

        for drone in self.env.drones:
            did  = drone.id
            alv  = drone.alive
            batt = drone.energy
            
            if not alv:
                status_txt = "OFFLINE"
                status_col = C["red"]
                task_txt   = "-"
            elif batt < 0.20:
                status_txt = "LOW BATT"
                status_col = C["yellow"]
                task_txt   = "Returning"
            elif cs and cs.drone_assignments.get(did) is not None:
                status_txt = "ON TASK"
                status_col = C["blue"]
                tid = cs.drone_assignments.get(did)
                task_txt = f"Target {tid}"
            else:
                status_txt = "IDLE"
                status_col = C["text_hi"]
                task_txt   = "Standby"

            ax.text(cols[0], y, f"D{did}", fontsize=FS_TBL, color=C["text_hi"],
                    transform=ax.transAxes, fontfamily="monospace", va="center")
            ax.text(cols[1], y, status_txt, fontsize=FS_TBL, color=status_col,
                    transform=ax.transAxes, fontfamily="monospace", va="center")
            ax.text(cols[2], y, task_txt, fontsize=FS_TBL, color=C["text_hi"],
                    transform=ax.transAxes, fontfamily="monospace", va="center")
            
            # Draw tiny battery bar
            bx = cols[3]
            bw = 0.20
            bh = 0.06
            # shift y by bh/2 so the rectangle is centered vertically with the text
            by = y - (bh / 2)
            ax.add_patch(matplotlib.patches.Rectangle((bx, by), bw, bh, transform=ax.transAxes, facecolor=C["border"]))
            ax.add_patch(matplotlib.patches.Rectangle((bx, by), bw * batt, bh, transform=ax.transAxes, facecolor=_energy_color(batt)))
            
            y -= lh

    def _render_mission_log(self):
        ax = self.ax_log
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_axes(ax, "  LIVE MISSION LOG")

        if not self.log_lines:
            ax.text(0.08, 0.5, "Awaiting events...",
                    fontsize=FS_LOG, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace", va="center")
            return

        y = 0.85
        lh = 0.12
        for line in reversed(self.log_lines[-7:]):
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

            parts = line.split(" ", 1)
            if len(parts) == 2:
                time_str, msg = parts
            else:
                time_str, msg = "", line
                
            ax.text(0.08, y, time_str, fontsize=FS_LOG, color=C["text_lo"],
                    transform=ax.transAxes, fontfamily="monospace", va="center")
            ax.text(0.24, y, msg, fontsize=FS_LOG, color=col,
                    transform=ax.transAxes, fontfamily="monospace", va="center")
            y -= lh

    # === Main frame update ===

    def _update_frame(self, step_reward: float, step: int):
        positions  = np.array([d.pos    for d in self.env.drones])
        energies   = np.array([d.energy for d in self.env.drones])
        alive_mask = np.array([d.alive  for d in self.env.drones], dtype=bool)
        phase      = self.planner.state.phase.name
        cs         = self.planner.state.coordination

        # Detect mission events
        self._detect_events(step)

        # Swarm map
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

        # Targets - gold when reached
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
            f"|  Step: {step}/{MAX_STEPS}",
            color=pcol, fontsize=7, labelpad=4,
        )

        # Energy bars
        for i, (bar, e) in enumerate(zip(self.energy_bars, energies)):
            if alive_mask[i]:
                bar.set_height(max(0.0, e))
                bar.set_color(_energy_color(e))
            else:
                bar.set_height(0.015)
                bar.set_color(C["border"])

        # Text panels
        self._render_top_cards(step)
        self._render_mission_summary()
        self._render_task_coordination()
        self._render_drone_fleet()
        self._render_mission_log()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # === Episode run ===

    def run(self, num_episodes: int = 20, pause: float = 0.04):
        """Run `num_episodes` evaluation episodes with live rendering."""
        self.env.curriculum_ep = 225   # full Phase-3 difficulty for evaluation
        plt.ion()
        plt.show(block=False)

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
            self._append_log(f"00:00 Mission {ep + 1} started")
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
                f"{self._format_time(step)} Ep {ep+1} done"
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

