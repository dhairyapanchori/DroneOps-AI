"""
DroneOps AI — Training Dashboard
================================

A dedicated dashboard for monitoring the AI model training pipeline.
Runs concurrently with the training loop, pulling metrics in real-time.
"""

import os
import time
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle

import utils.config as cfg
from training.trainer import Trainer

# ── Color palette ─────────────────────────────────────────────────────────────

C = {
    "bg"         : "#0a0e1a",
    "panel"      : "#0d1117",
    "border"     : "#1e2938",
    "accent"     : "#00d4aa",   # green
    "blue"       : "#4a9eff",
    "purple"     : "#b366ff",
    "yellow"     : "#f5c518",
    "red"        : "#ff4560",
    "text_hi"    : "#e6edf3",
    "text_lo"    : "#8b949e",
}

class TrainingDashboard:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.start_time = time.time()
        
        # Historical arrays for plotting
        self.plot_critic_loss = []
        self.plot_actor_loss = []
        self.plot_alpha = []
        self.plot_alpha_loss = []
        self.plot_rewards = []
        self.plot_episodes = []

        # ── Figure & layout ───────────────────────────────────────────────
        self.fig = plt.figure(figsize=(20, 11), facecolor=C["bg"])
        self.fig.canvas.manager.set_window_title("DroneOps AI — Training Dashboard")

        # Layout grid: 4 columns, 6 rows
        gs = gridspec.GridSpec(
            6, 4, figure=self.fig,
            left=0.02, right=0.98, top=0.96, bottom=0.03,
            wspace=0.25, hspace=0.35,
            height_ratios=[0.5, 2.5, 1, 1.5, 0.2, 1]
        )

        # Panels
        self.ax_top      = self.fig.add_subplot(gs[0, :])          # Top Bar
        self.ax_rew_ov   = self.fig.add_subplot(gs[1, 0])          # Reward Overview
        self.ax_chart    = self.fig.add_subplot(gs[1, 1:3])        # Main Chart
        self.ax_metrics  = self.fig.add_subplot(gs[1, 3])          # Training Metrics
        self.ax_status   = self.fig.add_subplot(gs[2, :])          # Mission Status
        
        self.ax_buffer   = self.fig.add_subplot(gs[3, 0])          # Replay Buffer
        self.ax_ckpt     = self.fig.add_subplot(gs[3, 1:3])        # Checkpoints
        self.ax_hyper    = self.fig.add_subplot(gs[3, 3])          # Hyperparameters
        
        self.ax_log      = self.fig.add_subplot(gs[4:, :])         # Training Log

        self._style_all_axes()
        self._setup_chart()
        self._setup_metrics_charts()

        plt.suptitle("DRONEOPS AI  ·  TRAINING DASHBOARD", 
                     color=C["text_hi"], fontsize=10, fontweight="bold", x=0.02, ha="left", y=0.98)

    def _style_axes(self, ax, title=""):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text_lo"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
            sp.set_linewidth(1.0)
        if title:
            ax.set_title(title, color=C["text_hi"], fontsize=8, fontweight="bold", pad=8, loc="left")

    def _style_all_axes(self):
        for ax, title in [
            (self.ax_top,    ""),
            (self.ax_rew_ov, "REWARD OVERVIEW"),
            (self.ax_chart,  "REWARD VS EPISODE"),
            (self.ax_metrics,"TRAINING METRICS"),
            (self.ax_status, "MISSION & SWARM STATUS"),
            (self.ax_buffer, "REPLAY BUFFER"),
            (self.ax_ckpt,   "MODEL CHECKPOINTS"),
            (self.ax_hyper,  "HYPERPARAMETERS"),
            (self.ax_log,    "TRAINING LOG"),
        ]:
            self._style_axes(ax, title)

        for ax in [self.ax_top, self.ax_rew_ov, self.ax_status, 
                   self.ax_buffer, self.ax_ckpt, self.ax_hyper, self.ax_log]:
            ax.axis("off")

    def _setup_chart(self):
        self.ax_chart.grid(True, color=C["border"], linestyle="-", linewidth=0.5, alpha=0.5)
        self.line_reward, = self.ax_chart.plot([], [], color=C["blue"], lw=1.2, label="Episode Reward")
        self.line_avg,    = self.ax_chart.plot([], [], color=C["purple"], lw=1.8, label="Average Reward (50)")
        self.line_best,   = self.ax_chart.plot([], [], color=C["accent"], lw=1.2, linestyle="--", label="Best Reward")
        self.ax_chart.legend(loc="upper left", frameon=False, labelcolor=C["text_lo"], fontsize=7, ncol=3)

    def _setup_metrics_charts(self):
        # We'll draw 4 mini charts inside ax_metrics using twinx or inset_axes.
        # But for simplicity and performance, we'll draw them manually using lines mapped to 0..1 bounding boxes.
        self.ax_metrics.axis("off")
        pass

    def _render_top_bar(self):
        ax = self.ax_top
        ax.clear()
        ax.axis("off")
        
        ep = self.trainer.current_ep
        max_ep = cfg.MAX_EPISODES
        prog = ep / max(1, max_ep)
        steps = self.trainer.total_steps
        max_steps = max_ep * cfg.MAX_STEPS # rough max
        
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(1, ep)) * (max_ep - ep) if ep > 0 else 0
        
        el_str = str(datetime.timedelta(seconds=int(elapsed)))
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        
        # Layout columns
        cols = [0.02, 0.25, 0.45, 0.65, 0.85]
        
        # Progress Bar
        ax.text(cols[0], 0.6, "Training Progress", color=C["text_lo"], fontsize=7)
        ax.add_patch(Rectangle((cols[0], 0.2), 0.15, 0.3, color=C["border"]))
        ax.add_patch(Rectangle((cols[0], 0.2), 0.15 * prog, 0.3, color=C["purple"]))
        ax.text(cols[0] + 0.16, 0.2, f"{int(prog*100)}%", color=C["text_hi"], fontsize=8, fontweight="bold")
        
        # Episode
        ax.text(cols[1], 0.6, "Episode", color=C["text_lo"], fontsize=7)
        ax.text(cols[1], 0.2, f"{ep}", color=C["text_hi"], fontsize=9, fontweight="bold")
        ax.text(cols[1] + 0.03, 0.2, f"/ {max_ep}", color=C["text_lo"], fontsize=7)
        
        # Steps
        ax.text(cols[2], 0.6, "Steps", color=C["text_lo"], fontsize=7)
        ax.text(cols[2], 0.2, f"{steps:,}", color=C["text_hi"], fontsize=9, fontweight="bold")
        
        # Elapsed & ETA
        ax.text(cols[3], 0.6, "Elapsed Time", color=C["text_lo"], fontsize=7)
        ax.text(cols[3], 0.2, el_str, color=C["text_hi"], fontsize=9, fontweight="bold")
        
        ax.text(cols[4], 0.6, "ETA", color=C["text_lo"], fontsize=7)
        ax.text(cols[4], 0.2, eta_str, color=C["text_hi"], fontsize=9, fontweight="bold")
        
        # Status
        status = "Running" if self.trainer.is_training else "Finished"
        color = C["accent"] if self.trainer.is_training else C["text_lo"]
        ax.add_patch(Rectangle((0.92, 0.1), 0.07, 0.7, color=C["bg"], ec=C["border"], lw=1, rx=0.1, ry=0.1))
        ax.text(0.93, 0.55, "Training Status", color=C["text_lo"], fontsize=6)
        ax.add_patch(Circle((0.935, 0.25), 0.05, color=color, transform=ax.transData))
        ax.text(0.945, 0.25, status, color=color, fontsize=7)

    def _render_reward_overview(self):
        ax = self.ax_rew_ov
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "REWARD OVERVIEW")
        
        if not self.plot_rewards:
            return
            
        cur_r = self.plot_rewards[-1]
        avg_r = np.mean(self.plot_rewards[-50:])
        best_r = self.trainer.best_ep_reward
        
        def draw_metric(y, title, val, color):
            ax.text(0.05, y, title, color=C["text_lo"], fontsize=7)
            ax.text(0.05, y - 0.15, f"{val:+.1f}", color=color, fontsize=18, fontweight="bold")
            
        draw_metric(0.75, "Episode Reward", cur_r, C["accent"] if cur_r > 0 else C["red"])
        draw_metric(0.45, "Average Reward (50)", avg_r, C["blue"])
        draw_metric(0.15, "Best Reward", best_r if best_r != -float('inf') else 0.0, C["purple"])

    def _render_chart(self):
        if not self.plot_rewards:
            return
            
        self.line_reward.set_data(self.plot_episodes, self.plot_rewards)
        
        avg50 = [np.mean(self.plot_rewards[max(0, i-50):i+1]) for i in range(len(self.plot_rewards))]
        self.line_avg.set_data(self.plot_episodes, avg50)
        
        best = self.trainer.best_ep_reward
        if best != -float('inf'):
            self.line_best.set_data([0, max(self.plot_episodes)], [best, best])
            
        self.ax_chart.relim()
        self.ax_chart.autoscale_view()

    def _render_training_metrics(self):
        ax = self.ax_metrics
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "TRAINING METRICS")
        
        metrics = [
            ("Critic Loss", self.trainer.last_c_loss, self.plot_critic_loss, C["blue"]),
            ("Actor Loss", self.trainer.last_a_loss, self.plot_actor_loss, C["purple"]),
            ("Alpha", self.trainer.last_alpha, self.plot_alpha, C["accent"]),
            ("Alpha Loss", self.trainer.last_alpha_loss, self.plot_alpha_loss, C["yellow"])
        ]
        
        y = 0.8
        dh = 0.22
        
        for name, val, hist, col in metrics:
            ax.text(0.05, y, name, color=C["text_lo"], fontsize=7)
            ax.text(0.05, y - 0.1, f"{val:.4g}", color=C["text_hi"], fontsize=10, fontweight="bold")
            
            if hist:
                # mini sparkline
                hx = np.linspace(0.5, 0.95, len(hist))
                hy = np.array(hist)
                # normalize hy to fit in bounding box
                if hy.max() != hy.min():
                    hy = (hy - hy.min()) / (hy.max() - hy.min())
                else:
                    hy = np.full_like(hy, 0.5)
                hy = (hy * 0.12) + (y - 0.12)
                ax.plot(hx, hy, color=col, lw=1)
                
            y -= dh
            if y > 0:
                ax.plot([0.05, 0.95], [y + 0.1, y + 0.1], color=C["border"], lw=0.5, solid_capstyle="butt")

    def _render_status(self):
        ax = self.ax_status
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "MISSION & SWARM STATUS")
        
        ms = self.trainer.planner.mission_summary() if hasattr(self.trainer.planner, 'mission_summary') else {}
        phase = ms.get("final_phase", "UNKNOWN")
        targets = len(self.trainer.env.targets_reached)
        total_targets = len(self.trainer.env.targets)
        failed = len(self.trainer.env.failed_ids)
        coord = len(set(t for _, t in self.trainer.env.targets_reached)) / max(1, total_targets)
        
        # Calculate avg mission success rate
        recent_rews = self.plot_rewards[-50:]
        success_rate = sum(1 for r in recent_rews if r > 10) / max(1, len(recent_rews)) * 100
        
        alive = sum(1 for d in self.trainer.env.drones if d.alive)
        avg_en = np.mean([d.energy for d in self.trainer.env.drones])
        
        cards = [
            ("Mission Phase", phase, C["blue"] if phase == "SEARCH" else C["accent"], ""),
            ("Targets Reached", f"{targets} / {total_targets}", C["accent"], ""),
            ("Failed Objectives", f"{failed}", C["red"] if failed > 0 else C["text_hi"], ""),
            ("Coordination Score", f"{coord:.2f}", C["accent"], "Excellent" if coord > 0.8 else "Fair"),
            ("Mission Success Rate", f"{success_rate:.1f}%", C["yellow"], "Last 50 Episodes"),
            ("Alive Drones", f"{alive} / 6", C["text_hi"] if alive == 6 else C["red"], ""),
            ("Average Energy", f"{int(avg_en*100)}%", C["accent"] if avg_en > 0.5 else C["yellow"], "Healthy" if avg_en > 0.5 else "Low"),
        ]
        
        cx = 0.02
        cw = 0.13
        
        for (title, val, col, sub) in cards:
            ax.add_patch(Rectangle((cx, 0.1), cw, 0.65, color=C["bg"], ec=C["border"], lw=1))
            ax.text(cx + 0.01, 0.6, title, color=C["text_lo"], fontsize=7)
            ax.text(cx + 0.01, 0.35, val, color=col, fontsize=12, fontweight="bold")
            if sub:
                ax.text(cx + 0.01, 0.18, sub, color=C["accent"], fontsize=6)
            cx += cw + 0.01

    def _render_buffer(self):
        ax = self.ax_buffer
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "REPLAY BUFFER")
        
        buf_size = len(self.trainer.buf)
        cap = cfg.BUFFER_SIZE
        pct = buf_size / cap
        
        ax.text(0.4, 0.75, "Size", color=C["text_lo"], fontsize=7)
        ax.text(0.9, 0.75, f"{buf_size:,}", color=C["text_hi"], fontsize=8, ha="right", fontweight="bold")
        
        ax.text(0.4, 0.55, "Capacity", color=C["text_lo"], fontsize=7)
        ax.text(0.9, 0.55, f"{cap:,}", color=C["text_hi"], fontsize=8, ha="right", fontweight="bold")
        
        avg_len = self.trainer.total_steps / max(1, self.trainer.current_ep)
        ax.text(0.4, 0.35, "Avg. Episode Len", color=C["text_lo"], fontsize=7)
        ax.text(0.9, 0.35, f"{int(avg_len)}", color=C["text_hi"], fontsize=8, ha="right", fontweight="bold")
        
        tps = self.trainer.total_steps / max(1, time.time() - self.start_time)
        ax.text(0.4, 0.15, "Transitions / Sec", color=C["text_lo"], fontsize=7)
        ax.text(0.9, 0.15, f"{int(tps)}", color=C["text_hi"], fontsize=8, ha="right", fontweight="bold")
        
        # Draw donut
        center = (0.2, 0.45)
        ax.add_patch(Circle(center, 0.18, color=C["border"], fill=False, lw=4))
        import matplotlib.patches as mpatches
        wedge = mpatches.Wedge(center, 0.18, 90 - 360*pct, 90, width=0.04, color=C["accent"])
        ax.add_patch(wedge)
        ax.text(center[0], center[1], f"{int(pct*100)}%", color=C["text_hi"], fontsize=10, ha="center", va="center", fontweight="bold")

    def _render_checkpoints(self):
        ax = self.ax_ckpt
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "MODEL CHECKPOINTS")
        
        models = [
            ("Actor", "actor_trained.pth"),
            ("Critic", "critic_trained.pth"),
            ("GNN", "gnn_trained.pth"),
            ("Transformer", "trans_trained.pth"),
            ("Meta Adapter", "meta_trained.pth")
        ]
        
        cx = 0.02
        cw = 0.18
        
        for name, fname in models:
            ax.add_patch(Rectangle((cx, 0.1), cw, 0.7, color=C["bg"], ec=C["border"], lw=1))
            ax.text(cx + 0.02, 0.65, name, color=C["text_hi"], fontsize=8, fontweight="bold")
            
            if os.path.exists(fname):
                mtime = os.path.getmtime(fname)
                age = time.time() - mtime
                if age < 60:
                    age_str = "Just now"
                elif age < 3600:
                    age_str = f"{int(age/60)} min ago"
                else:
                    age_str = f"{int(age/3600)} hrs ago"
                
                ax.text(cx + 0.02, 0.4, fname, color=C["text_lo"], fontsize=7)
                ax.text(cx + 0.02, 0.2, f"✓ {age_str}", color=C["accent"], fontsize=7)
            else:
                ax.text(cx + 0.02, 0.4, fname, color=C["text_lo"], fontsize=7)
                ax.text(cx + 0.02, 0.2, "Pending", color=C["text_lo"], fontsize=7)
                
            cx += cw + 0.015

    def _render_hyperparams(self):
        ax = self.ax_hyper
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "HYPERPARAMETERS")
        
        params = [
            ("Learning Rate (Actor)", f"{cfg.LR_ACTOR}"),
            ("Learning Rate (Critic)", f"{cfg.LR_CRITIC}"),
            ("Alpha (Entropy Coef.)", f"{self.trainer.last_alpha:.3g}"),
            ("Discount Factor (γ)", f"{cfg.GAMMA}"),
            ("Batch Size", f"{cfg.BATCH}"),
            ("Update Every", f"{cfg.UPDATE_EVERY} Steps"),
            ("Updates Per Step", f"{cfg.UPDATES_PER_STEP}"),
        ]
        
        y = 0.8
        for k, v in params:
            ax.text(0.05, y, k, color=C["text_lo"], fontsize=7)
            ax.text(0.95, y, v, color=C["text_hi"], fontsize=7, ha="right", fontweight="bold")
            y -= 0.12

    def _render_log(self):
        ax = self.ax_log
        ax.clear()
        ax.axis("off")
        
        # Don't overwrite the title, just draw text below it
        y = 0.95
        
        if not self.trainer.log_events:
            ax.text(0.02, y, "Waiting for episode events...", color=C["text_lo"], fontsize=7, fontfamily="monospace")
            return
            
        for line in reversed(self.trainer.log_events[-12:]):
            color = C["text_lo"]
            if "Saved" in line:
                color = C["accent"]
            elif "Reward +" in line:
                color = C["yellow"]
            elif "Reward -" in line:
                color = C["red"]
                
            ax.text(0.02, y, line, color=color, fontsize=7, fontfamily="monospace")
            y -= 0.08

    def update(self):
        # Read from trainer
        ep = self.trainer.current_ep
        
        # Append latest metrics if they changed (rough tracking via episode count length)
        if ep > len(self.plot_episodes) and hasattr(self.trainer.metrics, 'rewards') and len(self.trainer.metrics.rewards) > 0:
            self.plot_episodes.append(ep)
            self.plot_rewards.append(self.trainer.metrics.rewards[-1])
            self.plot_critic_loss.append(self.trainer.last_c_loss)
            self.plot_actor_loss.append(self.trainer.last_a_loss)
            self.plot_alpha.append(self.trainer.last_alpha)
            self.plot_alpha_loss.append(self.trainer.last_alpha_loss)

        self._render_top_bar()
        self._render_reward_overview()
        self._render_chart()
        self._render_training_metrics()
        self._render_status()
        self._render_buffer()
        self._render_checkpoints()
        self._render_hyperparams()
        self._render_log()
        
        self.fig.canvas.draw_idle()
        plt.pause(0.2) # Yield to GUI event loop

    def run(self):
        plt.ion()
        plt.show()
        
        try:
            while plt.fignum_exists(self.fig.number):
                self.update()
                if not self.trainer.is_training:
                    # Update one last time and stop
                    self.update()
                    print("Training completed. Dashboard will remain open until closed.")
                    plt.ioff()
                    plt.show()
                    break
        except KeyboardInterrupt:
            pass
