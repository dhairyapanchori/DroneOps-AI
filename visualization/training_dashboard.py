import os
import time
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

import utils.config as cfg
from training.trainer import Trainer

# Color palette
C = {
    "bg"         : "#0a0e1a",
    "panel"      : "#111621",
    "border"     : "#1e2938",
    "accent"     : "#00d4aa",
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
        self.plot_episodes = []
        self.plot_rewards = []
        self.plot_critic_loss = []
        self.plot_actor_loss = []
        self.plot_alpha = []
        self.plot_alpha_loss = []
        
        # KPI sparklines
        self.kpi_hist_avg_reward = []
        self.kpi_hist_best_reward = []
        self.kpi_hist_buffer = []
        self.kpi_hist_speed = []

        self.fig = plt.figure(figsize=(20, 11), facecolor=C["bg"])
        self.fig.canvas.manager.set_window_title("DroneOps AI - Enterprise Training Dashboard")
        
        # 12x12 GridSpec Architecture
        gs = gridspec.GridSpec(
            12, 12, figure=self.fig,
            left=0.02, right=0.98, top=0.95, bottom=0.03,
            wspace=0.3, hspace=0.8
        )
        
        # Section 1 (Row 1): 4 KPI cards
        self.ax_kpis = self.fig.add_subplot(gs[0, 0:9])
        
        # Section 2 & 4 (Rows 2-7): Centerpiece
        self.ax_reward = self.fig.add_subplot(gs[1:5, 0:6])
        self.ax_loss   = self.fig.add_subplot(gs[5:7, 0:3])
        self.ax_alpha  = self.fig.add_subplot(gs[5:7, 3:6])
        self.ax_map    = self.fig.add_subplot(gs[1:7, 6:9])
        
        # Section 5 & 6 (Rows 8-9): Mission Analytics & Checkpoints
        self.ax_mission = self.fig.add_subplot(gs[7:9, 0:4])
        self.ax_ckpt    = self.fig.add_subplot(gs[7:9, 4:9])
        
        # Section 8 (Rows 10-11): Scrolling Live Training Log
        self.ax_log = self.fig.add_subplot(gs[9:12, 0:9])
        
        # Section 7 & 9 (Right Sidebar Cols 9-11)
        self.ax_hyper   = self.fig.add_subplot(gs[0:5, 9:12])
        self.ax_metrics = self.fig.add_subplot(gs[5:12, 9:12])
        
        self.last_render_time = 0
        
        plt.suptitle("DRONEOPS AI  ·  ENTERPRISE TRAINING DASHBOARD", 
                     color=C["text_hi"], fontsize=12, fontweight="bold", x=0.02, ha="left", y=0.98)

    def _style_axes(self, ax, title=""):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text_lo"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
            sp.set_linewidth(1.0)
        if title:
            ax.set_title(title, color=C["text_hi"], fontsize=9, fontweight="bold", pad=10, loc="left")

    def _render_kpis(self):
        ax = self.ax_kpis
        ax.clear()
        ax.axis("off")
        
        avg_r = self.kpi_hist_avg_reward[-1] if self.kpi_hist_avg_reward else 0.0
        best_r = self.kpi_hist_best_reward[-1] if self.kpi_hist_best_reward else 0.0
        buf_sz = self.kpi_hist_buffer[-1] if self.kpi_hist_buffer else 0
        speed = self.kpi_hist_speed[-1] if self.kpi_hist_speed else 0.0
        
        cards = [
            ("Average Reward", f"{avg_r:+.1f}", self.kpi_hist_avg_reward, C["blue"]),
            ("Best Reward", f"{best_r:+.1f}", self.kpi_hist_best_reward, C["accent"]),
            ("Replay Buffer", f"{int(buf_sz):,}", self.kpi_hist_buffer, C["purple"]),
            ("Training Speed", f"{int(speed)} TPS", self.kpi_hist_speed, C["yellow"])
        ]
        
        width = 0.23
        for i, (title, val, hist, col) in enumerate(cards):
            x = i * 0.25 + 0.01
            box = FancyBboxPatch((x, 0.1), width, 0.8, boxstyle="round,pad=0.02", color=C["panel"], ec=C["border"], lw=1)
            ax.add_patch(box)
            ax.text(x + 0.02, 0.7, title, color=C["text_lo"], fontsize=8)
            ax.text(x + 0.02, 0.35, val, color=C["text_hi"], fontsize=12, fontweight="bold")
            
            if len(hist) > 1:
                hx = np.linspace(x + 0.1, x + width - 0.01, len(hist))
                hy = np.array(hist)
                if hy.max() != hy.min():
                    hy = (hy - hy.min()) / (hy.max() - hy.min())
                else:
                    hy = np.full_like(hy, 0.5)
                hy = (hy * 0.4) + 0.2 
                ax.plot(hx, hy, color=col, lw=1.5)

    def _render_charts(self):
        self.ax_reward.clear()
        self._style_axes(self.ax_reward, "REWARD VS EPISODE")
        if self.plot_rewards:
            self.ax_reward.plot(self.plot_episodes, self.plot_rewards, color=C["blue"], lw=1.2, label="Reward")
            avg = [np.mean(self.plot_rewards[max(0, i-50):i+1]) for i in range(len(self.plot_rewards))]
            self.ax_reward.plot(self.plot_episodes, avg, color=C["purple"], lw=1.8, label="Avg(50)")
            self.ax_reward.legend(loc="upper left", frameon=False, labelcolor=C["text_lo"], fontsize=8)
            self.ax_reward.grid(True, color=C["border"], linestyle=":", alpha=0.5)

        self.ax_loss.clear()
        self._style_axes(self.ax_loss, "LOSS METRICS")
        if self.plot_actor_loss:
            self.ax_loss.plot(self.plot_episodes, self.plot_actor_loss, color=C["purple"], lw=1.2, label="Actor")
            self.ax_loss.plot(self.plot_episodes, self.plot_critic_loss, color=C["blue"], lw=1.2, label="Critic")
            self.ax_loss.legend(loc="upper left", frameon=False, labelcolor=C["text_lo"], fontsize=7)
            self.ax_loss.grid(True, color=C["border"], linestyle=":", alpha=0.5)

        self.ax_alpha.clear()
        self._style_axes(self.ax_alpha, "ALPHA METRICS")
        if self.plot_alpha:
            self.ax_alpha.plot(self.plot_episodes, self.plot_alpha, color=C["accent"], lw=1.2, label="Alpha")
            self.ax_alpha.plot(self.plot_episodes, self.plot_alpha_loss, color=C["yellow"], lw=1.2, label="Alpha Loss")
            self.ax_alpha.legend(loc="upper left", frameon=False, labelcolor=C["text_lo"], fontsize=7)
            self.ax_alpha.grid(True, color=C["border"], linestyle=":", alpha=0.5)

    def _render_map(self):
        ax = self.ax_map
        ax.clear()
        self._style_axes(ax, "MINI SWARM MAP")
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        if not hasattr(self.trainer, 'env'):
            return

        if hasattr(self.trainer.env, 'obstacles') and len(self.trainer.env.obstacles) > 0:
            obs = np.array(self.trainer.env.obstacles)
            for o in obs:
                ax.add_patch(Circle((o[0], o[1]), 2.0, color=C["red"], alpha=0.15))
            ax.scatter(obs[:,0], obs[:,1], color=C["red"], marker='x', s=20, label="Obstacle")
                
        if hasattr(self.trainer.env, 'targets') and len(self.trainer.env.targets) > 0:
            tgts = np.array(self.trainer.env.targets)
            for t in tgts:
                ax.add_patch(Circle((t[0], t[1]), 1.5, color=C["accent"], alpha=0.15))
            ax.scatter(tgts[:,0], tgts[:,1], color=C["accent"], marker='*', s=40, label="Target")

        if hasattr(self.trainer.env, 'drones') and len(self.trainer.env.drones) > 0:
            a_pos, d_pos = [], []
            for d in self.trainer.env.drones:
                p = getattr(d, 'pos', [0,0])
                if getattr(d, 'alive', True): a_pos.append(p)
                else: d_pos.append(p)
            if a_pos:
                a_pos = np.array(a_pos)
                ax.scatter(a_pos[:,0], a_pos[:,1], color=C["blue"], marker='o', s=30, label="Drone")
            if d_pos:
                d_pos = np.array(d_pos)
                ax.scatter(d_pos[:,0], d_pos[:,1], color=C["text_lo"], marker='x', s=30)

    def _render_mission(self):
        ax = self.ax_mission
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "MISSION ANALYTICS")
        
        env = getattr(self.trainer, 'env', None)
        targets = len(getattr(env, 'targets_reached', [])) if env else 0
        total_targets = len(getattr(env, 'targets', [])) if env else 0
        failed = len(getattr(env, 'failed_ids', [])) if env else 0
        alive = sum(1 for d in getattr(env, 'drones', []) if getattr(d, 'alive', True)) if env else 0
        
        y = 0.7
        ax.text(0.05, y, "Targets Reached", color=C["text_lo"], fontsize=9)
        ax.text(0.95, y, f"{targets} / {max(1, total_targets)}", color=C["accent"], fontsize=11, ha="right", fontweight="bold")
        y -= 0.2
        ax.text(0.05, y, "Failed Drones", color=C["text_lo"], fontsize=9)
        ax.text(0.95, y, f"{failed}", color=C["red"] if failed > 0 else C["text_hi"], fontsize=11, ha="right", fontweight="bold")
        y -= 0.2
        ax.text(0.05, y, "Alive Drones", color=C["text_lo"], fontsize=9)
        ax.text(0.95, y, f"{alive}", color=C["text_hi"], fontsize=11, ha="right", fontweight="bold")

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
            ("Meta", "meta_trained.pth")
        ]
        
        cx = 0.02
        cw = 0.18
        for name, fname in models:
            box = FancyBboxPatch((cx, 0.2), cw, 0.6, boxstyle="round,pad=0.02", color=C["bg"], ec=C["border"], lw=1)
            ax.add_patch(box)
            ax.text(cx + 0.02, 0.6, name, color=C["text_hi"], fontsize=8, fontweight="bold")
            
            if os.path.exists(fname):
                age = time.time() - os.path.getmtime(fname)
                if age < 60: age_str = "Just now"
                elif age < 3600: age_str = f"{int(age/60)}m ago"
                else: age_str = f"{int(age/3600)}h ago"
                ax.text(cx + 0.02, 0.35, "✓ " + age_str, color=C["accent"], fontsize=8)
            else:
                ax.text(cx + 0.02, 0.35, "Pending", color=C["text_lo"], fontsize=8)
            cx += cw + 0.015

    def _render_hyperparams(self):
        ax = self.ax_hyper
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "HYPERPARAMETERS")
        import sys
        trainer_mod = sys.modules.get('training.trainer')
        
        params = [
            ("MAX_EPISODES", getattr(cfg, 'MAX_EPISODES', getattr(trainer_mod, 'MAX_EPISODES', 'N/A'))),
            ("MAX_STEPS", getattr(cfg, 'MAX_STEPS', getattr(trainer_mod, 'MAX_STEPS', 'N/A'))),
            ("BUFFER_SIZE", getattr(cfg, 'BUFFER_SIZE', getattr(trainer_mod, 'BUFFER_SIZE', 'N/A'))),
            ("BATCH", getattr(cfg, 'BATCH', getattr(trainer_mod, 'BATCH', 'N/A'))),
            ("GAMMA", getattr(cfg, 'GAMMA', getattr(trainer_mod, 'GAMMA', 'N/A'))),
            ("LR_ACTOR", getattr(cfg, 'LR_ACTOR', getattr(trainer_mod, 'LR_ACTOR', 'N/A'))),
            ("LR_CRITIC", getattr(cfg, 'LR_CRITIC', getattr(trainer_mod, 'LR_CRITIC', 'N/A'))),
            ("UPDATE_EVERY", getattr(cfg, 'UPDATE_EVERY', getattr(trainer_mod, 'UPDATE_EVERY', 'N/A'))),
            ("UPDATES_PER_STEP", getattr(cfg, 'UPDATES_PER_STEP', getattr(trainer_mod, 'UPDATES_PER_STEP', 'N/A'))),
        ]
        
        y = 0.85
        for k, v in params:
            ax.text(0.05, y, k, color=C["text_lo"], fontsize=8)
            ax.text(0.95, y, str(v), color=C["text_hi"], fontsize=8, ha="right", fontweight="bold")
            y -= 0.1

    def _render_log(self):
        ax = self.ax_log
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "LIVE TRAINING LOG")
        
        events = getattr(self.trainer, 'log_events', [])
        y = 0.8
        for line in reversed(events[-6:]):
            color = C["text_lo"]
            if "Saved" in line: color = C["accent"]
            elif "Reward +" in line: color = C["yellow"]
            elif "Reward -" in line: color = C["red"]
            ax.text(0.02, y, line, color=color, fontsize=8, fontfamily="monospace")
            y -= 0.15

    def _render_metrics(self):
        ax = self.ax_metrics
        ax.clear()
        ax.axis("off")
        self._style_axes(ax, "REAL-TIME METRICS")
        
        metrics = [
            ("Critic Loss", getattr(self.trainer, 'last_c_loss', 0.0), C["blue"]),
            ("Actor Loss", getattr(self.trainer, 'last_a_loss', 0.0), C["purple"]),
            ("Alpha", getattr(self.trainer, 'last_alpha', 0.0), C["accent"]),
            ("Alpha Loss", getattr(self.trainer, 'last_alpha_loss', 0.0), C["yellow"])
        ]
        
        y = 0.85
        for name, val, col in metrics:
            ax.text(0.05, y, name, color=C["text_lo"], fontsize=8)
            ax.text(0.95, y, f"{val:.4g}", color=col, fontsize=10, ha="right", fontweight="bold")
            y -= 0.1
            ax.plot([0.05, 0.95], [y+0.05, y+0.05], color=C["border"], lw=0.5)
            y -= 0.1

    def update_step(self):
        ep = getattr(self.trainer, 'current_ep', 0)
        metrics = getattr(self.trainer, 'metrics', None)
        
        if ep > len(self.plot_episodes) and metrics and hasattr(metrics, 'rewards') and len(metrics.rewards) > 0:
            self.plot_episodes.append(ep)
            self.plot_rewards.append(metrics.rewards[-1])
            self.plot_critic_loss.append(getattr(self.trainer, 'last_c_loss', 0.0))
            self.plot_actor_loss.append(getattr(self.trainer, 'last_a_loss', 0.0))
            self.plot_alpha.append(getattr(self.trainer, 'last_alpha', 0.0))
            self.plot_alpha_loss.append(getattr(self.trainer, 'last_alpha_loss', 0.0))
            
            avg_r = np.mean(self.plot_rewards[-50:]) if self.plot_rewards else 0.0
            best_r = getattr(self.trainer, 'best_ep_reward', 0.0)
            if best_r == -float('inf'): best_r = 0.0
            buf_size = len(self.trainer.buf) if hasattr(self.trainer, 'buf') else 0
            tps = getattr(self.trainer, 'total_steps', 0) / max(1, time.time() - self.start_time)
            
            self.kpi_hist_avg_reward.append(avg_r)
            self.kpi_hist_best_reward.append(best_r)
            self.kpi_hist_buffer.append(buf_size)
            self.kpi_hist_speed.append(tps)

        now = time.time()
        if now - self.last_render_time > 0.5:
            self.last_render_time = now
            self._render_kpis()
            self._render_charts()
            self._render_map()
            self._render_mission()
            self._render_checkpoints()
            self._render_hyperparams()
            self._render_log()
            self._render_metrics()
            
            self.fig.canvas.draw_idle()
            plt.pause(0.01)

    def run(self):
        plt.ion()
        plt.show()
        
        try:
            while plt.fignum_exists(self.fig.number):
                self.update_step()
                if not getattr(self.trainer, 'is_training', True):
                    self.update_step()
                    print("Training completed. Dashboard will remain open until closed.")
                    plt.ioff()
                    plt.show()
                    break
        except KeyboardInterrupt:
            pass
