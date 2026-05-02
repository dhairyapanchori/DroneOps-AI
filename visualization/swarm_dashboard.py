"""
Swarm Dashboard — v5
Uses full pipeline: MetaAdapter → GNN → Transformer → Actor
Shows drone failures, coordination, targets hit live.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import torch

from env.swarm_env import SwarmEnv, OBSTACLE_RADIUS, TARGET_RADIUS
from ml.marl.actor import Actor
from ml.gnn.swarm_gnn import SwarmGNN
from ml.transformer.mission_transformer import MissionTransformer
from ml.meta.meta_adapter import MetaAdapter
from utils.config import *


class SwarmDashboard:

    def __init__(self, actor_path="actor_trained.pth",
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

        print(f"[Dashboard] Loaded: {', '.join(loaded) if loaded else 'none (untrained)'}")
        for net in [self.actor, self.gnn, self.trans, self.meta]:
            net.eval()

        self.reward_history  = []
        self.episode_rewards = []
        self.episode_num     = 0

        self.fig = plt.figure(figsize=(15, 8), facecolor="#0d0d0d")
        self.fig.canvas.manager.set_window_title("🧬 Swarm Intelligence — v5")

        gs = GridSpec(3, 3, figure=self.fig,
                      left=0.05, right=0.97, top=0.93, bottom=0.07,
                      wspace=0.35, hspace=0.5)

        self.ax_main   = self.fig.add_subplot(gs[:, :2])
        self.ax_reward = self.fig.add_subplot(gs[0, 2])
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        self.ax_info   = self.fig.add_subplot(gs[2, 2])

        for ax in [self.ax_main, self.ax_reward, self.ax_energy, self.ax_info]:
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")

        self._setup_panels()
        plt.suptitle("🧬 Swarm Intelligence — MetaAdapter + GNN + Transformer + DDPG + Evolution",
                     color="#00ff88", fontsize=10, fontweight="bold")

    # ── Fusion (mirrors trainer exactly) ─────────────────────────────

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
            # Use deterministic mean action for clean visualisation
            if hasattr(self.actor, 'deterministic'):
                return self.actor.deterministic(fused).numpy()
            return self.actor(fused).numpy()

    # ── Panel setup ───────────────────────────────────────────────────

    def _setup_panels(self):
        ax = self.ax_main
        ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
        ax.set_title("Swarm Environment", color="#aaaaaa", fontsize=9)
        ax.grid(True, color="#1a1a1a", linewidth=0.5)
        ax.axhline(0, color="#222222", lw=0.8)
        ax.axvline(0, color="#222222", lw=0.8)

        self.drone_scatter    = ax.scatter([], [], s=90,  c="#00ff88", zorder=6, label="Drones (active)")
        self.dead_scatter     = ax.scatter([], [], s=90,  c="#444444", marker="x", zorder=6, label="Drones (failed)")
        self.obstacle_scatter = ax.scatter([], [], s=130, c="#ff4444", marker="X", zorder=4, label="Obstacles")
        self.target_scatter   = ax.scatter([], [], s=160, c="#4488ff", marker="*", zorder=5, label="Targets")
        ax.legend(loc="upper right", fontsize=6,
                  facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#cccccc")

        self.obs_rings = []; self.tgt_rings = []
        self.quivers = []; self.drone_labels = []; self.comm_lines = []

        self.ax_reward.set_title("Reward / Step", color="#aaaaaa", fontsize=8)
        self.reward_line, = self.ax_reward.plot([], [], color="#00ff88", lw=1)

        self.ax_energy.set_title("Drone Energy", color="#aaaaaa", fontsize=8)
        self.ax_energy.set_ylim(0, 1.05)
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, NUM_DRONES))
        self.energy_bars = self.ax_energy.bar(range(NUM_DRONES), [1.0]*NUM_DRONES,
                                               color=colors, width=0.6)
        self.ax_energy.set_xticks(range(NUM_DRONES))
        self.ax_energy.set_xticklabels([f"D{i}" for i in range(NUM_DRONES)],
                                        fontsize=6, color="#888888")

        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(0.05, 0.97, "",
                                            transform=self.ax_info.transAxes,
                                            color="#cccccc", fontsize=7.5,
                                            va="top", fontfamily="monospace")

    # ── Drawing ───────────────────────────────────────────────────────

    def _draw_rings(self):
        for r in self.obs_rings + self.tgt_rings: r.remove()
        self.obs_rings = []; self.tgt_rings = []
        for obs in self.env.obstacles:
            c = Circle(obs, OBSTACLE_RADIUS, color="#ff4444", fill=True, alpha=0.08, zorder=2)
            self.ax_main.add_patch(c); self.obs_rings.append(c)
        for tgt in self.env.targets:
            c = Circle(tgt, TARGET_RADIUS, color="#4488ff", fill=True, alpha=0.15, zorder=2)
            self.ax_main.add_patch(c); self.tgt_rings.append(c)

    def _draw_comm_lines(self, positions, alive_mask, threshold=8.0):
        for l in self.comm_lines: l.remove()
        self.comm_lines = []
        for i in range(NUM_DRONES):
            for j in range(i+1, NUM_DRONES):
                if not (alive_mask[i] and alive_mask[j]):
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold:
                    alpha = max(0.05, 1.0 - dist/threshold) * 0.5
                    ln, = self.ax_main.plot(
                        [positions[i][0], positions[j][0]],
                        [positions[i][1], positions[j][1]],
                        color="#00ff88", lw=0.4, alpha=alpha)
                    self.comm_lines.append(ln)

    def _draw_quivers(self, alive_mask):
        for q in self.quivers: q.remove()
        self.quivers = []
        for d in self.env.drones:
            if not d.alive: continue
            q = self.ax_main.quiver(d.pos[0], d.pos[1], d.vel[0], d.vel[1],
                                     color="#ffdd44", scale=8, width=0.004, alpha=0.7)
            self.quivers.append(q)

    def _draw_labels(self, positions, alive_mask):
        for l in self.drone_labels: l.remove()
        self.drone_labels = []
        for i, pos in enumerate(positions):
            color = "#88ffcc" if alive_mask[i] else "#666666"
            label = f"D{i}" if alive_mask[i] else f"D{i}✗"
            l = self.ax_main.text(pos[0]+0.4, pos[1]+0.4, label,
                                   color=color, fontsize=6, zorder=7)
            self.drone_labels.append(l)

    # ── Frame update ──────────────────────────────────────────────────

    def _update_frame(self, step_reward, step):
        positions  = np.array([d.pos    for d in self.env.drones])
        energies   = np.array([d.energy for d in self.env.drones])
        alive_mask = np.array([d.alive  for d in self.env.drones])

        # Separate active and dead drones
        active_pos = positions[alive_mask]
        dead_pos   = positions[~alive_mask]

        if len(active_pos) > 0:
            active_colors = plt.cm.RdYlGn(np.clip(energies[alive_mask], 0, 1))
            self.drone_scatter.set_offsets(active_pos)
            self.drone_scatter.set_color(active_colors)
        else:
            self.drone_scatter.set_offsets(np.empty((0, 2)))

        if len(dead_pos) > 0:
            self.dead_scatter.set_offsets(dead_pos)
        else:
            self.dead_scatter.set_offsets(np.empty((0, 2)))

        # Targets — flash gold when reached
        tgt_colors = []
        for ti in range(len(self.env.targets)):
            hit = any(k[1] == ti for k in self.env.targets_reached)
            tgt_colors.append("#ffd700" if hit else "#4488ff")
        self.target_scatter.set_offsets(self.env.targets)
        self.target_scatter.set_color(tgt_colors)
        self.obstacle_scatter.set_offsets(self.env.obstacles)

        self._draw_comm_lines(positions, alive_mask)
        self._draw_quivers(alive_mask)
        self._draw_labels(positions, alive_mask)

        # Reward graph
        self.reward_history.append(step_reward)
        self.reward_line.set_data(range(len(self.reward_history)), self.reward_history)
        self.ax_reward.relim(); self.ax_reward.autoscale_view()

        # Energy bars — grey out dead drones
        for i, (bar, e) in enumerate(zip(self.energy_bars, energies)):
            if alive_mask[i]:
                bar.set_height(max(0, e))
                bar.set_color(plt.cm.RdYlGn(np.clip(e, 0, 1)))
            else:
                bar.set_height(0.02)
                bar.set_color("#333333")

        # Info panel
        n_active  = int(alive_mask.sum())
        n_failed  = NUM_DRONES - n_active
        n_tgts    = len(self.env.targets_reached)
        covered   = len(set(ti for (_, ti) in self.env.targets_reached))
        coord     = covered / max(1, len(self.env.targets))

        avg_to_tgt = np.mean([
            min(np.linalg.norm(d.pos - t) for t in self.env.targets)
            for d in self.env.drones if d.alive
        ]) if n_active > 0 else 99.0

        info = (
            f"Episode    : {self.episode_num}\n"
            f"Step       : {step}/{MAX_STEPS}\n"
            f"────────────────────\n"
            f"Active     : {n_active}/{NUM_DRONES}\n"
            f"Failed     : {n_failed}\n"
            f"Targets Hit: {n_tgts}\n"
            f"Coordination:{coord:.2f}\n"
            f"Avg→Target : {avg_to_tgt:.2f}\n"
            f"Step Reward: {step_reward:+.2f}\n"
            f"────────────────────\n"
        )
        if self.episode_rewards:
            info += f"Best Ep    : {max(self.episode_rewards):.1f}\n"
            info += f"Last Ep    : {self.episode_rewards[-1]:.1f}\n"

        self.info_text.set_text(info)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ── Run ───────────────────────────────────────────────────────────

    def run(self, num_episodes=20, pause=0.04):
        self.env.curriculum_ep = 225   # ← add this
        plt.ion()
        plt.show()

        for ep in range(num_episodes):
            self.episode_num = ep + 1
            self.reward_history = []
            s    = self.env.reset()
            ep_r = 0.0
            done = False
            step = 0

            self._draw_rings()

            while not done:
                actions          = self._get_actions(s)
                s, rewards, done = self.env.step(actions)
                step_reward      = rewards.mean()
                ep_r            += step_reward
                step            += 1
                self._update_frame(step_reward, step)
                plt.pause(pause)

            self.episode_rewards.append(ep_r)
            n_failed = len(self.env.failed_ids)
            print(f"[Ep {ep+1:>3}] Reward={ep_r:.1f}  "
                  f"Targets={len(self.env.targets_reached)}  "
                  f"Failed={n_failed}")

        plt.ioff()
        plt.show()