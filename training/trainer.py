"""
DroneOps AI Trainer — SAC Edition
─────────────────────────────────────────
Algorithm: Soft Actor-Critic (SAC)
  - Entropy-regularised objective keeps policy from collapsing
  - Twin critics prevent Q overestimation
  - Automatic temperature tuning (alpha) — no manual noise schedule
  - No Ornstein-Uhlenbeck noise needed — stochastic actor handles exploration

Architecture — who learns and who doesn't
  All six drones share ONE actor and ONE critic (parameter sharing).
  A hierarchical MissionPlanner sits above the control stack: each step it
  reads swarm telemetry, tracks mission phase (SEARCH/RESCUE/RETURN) and
  objective status, and issues advisory per-drone directives — the plug-in
  point for future Dynamic Task Allocation. It is deterministic and does
  not feed the policy, so training trajectories are unaffected.
  Observations pass through a fusion pipeline:

      state (N,16) → MetaAdapter (FiLM) → GNN ┐
                                   └→ Transformer ┴→ concat (N,144)

  The three representation networks are FROZEN at initialisation and act as
  fixed random feature encoders. Rationale: they are shared between actor and
  critic, and letting both optimise them couples the two losses and causes
  circular instability — fixed weights give the critic stationary targets.
  Only the actor head, critic heads, and temperature alpha receive gradients.

Three complementary selection pressures act on the actor:
  1. SAC gradient steps (every UPDATE_EVERY env steps) — the main learner.
  2. Best-actor snapshot: if an episode scores below 70% of the best seen,
     the best weights are restored with small noise — a cheap collapse guard.
  3. EvolutionEngine: every EVOLVE_EVERY episodes a (1+λ) ES mutates the
     actor and keeps the champion — escapes local optima gradients can't.

Curriculum:
  Phase 0 (ep   0–74):  No obstacles, close targets, no failures
  Phase 1 (ep  75–149): 2 obstacles, rare failures
  Phase 2 (ep 150–224): 4 obstacles, 3 targets, occasional failures
  Phase 3 (ep 225–499): Full difficulty
"""

import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from env.swarm_env import SwarmEnv
from ml.marl.actor  import SACActorNet
from ml.marl.critic import SACCriticNet
from ml.gnn.swarm_gnn import SwarmGNN
from ml.transformer.mission_transformer import MissionTransformer
from ml.meta.meta_adapter import MetaAdapter
from ml.evolution.evolution_engine import EvolutionEngine
from ml.planner.mission_planner import MissionPlanner
from utils.replay_buffer import ReplayBuffer
from utils.config import *
from metrics.logger import Metrics

UPDATE_EVERY     = 4   # update every 4 env steps — critic stabilises between updates
UPDATES_PER_STEP = 2   # gradient steps per update round


class Trainer:

    def __init__(self):
        self.env = SwarmEnv()

        # Mission-command layer — advisory, sits above the control stack
        self.planner = MissionPlanner()

        # Actor — stochastic Gaussian policy
        self.actor  = SACActorNet(FUSED_DIM, ACTION_DIM)

        # Twin critics
        self.critic        = SACCriticNet(FUSED_DIM, ACTION_DIM)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Representation networks — frozen during training
        # Shared between actor and critic; updating them couples the two
        # and causes circular instability. Fixed weights = stable critic targets.
        self.gnn   = SwarmGNN(STATE_DIM)
        self.trans = MissionTransformer(STATE_DIM)
        self.meta  = MetaAdapter(STATE_DIM)

        for net in [self.gnn, self.trans, self.meta]:
            for p in net.parameters():
                p.requires_grad = False   # frozen

        self.gnn_target   = copy.deepcopy(self.gnn)
        self.trans_target = copy.deepcopy(self.trans)
        self.meta_target  = copy.deepcopy(self.meta)

        # Temperature (alpha) — learnable, starts at 0.2, minimum 0.05
        self.log_alpha     = torch.tensor(np.log(0.2), requires_grad=True)
        self.alpha_min     = 0.05
        self.target_entropy = torch.tensor(TARGET_ENTROPY)

        # Optimisers — actor head only (representation networks are frozen)
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.evo            = EvolutionEngine(self.actor, pop_size=EVOLVE_POP, sigma=EVOLVE_SIGMA)
        self.buf            = ReplayBuffer(BUFFER_SIZE)
        self.metrics        = Metrics()
        self.total_steps    = 0
        self.best_ep_reward = -float('inf')
        self.best_actor_state = None

        # Expose state for dashboard
        self.current_ep = 0
        self.is_training = False
        self.log_events = []
        self.last_c_loss = 0.0
        self.last_a_loss = 0.0
        self.last_alpha = 0.2
        self.last_alpha_loss = 0.0

    @property
    def alpha(self):
        # Clamp so alpha never drops below alpha_min
        return self.log_alpha.exp().clamp(min=self.alpha_min)

    # ── Fusion pipeline ───────────────────────────────────────────────

    def _fuse(self, s_t, meta_net, gnn_net, trans_net):
        """Build the fused per-drone embedding: [adapted | gnn | transformer].

        s_t: (N, STATE_DIM) or (B, N, STATE_DIM) → same leading dims, FUSED_DIM.
        """
        adapted = meta_net(s_t)
        gnn_ctx = gnn_net(adapted)
        mission = trans_net(adapted)
        return torch.cat([adapted, gnn_ctx, mission], dim=-1)

    def _soft_update(self, online, target):
        """Polyak-average online parameters into the target network (rate TAU)."""
        for po, pt in zip(online.parameters(), target.parameters()):
            pt.data.copy_(TAU * po.data + (1.0 - TAU) * pt.data)

    # ── SAC update ────────────────────────────────────────────────────

    def _update(self):
        """One SAC gradient step: critic → actor → temperature → target sync.

        Returns (critic_loss, actor_loss, alpha, alpha_loss) or (None, None, None, None)
        while the replay buffer is still warming up.
        """
        if len(self.buf) < WARMUP_STEPS:
            return None, None, None, None

        s, a, r, ns, d = self.buf.sample(BATCH)
        B, N = BATCH, NUM_DRONES

        # Un-flatten to (B, N, ·) so the fusion nets can pool across the
        # swarm dimension, then re-flatten: each drone becomes an
        # independent SAC sample that shares its swarm's context.
        s_r  = s.view(B, N, STATE_DIM)
        ns_r = ns.view(B, N, STATE_DIM)

        fused_s  = self._fuse(s_r,  self.meta,        self.gnn,        self.trans
                               ).view(B * N, FUSED_DIM)
        fused_ns = self._fuse(ns_r, self.meta_target, self.gnn_target, self.trans_target
                               ).view(B * N, FUSED_DIM)

        # ── Critic update ─────────────────────────────────────────────
        # Soft Bellman target: r + γ(1-d)·[min(Q1',Q2') - α·logπ(a'|s')]
        # min over twin critics counters overestimation bias; the -α·logπ
        # term folds the entropy objective into the value estimate.
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(fused_ns)
            q1_next, q2_next  = self.critic_target(fused_ns, next_a)
            q_next     = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
            q_target   = r + GAMMA * (1.0 - d) * q_next

        q1_pred, q2_pred = self.critic(fused_s, a)
        c_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.critic_opt.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────
        fused_s2 = self._fuse(s_r, self.meta, self.gnn, self.trans
                               ).view(B * N, FUSED_DIM)
        a_new, logp_new = self.actor.sample(fused_s2)
        q1_new = self.critic.q1_only(fused_s2, a_new)

        # SAC actor loss: minimise (alpha * log_pi - Q)
        # This has a natural fixed point — no unbounded drift
        a_loss = (self.alpha.detach() * logp_new - q1_new).mean()

        self.actor_opt.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Temperature (alpha) update ────────────────────────────────
        # Automatically adjust entropy: if policy is too deterministic,
        # raise alpha to encourage exploration; if too random, lower it.
        with torch.no_grad():
            _, logp_new2 = self.actor.sample(fused_s2)
        alpha_loss = (-self.log_alpha * (logp_new2 + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft-update critic target only (representation nets are frozen, targets = online)
        self._soft_update(self.critic, self.critic_target)

        return c_loss.item() / 2, a_loss.item(), self.alpha.item(), alpha_loss.item()

    # ── Evolution fitness rollout ─────────────────────────────────────

    def _fitness(self, actor):
        """Roll out `actor` deterministically for one episode; return mean reward.

        Used by the EvolutionEngine to score mutated actor candidates.
        """
        s    = self.env.reset()
        ep_r = 0.0
        done = False
        while not done:
            with torch.no_grad():
                fused = self._fuse(torch.FloatTensor(s),
                                   self.meta, self.gnn, self.trans)
                a = actor.deterministic(fused).numpy()
            s, r, done = self.env.step(a)
            ep_r += r.mean()
        return ep_r

    # ── Main training loop ────────────────────────────────────────────

    def train(self):
        """Full training run: curriculum episodes → SAC updates → evolution → save."""
        self.is_training = True

        for ep in tqdm(range(MAX_EPISODES)):
            self.current_ep = ep + 1
            # Tell environment which episode we're on (for curriculum)
            self.env.curriculum_ep = ep
            s    = self.env.reset()
            self.planner.begin_mission(self.env)
            ep_r = 0.0
            done = False

            while not done:
                s_t = torch.FloatTensor(s)
                with torch.no_grad():
                    fused = self._fuse(s_t, self.meta, self.gnn, self.trans)
                    # During warmup use random actions; then sample from policy
                    if self.total_steps < WARMUP_STEPS:
                        a = np.random.uniform(-1, 1, (NUM_DRONES, ACTION_DIM)).astype(np.float32)
                    else:
                        a = self.actor.sample(fused)[0].numpy()

                ns, r, done = self.env.step(a)
                self.planner.update(self.env)
                done_arr    = np.full(NUM_DRONES, float(done))
                self.buf.add(s, a, r / REWARD_SCALE, ns, done_arr)  # scale rewards
                self.total_steps += 1

                if self.total_steps % UPDATE_EVERY == 0:
                    for _ in range(UPDATES_PER_STEP):
                        c, al, alpha, al_loss = self._update()
                        if c is not None:
                            self.last_c_loss  = c
                            self.last_a_loss  = al
                            self.last_alpha   = alpha
                            self.last_alpha_loss = al_loss

                s     = ns
                ep_r += r.mean()

            # Track and possibly restore best actor.
            # Restore-with-noise (not plain restore) so the policy resumes
            # exploring from the good region instead of retracing the exact
            # trajectory that led to the collapse.
            if ep_r > self.best_ep_reward:
                self.best_ep_reward   = ep_r
                self.best_actor_state = copy.deepcopy(self.actor.state_dict())
            elif (self.best_actor_state is not None and
                  ep_r < self.best_ep_reward * 0.7 and ep > 50):
                self.actor.load_state_dict(self.best_actor_state)
                for p in self.actor.parameters():
                    p.data += torch.randn_like(p.data) * 0.005

            n_failed        = len(self.env.failed_ids)
            n_targets_hit   = len(self.env.targets_reached)
            targets_covered = set(ti for (_, ti) in self.env.targets_reached)
            coordination    = len(targets_covered) / max(1, len(self.env.targets))
            phase           = self.env._curriculum_phase()
            mission         = self.planner.mission_summary()

            self.metrics.log(
                reward=ep_r, targets_hit=n_targets_hit,
                drones_failed=n_failed, coordination=coordination,
                a_loss=self.last_a_loss if self.last_c_loss else None, 
                c_loss=self.last_c_loss if self.last_c_loss else None,
                mission_phase=mission["final_phase"],
            )

            if self.last_c_loss is not None:
                # ASCII-only console output — Windows terminals often use
                # cp1252, which cannot encode Greek letters or emoji.
                loss_str = f"  C={self.last_c_loss:.3f} A={self.last_a_loss:.3f} alpha={self.last_alpha:.3f}"
            else:
                loss_str = "  [warmup]"

            log_str = (f"Episode {ep+1} | Reward {ep_r:+.1f} | Mission {mission['final_phase']} | "
                       f"Targets {n_targets_hit} | Failures {n_failed} | "
                       f"Coord {coordination:.2f} | ")
            if self.last_c_loss is not None:
                log_str += f"Critic: {self.last_c_loss:.4f} | Actor: {self.last_a_loss:.3f} | Alpha: {self.last_alpha:.3f}"
            else:
                log_str += "Warmup"
            self.log_events.append(log_str)
            if len(self.log_events) > 100:
                self.log_events.pop(0)

            print(f"Ep {ep+1:>3}  R={ep_r:+7.1f}  "
                  f"Tgts={n_targets_hit}  "
                  f"Fail={n_failed}  "
                  f"Coord={coordination:.2f}  "
                  f"Ph={phase}  "
                  f"M={mission['final_phase']}"
                  f"{loss_str}")

            if (ep + 1) % EVOLVE_EVERY == 0:
                print(f"\n  [Evolution] Running {EVOLVE_POP} mutants...")
                best_actor, best_score, improved = self.evo.evolve(self._fitness)
                if improved:
                    self.actor = best_actor
                    self.evo.actor = best_actor
                    if best_score > self.best_ep_reward:
                        self.best_ep_reward   = best_score
                        self.best_actor_state = copy.deepcopy(self.actor.state_dict())
                    print(f"  [Evolution] Improved to {best_score:.2f}")
                else:
                    print(f"  [Evolution] No improvement (best mutant: {best_score:.2f})")
                self.metrics.log_evolution(best_score)

        print("\n=== Training Complete ===")
        if self.best_actor_state is not None:
            self.actor.load_state_dict(self.best_actor_state)
            print(f"Restored best actor (peak: {self.best_ep_reward:.1f})")
        print(self.metrics.summary())

        torch.save(self.actor.state_dict(),  "actor_trained.pth")
        torch.save(self.critic.state_dict(), "critic_trained.pth")
        torch.save(self.gnn.state_dict(),    "gnn_trained.pth")
        torch.save(self.trans.state_dict(),  "trans_trained.pth")
        torch.save(self.meta.state_dict(),   "meta_trained.pth")
        print("Saved: actor / critic / gnn / trans / meta")
        self.log_events.append("Checkpoints Saved")
        self.is_training = False