<div align="center">

# 🛰️ Drone Swarm Intelligence

**Multi-Agent Reinforcement Learning for Autonomous Drone Swarm Coordination**

*Soft Actor-Critic · Graph Neural Networks · Transformers · Evolutionary Search · Curriculum Learning*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-readable-brightgreen)](.)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](../../pulls)

[Overview](#-overview) •
[Features](#-key-features) •
[Architecture](#-architecture-overview) •
[Installation](#-installation) •
[Usage](#-usage) •
[Dashboard](#-dashboard) •
[Roadmap](#-future-work)

</div>

---

## 📖 Overview

**Drone Swarm Intelligence** is a research-grade playground for **cooperative multi-agent reinforcement learning (MARL)**. A swarm of six simulated drones learns — from scratch — to spread across a 2D world, reach mission targets, avoid obstacle zones, manage a limited energy budget, and keep operating when teammates fail mid-mission.

The learning stack combines a modern **Soft Actor-Critic (SAC)** core with a swarm-aware representation pipeline (**FiLM-style context adapter → Graph Neural Network + Transformer encoder**), a **4-phase curriculum** that ramps up difficulty during training, and a periodic **(1 + λ) evolutionary search** that hill-climbs the policy in weight space to escape local optima.

Everything is observable: a real-time **matplotlib mission dashboard** renders the swarm, its communication links, per-drone energy, live rewards, and failures — plus a standalone **7-scene demo mode** for showcasing the trained policy.

## 🎯 Problem Statement

Coordinating a drone swarm is hard because no single drone sees the whole picture:

- **Cooperation without a leader** — drones must implicitly divide targets among themselves rather than crowd the same one.
- **Partial, local observations** — each drone only senses its own kinematics, energy, neighbor distances, and the nearest obstacle/target.
- **Robustness to failure** — drones can die before or during a mission; survivors must re-cover the lost drone's targets.
- **Resource constraints** — every action drains energy; an exhausted drone goes offline permanently.
- **Sparse + shaped rewards** — reaching a target pays a one-time bonus; the swarm must also learn smooth navigation from dense shaping signals.

This project frames all of the above as a shared-policy MARL problem and solves it end-to-end with a single training command.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Soft Actor-Critic (SAC)** | Twin critics, entropy-regularised stochastic policy, automatic temperature (α) tuning, tanh-squashed Gaussian actions |
| 🕸️ **Swarm GNN** | Message-passing layer with mean aggregation gives every drone a view of the collective state |
| 🔀 **Mission Transformer** | 2-layer Transformer encoder treats each drone as a token for global mission context |
| 🎛️ **FiLM Meta-Adapter** | Context-conditioned feature modulation (scale + shift) normalises observations against swarm statistics |
| 🧬 **Evolutionary Search** | Elitist (1 + 8) evolution strategy periodically mutates the actor and keeps the fittest — a safety net against local optima |
| 📚 **Curriculum Learning** | 4 phases: from open field → 5 obstacles, 3 targets, random drone failures |
| 💥 **Failure Simulation** | Binomial pre-flight failures and mid-flight energy deaths; the swarm learns to adapt |
| 🔋 **Energy Model** | Action-proportional energy drain; drones die at zero energy |
| 📊 **Live Dashboard** | Real-time swarm map, comm links, velocity vectors, energy bars, reward curves |
| 🎬 **Demo Mode** | 7 scripted scenes (Ring Gauntlet, Hex Grid, Energy Crisis, …) with interactive target-focus buttons |

## 🛠️ Technologies

- **[PyTorch](https://pytorch.org/)** — neural networks, SAC optimisation, checkpointing
- **[NumPy](https://numpy.org/)** — simulation physics and vectorised environment math
- **[Matplotlib](https://matplotlib.org/)** — real-time dashboard and demo visualisation
- **[tqdm](https://github.com/tqdm/tqdm)** — training progress bars
- **Pure Python** environment — no external simulator required

## 🏗️ Architecture Overview

```
                        ┌──────────────────────────────┐
                        │          SwarmEnv            │
                        │  6 drones · obstacles ·      │
                        │  targets · energy · failures │
                        └──────────────┬───────────────┘
                                       │  per-drone state (6 × 16)
                                       ▼
                        ┌──────────────────────────────┐
                        │     MetaAdapter (FiLM)       │
                        │  swarm-context scale + shift │
                        └──────┬───────────────┬───────┘
                               │               │
                     ┌─────────▼────┐   ┌──────▼──────────┐
                     │   SwarmGNN   │   │ MissionTrans-   │
                     │  (6 × 64)    │   │ former (6 × 64) │
                     └─────────┬────┘   └──────┬──────────┘
                               │               │
                               └──────┬────────┘
                                      ▼
                        fused embedding (6 × 144)
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
        ┌─────────────────┐  ┌────────────────┐  ┌─────────────────┐
        │  SAC Actor      │  │  Twin Critics  │  │ EvolutionEngine │
        │  π(a|s) tanh-   │  │  Q1, Q2 + tgt  │  │ (1+8)-ES on the │
        │  Gaussian       │  │  networks      │  │ actor weights   │
        └─────────────────┘  └────────────────┘  └─────────────────┘
```

**Training loop:** environment rollouts fill a 200k-transition replay buffer → SAC updates run every 4 environment steps (2 gradient steps each) → the critic target network is Polyak-averaged (τ = 0.005) → every 50 episodes the evolution engine evaluates 8 mutated actors and keeps the champion → the best actor snapshot is restored at the end and all networks are checkpointed.

**Curriculum:**

| Phase | Episodes | Obstacles | Targets | Failures |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 – 74 | 0 | 2 (tight) | none |
| 1 | 75 – 149 | 2 | 2 | rare |
| 2 | 150 – 224 | 4 | 3 | occasional |
| 3 | 225 – 499 | 5 | 3 (wide) | full |

## 📁 Folder Structure

```
drone-swarm/
├── main.py                     # Training entry point
├── run_dashboard.py            # Visualise a trained swarm
├── swarm_dashboard_demo.py     # Standalone 7-scene showcase demo
│
├── core/
│   └── drone.py                # Single-drone physics, energy, observations
├── env/
│   └── swarm_env.py            # Swarm environment, rewards, curriculum
├── training/
│   └── trainer.py              # SAC training loop + evolution + checkpoints
├── ml/
│   ├── marl/
│   │   ├── actor.py            # SAC stochastic actor (tanh-Gaussian)
│   │   └── critic.py           # Twin Q-networks
│   ├── gnn/
│   │   └── swarm_gnn.py        # Swarm message-passing network
│   ├── transformer/
│   │   └── mission_transformer.py  # Drones-as-tokens encoder
│   ├── meta/
│   │   └── meta_adapter.py     # FiLM context adapter
│   └── evolution/
│       └── evolution_engine.py # (1+λ) evolution strategy
├── utils/
│   ├── config.py               # All hyperparameters
│   └── replay_buffer.py        # Uniform experience replay
├── metrics/
│   └── logger.py               # Per-episode training metrics
├── visualization/
│   └── swarm_dashboard.py      # Live matplotlib dashboard
│
├── *_trained.pth               # Pretrained checkpoints (actor, critic, gnn, transformer, meta)
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
# 1. Clone
git clone https://github.com/dhairyapanchori/drone-swarm.git
cd drone-swarm

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Requirements:** Python 3.10+ · runs on CPU — no GPU needed.

## 💡 Usage

### Train from scratch

```bash
python main.py
```

Trains for 500 episodes (~150 steps each) through all 4 curriculum phases and saves five checkpoints in the project root:

```
actor_trained.pth · critic_trained.pth · gnn_trained.pth · trans_trained.pth · meta_trained.pth
```

### Visualise a trained swarm

```bash
python run_dashboard.py
```

Runs 20 full-difficulty episodes with the deterministic policy and renders the live dashboard.

### Run the showcase demo

```bash
python swarm_dashboard_demo.py
```

Loops through 7 curated scenes forever — perfect for presentations. Pretrained checkpoints are included, so this works out of the box.

## 🔄 Training Pipeline

1. **Warm-up** — first 1 000 steps use uniform random actions to seed the replay buffer.
2. **Rollout** — observations pass through MetaAdapter → GNN + Transformer → fused 144-dim embedding → SAC actor samples actions.
3. **Experience replay** — transitions stored with reward scaling (÷5) in a 200 k buffer.
4. **SAC updates** — every 4 env steps: twin-critic regression against the entropy-regularised target, actor update, automatic α update, Polyak target sync.
5. **Stability guard** — the best actor is snapshotted; if performance collapses below 70 % of the best, the snapshot is restored with small parameter noise.
6. **Evolution** — every 50 episodes, 8 mutated actors are rolled out; the champion replaces the current actor if it scores higher.
7. **Checkpointing** — the best actor is restored and all five networks are saved.

## 📊 Dashboard

The dashboard renders in real time:

- 🗺️ **Swarm map** — energy-coloured drones, failed drones (grey ✗), obstacle penalty zones, targets (⭐ turn gold when reached)
- 🔗 **Communication links** — proximity-based link rendering between drones
- ➡️ **Velocity vectors** — per-drone heading and speed
- 🔋 **Energy bars** — live per-drone energy with colour gradient
- 📈 **Reward curve** — mean step reward across the episode
- ℹ️ **Mission panel** — active/failed counts, targets hit, coordination score

**Demo scenes:** Open Field · Ring Gauntlet · Cross Formation · Hex Grid · Energy Crisis · Binomial Failure · Star Pattern (with interactive T0/T1/T2 focus buttons).

## 🏆 Results

With the included pretrained checkpoints, the swarm demonstrates:

- ✅ Emergent **target splitting** — drones distribute across separate targets rather than clustering
- ✅ **Obstacle-aware navigation** through ring, cross, hex, and star obstacle fields
- ✅ **Failure resilience** — surviving drones re-cover targets after teammates go offline
- ✅ Stable SAC training across all curriculum phases with automatic entropy tuning

*Retrain any time with `python main.py` — full training takes well under an hour on a modern CPU.*

## 📸 Screenshots

> *Placeholders — add captures to `docs/assets/` and update the links below.*

| Training Dashboard | Demo — Ring Gauntlet |
|:-:|:-:|
| ![Dashboard](docs/assets/dashboard.png) | ![Ring Gauntlet](docs/assets/ring_gauntlet.png) |

| Energy Crisis Scene | Failure Adaptation |
|:-:|:-:|
| ![Energy Crisis](docs/assets/energy_crisis.png) | ![Failure Adaptation](docs/assets/failure_adaptation.png) |

## 🔮 Future Work

- [ ] Train the representation networks (GNN / Transformer / MetaAdapter) end-to-end instead of using fixed encoders
- [ ] Distance-based graph topology in the GNN (currently fully-connected mean aggregation)
- [ ] Proper time-limit bootstrapping (terminated vs. truncated)
- [ ] Configurable swarm size (currently fixed at 6 drones)
- [ ] Seeded, reproducible training runs + evaluation protocol
- [ ] TensorBoard / CSV metric logging and periodic checkpoints
- [ ] Gymnasium-compatible environment API
- [ ] 3D physics, drone-drone collision, and wind disturbance models
- [ ] GPU / vectorised-environment support for faster training

## 👥 Contributors

| | |
|---|---|
| **Dhairya Panchori** | Creator & Maintainer — [@dhairyapanchori](https://github.com/dhairyapanchori) |

Contributions are welcome! Feel free to open an [issue](../../issues) or submit a [pull request](../../pulls).

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this project helped or inspired you, consider giving it a star! ⭐**

Built with 🧠 + 🛰️ by [Dhairya Panchori](https://github.com/dhairyapanchori)

*Soft Actor-Critic · Graph Neural Networks · Transformers · Evolution · One Swarm*

</div>
