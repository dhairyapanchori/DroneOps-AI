"""
run_dashboard.py
────────────────
python main.py          ← train first (saves 5 .pth files)
python run_dashboard.py ← visualise
"""
import sys, os
sys.path.append(os.path.dirname(__file__))
from visualization.swarm_dashboard import SwarmDashboard

if __name__ == "__main__":
    SwarmDashboard(
        actor_path="actor_trained.pth",
        gnn_path="gnn_trained.pth",
        trans_path="trans_trained.pth",
        meta_path="meta_trained.pth",
    ).run(num_episodes=20, pause=0.04)
