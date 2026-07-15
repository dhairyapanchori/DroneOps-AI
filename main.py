"""Training entry point — runs the full SAC + curriculum + evolution pipeline.

Saves five checkpoints (actor / critic / gnn / trans / meta) to the project
root on completion; visualise them with run_dashboard.py.
"""

from training.trainer import Trainer

if __name__ == "__main__":
    Trainer().train()
