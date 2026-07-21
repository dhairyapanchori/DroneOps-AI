"""Training entry point — runs the full SAC + curriculum + evolution pipeline.

Saves five checkpoints (actor / critic / gnn / trans / meta) to the project
root on completion; visualise them with run_dashboard.py.
"""

import threading
from training.trainer import Trainer

if __name__ == "__main__":
    from visualization.training_dashboard import TrainingDashboard
    
    trainer = Trainer()
    dashboard = TrainingDashboard(trainer)
    
    # Run training in background thread so dashboard can run in main thread
    train_thread = threading.Thread(target=trainer.train, daemon=True)
    train_thread.start()
    
    # Run dashboard in main thread (blocking until window is closed or training ends)
    dashboard.run()
