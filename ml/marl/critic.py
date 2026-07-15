import torch
import torch.nn as nn


class SACCriticNet(nn.Module):
    """
    Twin Q-network for SAC.
    Both Q1 and Q2 in one module — forward returns (q1, q2).
    Taking min(q1, q2) prevents overestimation: a single critic's errors
    are optimistically exploited by the actor; two independently
    initialised critics rarely overestimate the same state-action pair.
    """
    def __init__(self, in_dim, action_dim):
        super().__init__()
        def _net():
            return nn.Sequential(
                nn.Linear(in_dim + action_dim, 256), nn.ReLU(),
                nn.Linear(256, 256),                 nn.ReLU(),
                nn.Linear(256, 1),
            )
        self.q1 = _net()
        self.q2 = _net()

    def forward(self, state, action):
        x  = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, state, action):
        """Q1 head alone — the actor loss needs one estimate, and skipping
        Q2 halves the cost of the policy update."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# Keep backward-compatible name
Critic = SACCriticNet