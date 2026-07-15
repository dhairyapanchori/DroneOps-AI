import torch
import torch.nn as nn


class SwarmGNN(nn.Module):
    """
    Graph Neural Network for swarm communication.

    One round of message passing with mean aggregation over a
    fully-connected drone graph: each drone's output mixes its own
    embedding with the swarm average, giving every drone a view of
    the collective state that its local observation lacks.

    Supports both single-episode (N, state_dim) and batched
    (B, N, state_dim) input so the trainer can process a full
    replay-buffer batch in one vectorised forward pass.

    Note: in this project the network is frozen at initialisation and
    used as a fixed random feature encoder — see training/trainer.py
    for the rationale.
    """

    def __init__(self, state_dim):
        super().__init__()
        self.embed = nn.Linear(state_dim, 64)
        self.msg   = nn.Linear(64, 64)
        # Use Linear instead of GRUCell so it handles arbitrary batch shapes
        self.update = nn.Sequential(
            nn.Linear(128, 64),   # concat(msg, h) → 128
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, states, adj=None):
        """
        states : (N, state_dim)  or  (B, N, state_dim)
        adj    : ignored — always uses fully-connected mean aggregation
        returns: same leading dims as states, last dim = 64
        """
        h   = self.embed(states)                   # (..., N, 64)
        m   = self.msg(h)                          # (..., N, 64)
        # Mean aggregation over the N dimension (works for any batch shape)
        agg = m.mean(dim=-2, keepdim=True).expand_as(m)  # (..., N, 64)
        out = self.update(torch.cat([m, agg], dim=-1))    # (..., N, 64)
        return out