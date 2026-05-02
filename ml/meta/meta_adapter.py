import torch
import torch.nn as nn


class MetaAdapter(nn.Module):
    """
    FiLM-style state normalisation conditioned on swarm context.

    Supports both (N, state_dim) and (B, N, state_dim) input.
    Context is computed as the mean across the N (drone) dimension.
    """

    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.scale_head = nn.Linear(hidden, state_dim)
        self.shift_head = nn.Linear(hidden, state_dim)

    def forward(self, states):
        """
        states : (N, state_dim)  or  (B, N, state_dim)
        returns: same shape as states
        """
        # Mean over drone dimension (second-to-last)
        context = states.mean(dim=-2, keepdim=True)          # (..., 1, state_dim)
        ctx     = self.context_encoder(context)               # (..., 1, hidden)
        scale   = torch.sigmoid(self.scale_head(ctx)) + 0.5  # (..., 1, state_dim)
        shift   = self.shift_head(ctx)                        # (..., 1, state_dim)
        return states * scale + shift                         # (..., N, state_dim)