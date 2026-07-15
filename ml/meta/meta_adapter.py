import torch
import torch.nn as nn


class MetaAdapter(nn.Module):
    """
    FiLM-style state modulation conditioned on swarm context.

    The mean state across drones is encoded into per-feature scale and
    shift parameters applied to every drone's observation. The sigmoid
    keeps scale in [0.5, 1.5], so the adapter can re-weight features
    based on the swarm's situation but never zero them out or blow
    them up — a bounded, stability-friendly transform.

    Supports both (N, state_dim) and (B, N, state_dim) input.
    Context is computed as the mean across the N (drone) dimension.

    Frozen at initialisation in this project (fixed feature encoder) —
    see training/trainer.py for the rationale.
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