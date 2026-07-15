import torch.nn as nn


class MissionTransformer(nn.Module):
    """
    Transformer encoder over the swarm — each drone is a token.

    Self-attention lets every drone attend to every other drone's
    (adapted) state, complementing the GNN's uniform mean-pooling with
    content-dependent mixing. No positional encoding on purpose: a
    swarm is a set, not a sequence, so the encoding must stay
    permutation-equivariant.

    Supports both (N, state_dim) and (B, N, state_dim) input.
    TransformerEncoder with batch_first=True handles the batch dim natively.

    Frozen at initialisation in this project (fixed feature encoder) —
    see training/trainer.py for the rationale.
    """

    def __init__(self, state_dim=16, d_model=64):
        super().__init__()
        self.proj = nn.Linear(state_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, states):
        """
        states : (N, state_dim)  or  (B, N, state_dim)
        returns: same leading dims, last dim = 64
        """
        squeeze = states.dim() == 2
        if squeeze:
            states = states.unsqueeze(0)     # (1, N, state_dim)

        x = self.proj(states)                # (B, N, d_model)
        e = self.encoder(x)                  # (B, N, d_model)
        o = self.out(e)                      # (B, N, d_model)

        if squeeze:
            o = o.squeeze(0)                 # (N, d_model)
        return o