import torch
import torch.nn as nn


LOG_STD_MIN = -5
LOG_STD_MAX = 2


class SACActorNet(nn.Module):
    """
    SAC stochastic actor — outputs a Gaussian distribution over actions.

    The tanh squash bounds actions to [-1, 1]; its log-det-Jacobian
    correction in `sample` keeps the log-probabilities exact, which the
    entropy term of SAC depends on. Uses the reparameterisation trick so
    gradients flow through sampled actions into the network.

    Heads are initialised near zero so the initial policy is a broad,
    roughly centred Gaussian — uniform-ish early exploration.
    """
    def __init__(self, in_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
        )
        self.mean_head    = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

        nn.init.uniform_(self.mean_head.weight,    -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias,      -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias,   -3e-3, 3e-3)

    def forward(self, x):
        h       = self.net(x)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        """
        Returns (action, log_prob) using reparameterisation + tanh squashing.
        action is in [-1, 1].
        """
        mean, log_std = self.forward(x)
        std  = log_std.exp()
        eps  = torch.randn_like(mean)
        raw  = mean + eps * std                        # reparameterised sample

        action   = torch.tanh(raw)
        # log prob with tanh correction
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(raw)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, x):
        """Mean action for evaluation (no sampling)."""
        mean, _ = self.forward(x)
        return torch.tanh(mean)


# Keep backward-compatible name
Actor = SACActorNet