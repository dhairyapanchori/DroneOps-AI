import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Uniform experience replay for full swarm transitions.

    Each entry stores one environment step for the whole swarm:
    (states, actions, rewards, next_states, dones), each with a
    leading NUM_DRONES dimension. `sample` flattens the drone
    dimension so every drone becomes an independent training sample.
    """

    def __init__(self, size):
        self.buf = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample `batch_size` swarm steps, flattened to per-drone tensors."""
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        def to_t(lst):
            return torch.tensor(np.array(lst), dtype=torch.float32)

        s  = to_t(states).view(-1,  states[0].shape[-1])
        a  = to_t(actions).view(-1, actions[0].shape[-1])
        r  = to_t(rewards).view(-1, 1)
        ns = to_t(next_states).view(-1, next_states[0].shape[-1])
        d  = to_t(dones).view(-1, 1)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)