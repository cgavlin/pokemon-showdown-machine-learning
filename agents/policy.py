"""
Policy network for the battle agent.

Starts simple on purpose: a feed-forward Dueling DQN over the fixed-size
observation vector produced by environment/state.py, with masked
action selection so illegal actions are never chosen at inference time
even though the env would also catch and penalize them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_size = obs_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        self.trunk = nn.Sequential(*layers)

        self.value_head = nn.Linear(in_size, 1)
        self.advantage_head = nn.Linear(in_size, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


def masked_argmax(q_values: np.ndarray, action_mask: np.ndarray) -> int:
    """Pick the highest-Q legal action; never returns an illegal index."""
    masked = np.where(action_mask.astype(bool), q_values, -np.inf)
    return int(np.argmax(masked))


def epsilon_greedy_action(
    q_network: DuelingQNetwork,
    obs: np.ndarray,
    action_mask: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    legal_indices = np.flatnonzero(action_mask)
    if len(legal_indices) == 0:
        return 0  # should not happen; env guarantees action 0 is always legal

    if rng.random() < epsilon:
        return int(rng.choice(legal_indices))

    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_network(obs_t).squeeze(0).cpu().numpy()
    return masked_argmax(q_values, action_mask)
