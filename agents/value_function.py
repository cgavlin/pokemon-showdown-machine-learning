"""
Experience replay buffer.

Named value_function.py to match the CLAUDE.md suggested architecture
(agents/{policy,value_function,inference}); in this Dueling-DQN setup
the "value function" concern is the Q-value target/replay machinery
that the policy network is trained against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int, n_actions: int, seed: int | None = None):
        self.capacity = capacity
        self.obs_size = obs_size
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.zeros((capacity, n_actions), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, n_actions), dtype=np.float32)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def add(self, t: Transition) -> None:
        i = self._ptr
        self.obs[i] = t.obs
        self.next_obs[i] = t.next_obs
        self.actions[i] = t.action
        self.rewards[i] = t.reward
        self.dones[i] = float(t.done)
        self.action_masks[i] = t.action_mask
        self.next_action_masks[i] = t.next_action_mask

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "action_masks": self.action_masks[idx],
            "next_action_masks": self.next_action_masks[idx],
        }
