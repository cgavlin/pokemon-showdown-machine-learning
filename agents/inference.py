"""
Loading a trained checkpoint and using it to act -- used by evaluation,
self-play opponent sampling, and (eventually, behind the safety gate)
live Showdown play.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agents.policy import DuelingQNetwork, masked_argmax
from environment.state import observation_size


class TrainedAgent:
    """Thin wrapper: obs + action_mask in, legal action index out."""

    def __init__(self, checkpoint_path: str | Path, n_actions: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.network = DuelingQNetwork(observation_size(), n_actions).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(state_dict["q_network"])
        self.network.eval()
        self.metadata = state_dict.get("metadata", {})

    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.network(obs_t).squeeze(0).cpu().numpy()
        return masked_argmax(q_values, action_mask)
