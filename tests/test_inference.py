"""
Direct unit tests for agents/inference.py's TrainedAgent.

Previously only exercised indirectly (via evaluation/benchmarks.py and
the pooled self-play trainer tests). These tests build real, small
checkpoints on disk and load them for real -- no network involved --
so they also serve as a regression check on the checkpoint file format
(the "q_network" / "metadata" keys) every trainer in this project
saves to and TrainedAgent reads from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agents.inference import TrainedAgent
from agents.policy import DuelingQNetwork
from environment.state import observation_size

N_ACTIONS = 5


def _write_checkpoint(path, net: DuelingQNetwork, metadata: dict | None = None) -> None:
    payload = {"q_network": net.state_dict()}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)


def test_trained_agent_loads_matching_weights_from_checkpoint(tmp_path):
    net = DuelingQNetwork(observation_size(), N_ACTIONS)
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, net)

    agent = TrainedAgent(checkpoint_path, n_actions=N_ACTIONS, device="cpu")

    expected_state = net.state_dict()
    actual_state = agent.network.state_dict()
    for key in expected_state:
        assert torch.equal(expected_state[key], actual_state[key]), f"mismatched weights for {key}"


def test_trained_agent_puts_the_network_in_eval_mode(tmp_path):
    net = DuelingQNetwork(observation_size(), N_ACTIONS)
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, net)

    agent = TrainedAgent(checkpoint_path, n_actions=N_ACTIONS, device="cpu")

    assert agent.network.training is False


def test_trained_agent_stores_metadata_from_checkpoint(tmp_path):
    net = DuelingQNetwork(observation_size(), N_ACTIONS)
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, net, metadata={"step": 42, "run_id": "abc123"})

    agent = TrainedAgent(checkpoint_path, n_actions=N_ACTIONS, device="cpu")

    assert agent.metadata == {"step": 42, "run_id": "abc123"}


def test_trained_agent_defaults_metadata_to_empty_dict_when_absent(tmp_path):
    net = DuelingQNetwork(observation_size(), N_ACTIONS)
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, net, metadata=None)

    agent = TrainedAgent(checkpoint_path, n_actions=N_ACTIONS, device="cpu")

    assert agent.metadata == {}


def test_trained_agent_missing_q_network_key_raises(tmp_path):
    checkpoint_path = tmp_path / "bad_checkpoint.pt"
    torch.save({"not_q_network": {}}, checkpoint_path)

    with pytest.raises(KeyError):
        TrainedAgent(checkpoint_path, n_actions=N_ACTIONS, device="cpu")


def test_trained_agent_missing_checkpoint_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainedAgent(tmp_path / "does_not_exist.pt", n_actions=N_ACTIONS, device="cpu")


# --- act(): masking behavior -----------------------------------------------


class _FixedQNetwork(nn.Module):
    """Always returns the same fixed Q-values regardless of input --
    isolates TrainedAgent.act()'s masking/argmax logic from whatever a
    real (randomly initialized) network would happen to output."""

    def __init__(self, q_values: list[float]):
        super().__init__()
        self._q_values = torch.tensor(q_values, dtype=torch.float32)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        return self._q_values.unsqueeze(0).repeat(batch_size, 1)


def _agent_with_fixed_q_values(q_values: list[float], tmp_path) -> TrainedAgent:
    net = DuelingQNetwork(observation_size(), len(q_values))
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, net)
    agent = TrainedAgent(checkpoint_path, n_actions=len(q_values), device="cpu")
    agent.network = _FixedQNetwork(q_values)
    return agent


def test_act_picks_the_highest_q_legal_action(tmp_path):
    agent = _agent_with_fixed_q_values([1.0, 5.0, 2.0], tmp_path)
    mask = np.array([1, 1, 1], dtype=np.int64)

    action = agent.act(np.zeros(observation_size(), dtype=np.float32), mask)

    assert action == 1


def test_act_never_returns_a_masked_out_action(tmp_path):
    agent = _agent_with_fixed_q_values([1.0, 5.0, 2.0], tmp_path)
    # The best action (index 1) is masked out -- the agent must fall
    # back to the best REMAINING legal action (index 2), never index 1.
    mask = np.array([1, 0, 1], dtype=np.int64)

    action = agent.act(np.zeros(observation_size(), dtype=np.float32), mask)

    assert action == 2


def test_act_returns_a_plain_python_int(tmp_path):
    agent = _agent_with_fixed_q_values([1.0, 5.0, 2.0], tmp_path)
    mask = np.array([1, 1, 1], dtype=np.int64)

    action = agent.act(np.zeros(observation_size(), dtype=np.float32), mask)

    assert isinstance(action, int)