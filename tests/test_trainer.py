"""
Tests for training/trainer.py's Trainer construction-time behavior --
specifically `init_checkpoint` warm-starting, added so
training/curriculum_runner.py can carry a checkpoint from one
curriculum stage into the next instead of every stage starting from a
freshly, randomly initialized network.

Trainer.train() itself needs a real ShowdownBattleEnv (a live local
Showdown server) to exercise meaningfully, so -- matching the pattern
used for the other trainers' offline tests -- only __init__ behavior is
covered here, against a minimal fake env satisfying the small surface
Trainer.__init__/_write_run_metadata actually touch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from agents.policy import DuelingQNetwork
from environment.rewards import RewardConfig
from environment.state import observation_size
from training.trainer import Trainer, TrainingConfig

N_ACTIONS = 26


class _FakeOpponent:
    format = "gen9randombattle"


class _FakeEnv:
    """Minimal stand-in for ShowdownBattleEnv covering only what
    Trainer.__init__ and _write_run_metadata read."""

    def __init__(self, n_actions=N_ACTIONS):
        self.action_space = type("Sp", (), {"n": n_actions})()
        self.opponent = _FakeOpponent()
        self.n_invalid_actions = 0
        self.n_total_actions = 0

    def reset(self, seed=None):
        return np.zeros(observation_size(), dtype=np.float32), {}

    def get_action_mask(self):
        return np.ones(N_ACTIONS, dtype=np.int64)

    def close(self):
        pass


def _make_training_config(**overrides) -> TrainingConfig:
    return TrainingConfig(total_steps=10, batch_size=4, device="cpu", **overrides)


def test_no_init_checkpoint_starts_from_a_fresh_random_network(tmp_path):
    trainer = Trainer(
        env=_FakeEnv(),
        training_config=_make_training_config(),
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        curriculum_stage_name="stage1_basic_mechanics",
    )
    assert trainer.init_checkpoint is None
    metadata = json.loads((trainer.run_dir / "run_metadata.json").read_text())
    assert metadata["init_checkpoint"] is None


def test_init_checkpoint_warm_starts_the_network_instead_of_random_init(tmp_path):
    net = DuelingQNetwork(observation_size(), N_ACTIONS)
    checkpoint_path = tmp_path / "prev_stage_checkpoint.pt"
    torch.save({"q_network": net.state_dict(), "metadata": {}}, checkpoint_path)

    trainer = Trainer(
        env=_FakeEnv(),
        training_config=_make_training_config(),
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        curriculum_stage_name="stage2_tactical_decisions",
        init_checkpoint=checkpoint_path,
    )

    expected_state = net.state_dict()
    actual_state = trainer.q_network.state_dict()
    for key in expected_state:
        assert torch.equal(expected_state[key], actual_state[key]), f"mismatched weights for {key}"

    # Target network is a deep copy taken AFTER the warm-start load, so
    # it must match too -- otherwise the very first target update would
    # silently discard the warm-started weights on the target side.
    target_state = trainer.target_network.state_dict()
    for key in expected_state:
        assert torch.equal(expected_state[key], target_state[key])

    metadata = json.loads((trainer.run_dir / "run_metadata.json").read_text())
    assert metadata["init_checkpoint"] == str(checkpoint_path)


def test_init_checkpoint_missing_q_network_key_raises(tmp_path):
    bad_checkpoint = tmp_path / "bad_checkpoint.pt"
    torch.save({"not_q_network": {}}, bad_checkpoint)

    with pytest.raises(ValueError, match="q_network"):
        Trainer(
            env=_FakeEnv(),
            training_config=_make_training_config(),
            reward_config=RewardConfig(),
            run_dir=tmp_path,
            curriculum_stage_name="stage2_tactical_decisions",
            init_checkpoint=bad_checkpoint,
        )