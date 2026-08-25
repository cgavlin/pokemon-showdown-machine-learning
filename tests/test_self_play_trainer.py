"""
Tests for training/self_play_trainer.py's training loop.

Per CLAUDE.md's testing requirements, this exercises the loop's
bookkeeping (replay buffer growth from BOTH players, episode counting,
checkpointing, metrics logging) without requiring a live Showdown
server: SelfPlayTrainer.env is swapped for a MockParallelEnv that
mimics poke-env's two-agent parallel step/reset contract.
"""

from __future__ import annotations

import json

import numpy as np

from environment.rewards import RewardConfig
from environment.state import observation_size
from training.self_play_trainer import SelfPlayTrainer
from training.trainer import TrainingConfig


class MockParallelEnv:
    """Mimics poke-env PokeEnv's two-agent parallel API contract:
    reset() -> (observations, infos); step(actions) -> (observations,
    rewards, terminated, truncated, infos), all dicts keyed by agent
    username. `.agents` is cleared to [] once a battle finishes,
    exactly as poke-env's PokeEnv.step does."""

    def __init__(self, n_actions: int, obs_size: int, max_turns: int = 5):
        self.possible_agents = ["Player1 A", "Player2 B"]
        self.agents = list(self.possible_agents)
        self.action_spaces = {a: type("S", (), {"n": n_actions})() for a in self.possible_agents}
        self._obs_size = obs_size
        self._n_actions = n_actions
        self._turn = 0
        self._max_turns = max_turns
        self._rng = np.random.default_rng(0)
        self.closed = False

    def _obs_dict(self):
        return {
            a: {
                "observation": self._rng.standard_normal(self._obs_size).astype(np.float32),
                "action_mask": np.ones(self._n_actions, dtype=np.int64),
            }
            for a in self.agents
        }

    def reset(self, seed=None):
        self.agents = list(self.possible_agents)
        self._turn = 0
        return self._obs_dict(), {}

    def step(self, actions):
        self._turn += 1
        done = self._turn >= self._max_turns
        obs = self._obs_dict()
        rewards = {a: 1.0 for a in self.agents}
        terminated = {a: done for a in self.agents}
        truncated = {a: False for a in self.agents}
        if done:
            self.agents = []
        return obs, rewards, terminated, truncated, {}

    def close(self):
        self.closed = True


def _make_trainer(tmp_path, **cfg_overrides) -> SelfPlayTrainer:
    cfg = TrainingConfig(
        total_steps=40,
        batch_size=4,
        replay_capacity=200,
        learning_starts=2,
        train_every=1,
        target_update_every=5,
        eval_every_steps=10,
        epsilon_decay_steps=10,
        **cfg_overrides,
    )
    # n_actions must be picked before construction since SelfPlayTrainer
    # reads it from env.action_spaces to size the Q-network; use the
    # same gen9 singles action space size the real env would report.
    n_actions = 26
    mock_env = MockParallelEnv(n_actions=n_actions, obs_size=observation_size())
    # Inject the mock env directly so no real websocket connection is
    # ever attempted -- this test needs no live Showdown server.
    trainer = SelfPlayTrainer(
        training_config=cfg, reward_config=RewardConfig(), run_dir=tmp_path, env=mock_env
    )
    return trainer


def test_self_play_trainer_constructs_two_distinct_players(tmp_path):
    trainer = _make_trainer(tmp_path)
    assert trainer.player1_username != trainer.player2_username
    assert trainer.player1_username == "Player1 A"
    assert trainer.player2_username == "Player2 B"


def test_self_play_buffer_receives_transitions_from_both_players(tmp_path):
    """Each step should push one transition per player, not just one
    -- this is the whole point of self-play: the network learns from
    both sides of every battle."""
    trainer = _make_trainer(tmp_path)
    trainer.train()
    # 40 steps x 2 players per step = 80 transitions.
    assert len(trainer.buffer) == 80


def test_self_play_trainer_logs_per_player_episode_rewards(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.train()
    lines = [json.loads(l) for l in trainer.metrics_log_path.read_text().strip().split("\n")]
    episode_lines = [l for l in lines if l.get("event") == "episode_end"]
    assert len(episode_lines) > 0
    for line in episode_lines:
        assert "player1_reward" in line
        assert "player2_reward" in line


def test_self_play_trainer_saves_checkpoints(tmp_path):
    trainer = _make_trainer(tmp_path)
    final_checkpoint = trainer.train()
    assert final_checkpoint.exists()
    # Closing the env is the CALLER's responsibility (matching Trainer's
    # convention) -- train() itself doesn't auto-close, so a caller that
    # crashes mid-training or wants to close explicitly, both work the
    # same way via trainer.env.close().
    assert trainer.env.closed is False
    trainer.env.close()
    assert trainer.env.closed is True
