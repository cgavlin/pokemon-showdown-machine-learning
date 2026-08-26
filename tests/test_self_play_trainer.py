"""
Tests for training/self_play_trainer.py's training loop.

This exercises the loop's bookkeeping (replay buffer growth from BOTH players, 
episode counting, checkpointing, metrics logging) without requiring a live Showdown
server: SelfPlayTrainer.env is swapped for a MockParallelEnv that
mimics poke-env's two-agent parallel step/reset contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from environment.rewards import RewardConfig
from environment.state import observation_size
from training.self_play_trainer import SelfPlayTrainer
from training.trainer import TrainingConfig


class _FakeBattle:
    def __init__(self, won: bool):
        self.won = won


class MockParallelEnv:
    """Mimics poke-env PokeEnv's two-agent parallel API contract:
    reset() -> (observations, infos); step(actions) -> (observations,
    rewards, terminated, truncated, infos), all dicts keyed by agent
    username. `.agents` is cleared to [] once a battle finishes,
    exactly as poke-env's PokeEnv.step does. `.battle1`/`.battle2` are
    set at that same point too, mimicking each player's own
    perspective on the just-finished battle -- deterministically,
    player1 always "wins" in this mock, so tests can check the
    resulting player1_won/player2_won fields without real battle logic."""

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
        self.battle1 = None
        self.battle2 = None

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
            self.battle1 = _FakeBattle(won=True)
            self.battle2 = _FakeBattle(won=False)
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


def test_self_play_trainer_logs_per_player_win_loss(tmp_path):
    """MockParallelEnv deterministically makes player1 "win" every
    episode -- this checks episode_end actually reports that, not just
    reward, so a caller (e.g. --learn's training summary) can compute a
    real win rate."""
    trainer = _make_trainer(tmp_path)
    trainer.train()
    lines = [json.loads(l) for l in trainer.metrics_log_path.read_text().strip().split("\n")]
    episode_lines = [l for l in lines if l.get("event") == "episode_end"]
    assert len(episode_lines) > 0
    for line in episode_lines:
        assert line["player1_won"] is True
        assert line["player2_won"] is False


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


def test_init_checkpoint_warm_starts_the_network_instead_of_random_init(tmp_path):
    # Train a first trainer briefly, save its checkpoint, then build a
    # second trainer with init_checkpoint pointing at it -- the second
    # trainer's freshly-constructed network should start with those
    # EXACT weights, not a new random initialization.
    trainer_a = _make_trainer(tmp_path / "a")
    checkpoint_path = trainer_a.train()

    trainer_b = SelfPlayTrainer(
        training_config=TrainingConfig(total_steps=1, batch_size=4, device="cpu"),
        reward_config=RewardConfig(),
        run_dir=tmp_path / "b",
        env=MockParallelEnv(n_actions=26, obs_size=observation_size()),
        init_checkpoint=checkpoint_path,
    )

    a_state = trainer_a.q_network.state_dict()
    b_state = trainer_b.q_network.state_dict()
    for key in a_state:
        assert (a_state[key] == b_state[key]).all(), f"mismatched weights for {key}"

    # Also recorded in run_metadata.json for reproducibility.
    metadata = json.loads((trainer_b.run_dir / "run_metadata.json").read_text())
    assert metadata["init_checkpoint"] == str(checkpoint_path)


def test_no_init_checkpoint_records_null_in_metadata(tmp_path):
    trainer = _make_trainer(tmp_path)
    metadata = json.loads((trainer.run_dir / "run_metadata.json").read_text())
    assert metadata["init_checkpoint"] is None


def test_missing_init_checkpoint_raises_before_opening_a_connection(tmp_path, monkeypatch):
    """A bad init_checkpoint path must fail immediately with a clear
    error -- before EncodedSinglesEnv (a real websocket connection) is
    even constructed. Verified here by making EncodedSinglesEnv itself
    raise if called, so the test fails loudly if the ordering ever
    regresses (e.g. someone moves the validation after env setup)."""
    import training.self_play_trainer as spt_module

    def _must_not_be_called(**kwargs):
        raise AssertionError(
            "EncodedSinglesEnv must not be constructed when init_checkpoint is invalid"
        )

    monkeypatch.setattr(spt_module, "EncodedSinglesEnv", _must_not_be_called)

    cfg = TrainingConfig(total_steps=1, batch_size=4, device="cpu")
    with pytest.raises(FileNotFoundError, match="init_checkpoint not found"):
        SelfPlayTrainer(
            training_config=cfg,
            reward_config=RewardConfig(),
            run_dir=tmp_path,
            init_checkpoint=str(tmp_path / "missing.pt"),
        )


def test_select_actions_returns_numpy_scalars_not_plain_python_ints(tmp_path):
    """Regression test for a bug only a live server run surfaced:
    poke-env's own PokeEnv.step() -> SinglesEnv.action_to_order calls
    `.item()` on each agent's action, which plain Python ints don't
    have. MockParallelEnv (used by every other test in this file)
    never enforced this -- it happily accepts a dict of plain ints --
    so this needs a test that checks the actual TYPE _select_actions
    produces, not just that training completes without error."""
    trainer = _make_trainer(tmp_path)
    observations = {
        "Player1 A": {
            "observation": np.zeros(observation_size(), dtype=np.float32),
            "action_mask": np.ones(26, dtype=np.int64),
        },
        "Player2 B": {
            "observation": np.zeros(observation_size(), dtype=np.float32),
            "action_mask": np.ones(26, dtype=np.int64),
        },
    }
    actions = trainer._select_actions(observations, ["Player1 A", "Player2 B"], epsilon=1.0)

    for agent_id, action in actions.items():
        assert isinstance(action, np.integer), (
            f"action for {agent_id} is {type(action).__name__}, expected a numpy "
            "integer -- poke-env's action_to_order calls .item() on this value"
        )