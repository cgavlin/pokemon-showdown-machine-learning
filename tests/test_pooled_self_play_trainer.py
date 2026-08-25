"""
Tests for training/pooled_self_play_trainer.py's PooledSelfPlayTrainer.

Verifies, without any live Showdown server, that:
  - the learner starts against a scripted bootstrap opponent while the
    pool is empty;
  - opponents actually get swapped every `episodes_per_opponent`
    episodes, and each swap closes the old env and opens a new one;
  - the learner's own checkpoints get added to the pool as it trains,
    so later swaps sample past versions of itself instead of only the
    scripted bootstrap;
  - win rate against the just-finished opponent is recorded back into
    the pool (SelfPlayPool.update_win_rate), which is the whole point
    of wiring the pool into training at all -- without it, challenging
    opponents never get pinned and the sampling never adapts.

Both the environment and the scripted bootstrap-opponent factory are
replaced with lightweight fakes: `env_factory` is a constructor
argument the trainer already supports for exactly this purpose, and
`OPPONENT_FACTORIES` is monkeypatched so no real poke-env Player (which
would otherwise open a background websocket connection attempt) is
ever constructed.
"""

from __future__ import annotations

import json

import numpy as np

from environment.rewards import RewardConfig
from environment.state import observation_size
from training import pooled_self_play_trainer as pspt
from training.trainer import TrainingConfig

# Matches poke-env's real gen9 singles action space size, so a
# checkpoint saved by the trainer always loads cleanly back into
# CheckpointPlayer's network (which independently computes the same
# number from the battle format) when a swap samples it as an opponent.
N_ACTIONS = 26


class _FakeOpponent:
    """Stands in for a real poke-env Player (e.g. RandomPlayer) without
    ever opening a network connection. The mock env below never calls
    any of its methods -- it only needs to exist and be inert."""


def _fake_opponent_factory(battle_format, team):
    return _FakeOpponent()


class _FakeBattle:
    def __init__(self, won: bool):
        self.won = won
        self.lost = not won
        self.team = {}
        self.opponent_team = {}
        # evaluate_agent (used by the absolute-skill eval check) also
        # classifies move effectiveness per-action; an empty
        # available_moves list means every action is treated as a
        # switch/default for that purpose, which is fine here since
        # these tests don't assert on effectiveness stats.
        self.available_moves = []
        self.opponent_active_pokemon = None


class _FakeUnderlying:
    def __init__(self):
        self.battle1 = _FakeBattle(won=False)


class MockSingleAgentEnv:
    """Mimics ShowdownBattleEnv's single-agent API contract closely
    enough for the training loop: reset()->(obs, info),
    get_action_mask()->mask, step(action)->(obs, reward, terminated,
    truncated, info), close(), plus the `_underlying.battle1.won`
    attribute the trainer reads at episode end (matching the
    battle1-not-battle2 convention fixed in evaluation/benchmarks.py)."""

    def __init__(self, opponent, n_actions=N_ACTIONS, obs_size=None, episode_len=3, win_every_n=2):
        self.opponent = opponent
        self.n_actions = n_actions
        self.obs_size = obs_size or observation_size()
        self.episode_len = episode_len
        self.win_every_n = win_every_n
        self._turn = 0
        self._episode_idx = 0
        self._underlying = _FakeUnderlying()
        self.action_space = type("Sp", (), {"n": n_actions})()
        self.closed = False
        self._rng = np.random.default_rng(0)

    def _obs(self):
        return self._rng.standard_normal(self.obs_size).astype(np.float32)

    def reset(self, seed=None):
        self._turn = 0
        return self._obs(), {}

    def get_action_mask(self):
        return np.ones(self.n_actions, dtype=np.int64)

    def step(self, action):
        self._turn += 1
        done = self._turn >= self.episode_len
        if done:
            self._episode_idx += 1
            won = self._episode_idx % self.win_every_n == 0
            self._underlying.battle1 = _FakeBattle(won=won)
        return self._obs(), 1.0, done, False, {}

    def close(self):
        self.closed = True


def _make_trainer(tmp_path, monkeypatch, episodes_per_opponent=2, total_steps=40):
    monkeypatch.setattr(pspt, "OPPONENT_FACTORIES", {"random": _fake_opponent_factory})

    created_envs = []

    def env_factory(opponent):
        env = MockSingleAgentEnv(opponent)
        created_envs.append(env)
        return env

    cfg = TrainingConfig(
        total_steps=total_steps,
        batch_size=4,
        replay_capacity=200,
        learning_starts=2,
        train_every=1,
        target_update_every=5,
        eval_every_steps=10,
        epsilon_decay_steps=10,
    )
    trainer = pspt.PooledSelfPlayTrainer(
        training_config=cfg,
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        episodes_per_opponent=episodes_per_opponent,
        env_factory=env_factory,
    )
    return trainer, created_envs


def test_starts_against_scripted_bootstrap_when_pool_is_empty(tmp_path, monkeypatch):
    trainer, created_envs = _make_trainer(tmp_path, monkeypatch)
    assert isinstance(trainer.env.opponent, _FakeOpponent)
    assert trainer._current_opponent_entry is None


def test_opponents_swap_and_old_envs_get_closed(tmp_path, monkeypatch):
    trainer, created_envs = _make_trainer(tmp_path, monkeypatch)
    trainer.train()

    assert len(created_envs) > 1, "expected at least one opponent swap over 40 steps"
    # Every env that got REPLACED by a swap must be closed by the swap
    # itself. The final env is the caller's responsibility to close
    # (matching Trainer's convention -- train() doesn't auto-close),
    # so it should still be open right after train() returns.
    assert all(env.closed for env in created_envs[:-1]), "every replaced env must be closed"
    assert created_envs[-1].closed is False
    trainer.env.close()
    assert created_envs[-1].closed is True


def test_learner_checkpoints_are_added_to_the_pool(tmp_path, monkeypatch):
    trainer, _ = _make_trainer(tmp_path, monkeypatch)
    trainer.train()

    # At least one swap must have happened, each adding a checkpoint,
    # plus the final checkpoint added at the end of train().
    assert len(trainer.pool.entries) >= 2
    for entry in trainer.pool.entries:
        assert entry.checkpoint_path.startswith(str(trainer.pool.pool_dir))


def test_later_swaps_sample_past_checkpoints_not_only_bootstrap(tmp_path, monkeypatch):
    trainer, _ = _make_trainer(tmp_path, monkeypatch)
    trainer.train()

    lines = [json.loads(l) for l in trainer.metrics_log_path.read_text().strip().split("\n")]
    opponents_seen = {l["opponent"] for l in lines if l.get("event") == "episode_end"}

    assert "scripted:random" in opponents_seen
    checkpoint_opponents = {o for o in opponents_seen if o != "scripted:random"}
    assert len(checkpoint_opponents) > 0, "expected later episodes to play against pool checkpoints"
    assert all(o.startswith("checkpoint_step") for o in checkpoint_opponents)


def test_win_rate_against_opponent_is_recorded_back_into_pool(tmp_path, monkeypatch):
    trainer, _ = _make_trainer(tmp_path, monkeypatch)
    trainer.train()

    # MockSingleAgentEnv is deterministic (win_every_n=2), so at least
    # one pool entry must end up with a win rate that moved away from
    # the freshly-added default of 0.5 once it was actually sampled and
    # played against.
    sampled_entries = [e for e in trainer.pool.entries if e.times_sampled > 0]
    assert len(sampled_entries) > 0
    assert any(e.win_rate_vs_current != 0.5 for e in sampled_entries)


def test_fixed_opponent_eval_disabled_by_default(tmp_path, monkeypatch):
    """eval_every_n_swaps=0 (the default) must never trigger an eval --
    this is also what keeps every other test in this file offline."""
    eval_calls = []
    trainer, _ = _make_trainer(tmp_path, monkeypatch)
    trainer._run_fixed_opponent_eval = lambda *a, **k: eval_calls.append((a, k))
    trainer.train()
    assert eval_calls == []


def test_fixed_opponent_eval_runs_every_n_swaps_and_logs_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pspt,
        "OPPONENT_FACTORIES",
        {"random": _fake_opponent_factory, "heuristic": _fake_opponent_factory},
    )

    created_envs = []
    created_eval_envs = []

    def env_factory(opponent):
        env = MockSingleAgentEnv(opponent)
        created_envs.append(env)
        return env

    def eval_env_factory(opponent):
        # A fresh env per eval, exactly like evaluate_agent expects:
        # deterministic win/loss so summary()'s win_rate is checkable.
        env = MockSingleAgentEnv(opponent, episode_len=2, win_every_n=1)  # always "wins"
        created_eval_envs.append(env)
        return env

    cfg = TrainingConfig(
        total_steps=40,
        batch_size=4,
        replay_capacity=200,
        learning_starts=2,
        train_every=1,
        target_update_every=5,
        eval_every_steps=10,
        epsilon_decay_steps=10,
    )
    trainer = pspt.PooledSelfPlayTrainer(
        training_config=cfg,
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        episodes_per_opponent=2,
        env_factory=env_factory,
        eval_env_factory=eval_env_factory,
        eval_opponent="heuristic",
        eval_every_n_swaps=2,  # trigger on the 2nd, 4th, ... swap
        eval_battles=3,
    )
    trainer.train()

    assert len(created_eval_envs) > 0, "expected at least one absolute-skill eval to run"
    assert all(env.closed for env in created_eval_envs), "every eval env must be closed after use"

    lines = [json.loads(l) for l in trainer.metrics_log_path.read_text().strip().split("\n")]
    eval_lines = [l for l in lines if l.get("event") == "eval_vs_fixed_opponent"]
    assert len(eval_lines) > 0
    for line in eval_lines:
        assert line["opponent"] == "heuristic"
        assert "win_rate" in line
        assert line["win_rate"] == 1.0  # eval env is rigged to always "win"
