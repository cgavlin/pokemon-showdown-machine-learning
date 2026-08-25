"""
Tests for evaluation/benchmarks.py's move-effectiveness classification
(n_super_effective_moves / n_ineffective_moves), which used to be
hardcoded to 0 in evaluate_agent's output even though the same
>=2 / <1 multiplier cutoffs already existed in
environment/rewards.py's move_effectiveness_reward for training.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from evaluation.benchmarks import _classify_action_effectiveness, evaluate_agent


def _fake_type(name):
    return SimpleNamespace(name=name.upper())


def _fake_move(move_type="fire"):
    return SimpleNamespace(type=_fake_type(move_type))


def _fake_opponent_pokemon(multiplier):
    return SimpleNamespace(damage_multiplier=lambda move: multiplier)


def _fake_battle(moves, opponent_multiplier):
    return SimpleNamespace(
        available_moves=moves,
        opponent_active_pokemon=_fake_opponent_pokemon(opponent_multiplier),
    )


def test_classify_action_effectiveness_super_effective():
    battle = _fake_battle([_fake_move()], opponent_multiplier=2.0)
    is_super, is_ineffective = _classify_action_effectiveness(battle, action_index=1, action_space_size=26)
    assert is_super is True
    assert is_ineffective is False


def test_classify_action_effectiveness_ineffective():
    battle = _fake_battle([_fake_move()], opponent_multiplier=0.5)
    is_super, is_ineffective = _classify_action_effectiveness(battle, action_index=1, action_space_size=26)
    assert is_super is False
    assert is_ineffective is True


def test_classify_action_effectiveness_neutral_is_neither():
    battle = _fake_battle([_fake_move()], opponent_multiplier=1.0)
    is_super, is_ineffective = _classify_action_effectiveness(battle, action_index=1, action_space_size=26)
    assert is_super is False
    assert is_ineffective is False


def test_classify_action_effectiveness_switch_is_always_neither():
    battle = _fake_battle([_fake_move()], opponent_multiplier=4.0)
    # action_index=9 is a switch in the [default, move1-4, move+tera1-4,
    # switch1-6] layout describe_action assumes.
    is_super, is_ineffective = _classify_action_effectiveness(battle, action_index=9, action_space_size=26)
    assert is_super is False
    assert is_ineffective is False


def test_classify_action_effectiveness_none_battle_is_safe():
    is_super, is_ineffective = _classify_action_effectiveness(None, action_index=1, action_space_size=26)
    assert (is_super, is_ineffective) == (False, False)


def test_classify_action_effectiveness_slot_out_of_range_is_safe():
    battle = _fake_battle([_fake_move()], opponent_multiplier=2.0)
    # Only 1 move available; action_index=4 asks for move slot 3.
    is_super, is_ineffective = _classify_action_effectiveness(battle, action_index=4, action_space_size=26)
    assert (is_super, is_ineffective) == (False, False)


class _FakeUnderlying:
    def __init__(self, battle):
        self.battle1 = battle


class _FakeEnv:
    """A fake env whose sole battle is rigged so action 1 (move slot 0)
    is always super-effective, letting evaluate_agent's aggregation be
    checked end-to-end without a live server."""

    def __init__(self):
        self.action_space = type("Sp", (), {"n": 5})()
        battle = SimpleNamespace(
            available_moves=[_fake_move()],
            opponent_active_pokemon=_fake_opponent_pokemon(multiplier=2.0),
            team={},
            opponent_team={},
            won=True,
            lost=False,
        )
        self._underlying = _FakeUnderlying(battle)
        self._turn = 0

    def reset(self, seed=None):
        self._turn = 0
        return np.zeros(4, dtype=np.float32), {}

    def get_action_mask(self):
        return np.ones(5, dtype=np.int64)

    def step(self, action):
        self._turn += 1
        done = self._turn >= 2
        return np.zeros(4, dtype=np.float32), 1.0, done, False, {}


class _FakeAgent:
    def act(self, obs, mask):
        return 1  # always pick move slot 0 -- the rigged super-effective move


def test_evaluate_agent_surfaces_super_effective_move_counts():
    result = evaluate_agent(_FakeEnv(), _FakeAgent(), n_battles=3)
    summary = result.summary()

    # 2 moves per battle x 3 battles, all super-effective.
    assert summary["move_effectiveness_rate"] == 1.0
    assert summary["move_ineffectiveness_rate"] == 0.0
