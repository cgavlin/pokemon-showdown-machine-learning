"""
Regression test for evaluate_agent's perspective bug: it must read
metrics from battle1 (OUR agent's view, per poke-env's SingleAgentWrapper
convention: agent1/battle1 = the main agent whose actions are passed
into env.step()) and NEVER from battle2 (the opponent's mirrored view,
where "team" is the opponent's team and "won"/"lost" are from their
side). Mixing these up silently flips win rate and damage-dealt/taken
for every battle -- there is no exception or type error to catch it,
just wrong numbers, which is why this needs a standing regression test
rather than relying on the earlier live-debugging session that caught
it once.
"""

from __future__ import annotations

import numpy as np

from evaluation.benchmarks import evaluate_agent


class _FakePokemon:
    def __init__(self, fainted: bool = False, hp: float = 1.0):
        self.fainted = fainted
        self.current_hp_fraction = hp


class _FakeBattle1:
    """OUR agent's perspective: we won; opponent has 2 fainted, we have 0."""

    team = {"a": _FakePokemon(hp=1.0), "b": _FakePokemon(hp=1.0)}
    opponent_team = {
        "x": _FakePokemon(fainted=True, hp=0.0),
        "y": _FakePokemon(fainted=True, hp=0.0),
    }
    won = True
    lost = False


class _FakeBattle2:
    """The opponent's mirrored perspective on the SAME battle: from
    their point of view they lost, their own team is fainted, and
    "opponent_team" (from their side) is us, still at full HP."""

    team = {"x": _FakePokemon(fainted=True, hp=0.0), "y": _FakePokemon(fainted=True, hp=0.0)}
    opponent_team = {"a": _FakePokemon(hp=1.0), "b": _FakePokemon(hp=1.0)}
    won = False
    lost = True


class _FakeUnderlying:
    battle1 = _FakeBattle1()
    battle2 = _FakeBattle2()


class _FakeEnv:
    _underlying = _FakeUnderlying()

    def reset(self, seed=None):
        return np.zeros(4, dtype=np.float32), {}

    def get_action_mask(self):
        return np.ones(4, dtype=np.int64)

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 1.0, True, False, {}


class _FakeAgent:
    def act(self, obs, mask):
        return 0


def test_evaluate_agent_reads_metrics_from_our_agents_perspective():
    result = evaluate_agent(_FakeEnv(), _FakeAgent(), n_battles=1)
    summary = result.summary()

    assert summary["win_rate"] == 1.0
    assert summary["loss_rate"] == 0.0
    assert summary["avg_own_fainted"] == 0
    assert summary["avg_opponent_fainted"] == 2
