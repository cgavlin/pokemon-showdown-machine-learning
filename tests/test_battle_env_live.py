"""
Regression tests for three bugs in environment/battle_env.py, all
found the same way: by actually running training against a real local
Showdown server. Every mocked/offline test in this project (elsewhere
in the suite) had been built on assumptions about poke-env's API that
turned out to be subtly wrong in ways that never surfaced until a real
connection exercised the real code:

1. Observation-dict unwrapping: poke-env's own PokeEnv.reset()/step()
   (called via SingleAgentWrapper) always return
   {"observation": ..., "action_mask": ...} per agent, not a flat
   array -- built-in action-masking support we don't use (we compute
   our own mask separately). ShowdownBattleEnv.reset()/step() must
   unwrap this to the flat array the rest of the codebase (Trainer,
   ReplayBuffer, epsilon_greedy_action) expects.

2. Action type: poke-env's SinglesEnv.action_to_order calls `.item()`
   on the action; ShowdownBattleEnv.step() must convert to a numpy
   scalar (environment/actions.py's as_poke_env_action) before handing
   it to poke-env.

3. Reward-state key collision: EncodedSinglesEnv.calc_reward() is
   called once per agent per step with that agent's own Battle object
   (self.battle1 for agent1, self.battle2 for agent2) -- two DIFFERENT
   objects representing opposite perspectives on the SAME real battle.
   poke-env gives them the IDENTICAL battle_tag string, so keying
   incremental reward state by battle_tag alone made agent1's and
   agent2's calc_reward() calls silently overwrite each other's
   prev_own_hp_total/prev_opp_hp_total tracking every step, corrupting
   the resulting reward for both sides. This one didn't crash or
   error -- it just silently produced wrong numbers, which is exactly
   why it needs a standing regression test rather than relying on
   noticing implausible reward signs during a future live run.

These tests construct bare ShowdownBattleEnv/EncodedSinglesEnv
instances via __new__ (bypassing __init__, which would otherwise open
a real websocket connection) and manually set only the attributes each
method under test actually touches.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from environment.battle_env import EncodedSinglesEnv, ShowdownBattleEnv
from environment.rewards import RewardConfig


class _FakeWrapped:
    """Stands in for SingleAgentWrapper, returning exactly the dict
    shape poke-env's real PokeEnv.reset()/step() produce."""

    def __init__(self):
        self.reset_obs = {
            "observation": np.arange(649, dtype=np.float32),
            "action_mask": np.ones(26, dtype=np.int64),
        }
        self.step_obs = {
            "observation": np.full(649, 2.0, dtype=np.float32),
            "action_mask": np.ones(26, dtype=np.int64),
        }
        self.last_step_action = None

    def reset(self, seed=None, options=None):
        return self.reset_obs, {"reset": True}

    def step(self, action):
        self.last_step_action = action
        return self.step_obs, 1.0, False, False, {"stepped": True}


class _FakeUnderlying:
    def __init__(self):
        self.battle1 = None

    def get_action_mask(self, battle):
        return [1] * 26


def _make_bare_showdown_env() -> ShowdownBattleEnv:
    env = ShowdownBattleEnv.__new__(ShowdownBattleEnv)
    env._wrapped = _FakeWrapped()
    env._underlying = _FakeUnderlying()
    env.n_invalid_actions = 0
    env.n_total_actions = 0
    return env


def test_reset_unwraps_observation_dict_to_flat_array():
    env = _make_bare_showdown_env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (649,)
    assert info == {"reset": True}


def test_step_unwraps_observation_dict_to_flat_array():
    env = _make_bare_showdown_env()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (649,)
    assert reward == 1.0


def test_step_converts_action_to_numpy_scalar_before_calling_wrapped_step():
    env = _make_bare_showdown_env()
    env.step(5)  # a plain Python int, as every caller in this codebase passes
    assert isinstance(env._wrapped.last_step_action, np.integer)
    assert env._wrapped.last_step_action == 5


# --- reward-state key collision ------------------------------------------


def _fake_pokemon(hp_frac=1.0, fainted=False):
    return SimpleNamespace(current_hp_fraction=hp_frac, fainted=fainted)


def _fake_battle(tag, own_hp, opp_hp, finished=False, won=False, lost=False):
    return SimpleNamespace(
        battle_tag=tag,
        team={"a": _fake_pokemon(hp_frac=own_hp)},
        opponent_team={"x": _fake_pokemon(hp_frac=opp_hp)},
        finished=finished,
        won=won,
        lost=lost,
    )


def _make_bare_encoded_env() -> EncodedSinglesEnv:
    env = EncodedSinglesEnv.__new__(EncodedSinglesEnv)
    env._reward_config = RewardConfig()
    env._reward_states = {}
    return env


def test_calc_reward_does_not_cross_contaminate_same_tag_different_battles():
    """battle1 and battle2 share the SAME battle_tag (poke-env's real
    behavior) but are different objects representing opposite
    perspectives. Interleaved calc_reward() calls for both -- as
    happens every real step -- must not corrupt each other's
    incremental HP-delta tracking.

    Real poke-env Battle objects are mutated in place turn over turn
    (not recreated), which is exactly why keying reward state by
    id(battle) works: the same object's identity stays stable for its
    whole battle. This test mirrors that by mutating the SAME fake
    battle objects between calls rather than creating new ones -- using
    fresh objects per turn would trivially "pass" without exercising
    the actual per-battle state persistence this regresses against.
    """
    env = _make_bare_encoded_env()

    battle1 = _fake_battle("battle-tag-1", own_hp=1.0, opp_hp=1.0)
    battle2 = _fake_battle("battle-tag-1", own_hp=1.0, opp_hp=1.0)
    assert battle1.battle_tag == battle2.battle_tag  # the real-world collision condition

    # Turn 0: establish each battle's own baseline HP state. The very
    # first calc_reward() call for any battle compares against
    # BattleRewardState()'s defaults (a full 6-pokemon team), which
    # doesn't represent a real prior turn -- so this reward value is
    # thrown away; only the resulting stored state matters from here.
    env.calc_reward(battle1)
    env.calc_reward(battle2)

    # Turn 1: agent1 lands a hit (opponent HP drops from their view);
    # battle2 is the SAME real event from agent2's opposite
    # perspective (their own HP dropped by the same amount). Mutate the
    # SAME objects in place, matching real poke-env behavior.
    battle1.opponent_team["x"].current_hp_fraction = 0.5
    battle2.team["a"].current_hp_fraction = 0.5

    reward1_turn1 = env.calc_reward(battle1)
    reward2_turn1 = env.calc_reward(battle2)

    # agent1: opponent HP dropped -> positive positional reward, own HP
    # unchanged -> no damage penalty. If state were shared/overwritten
    # by battle_tag alone, agent2's calc_reward call (computed second)
    # would compare against agent1's just-written prev_hp values
    # instead of its own, producing a nonsensical result here.
    assert reward1_turn1 > 0
    # agent2: own HP dropped -> damage penalty; opponent (unchanged
    # from their view) contributes nothing positive.
    assert reward2_turn1 < 0

    # Turn 2: no further HP change for either side -- deltas should be
    # exactly zero for both now, proving each kept its OWN
    # previous-HP baseline rather than inheriting the other's.
    reward1_turn2 = env.calc_reward(battle1)
    reward2_turn2 = env.calc_reward(battle2)

    assert reward1_turn2 == 0.0
    assert reward2_turn2 == 0.0


def test_calc_reward_cleans_up_state_on_finish_independently_per_battle():
    env = _make_bare_encoded_env()

    battle1 = _fake_battle("battle-tag-2", own_hp=1.0, opp_hp=0.0, finished=True, won=True)
    battle2 = _fake_battle("battle-tag-2", own_hp=0.0, opp_hp=1.0, finished=True, lost=True)

    env.calc_reward(battle1)
    assert id(battle1) not in env._reward_states

    env.calc_reward(battle2)
    assert id(battle2) not in env._reward_states