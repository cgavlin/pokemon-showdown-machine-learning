"""
Direct unit tests for agents/checkpoint_player.py.

Previously only exercised indirectly (via
training/pooled_self_play_trainer.py's tests, which never construct a
real CheckpointPlayer either -- they use OPPONENT_FACTORIES
monkeypatching instead). Constructing a real poke-env Player attempts
to start listening on a websocket in Player.__init__ (start_listening
defaults to True), so:

  - CheckpointPlayer.__init__ is tested with poke_env.player.Player's
    own __init__ monkeypatched to a no-op, and agents/inference.py's
    TrainedAgent monkeypatched to a lightweight fake -- so no real
    checkpoint file, network connection, or torch.load is needed to
    verify the constructor's own logic: parsing the generation out of
    battle_format and sizing the action space accordingly.
  - choose_move() is tested against a bare instance built via __new__
    (bypassing __init__ entirely), matching the pattern
    tests/test_battle_env_live.py already uses for the same reason.
"""

from __future__ import annotations

import numpy as np
import pytest
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player

import agents.checkpoint_player as cp_module
from agents.checkpoint_player import CheckpointPlayer, _gen_from_format

# --- _gen_from_format --------------------------------------------------


@pytest.mark.parametrize(
    "battle_format,expected_gen",
    [
        ("gen9randombattle", 9),
        ("gen9ou", 9),
        ("gen8ou", 8),
        ("gen1ou", 1),
    ],
)
def test_gen_from_format_parses_the_generation_number(battle_format, expected_gen):
    assert _gen_from_format(battle_format) == expected_gen


@pytest.mark.parametrize("battle_format", ["randombattle", "", "ou", "notagenatall"])
def test_gen_from_format_defaults_to_9_when_unparseable(battle_format):
    assert _gen_from_format(battle_format) == 9


def test_gen_from_format_handles_future_multi_digit_generations():
    assert _gen_from_format("gen10ou") == 10


# --- __init__: gen parsing feeds the right action-space size -----------


class _FakeTrainedAgent:
    """Records the args CheckpointPlayer.__init__ builds it with,
    instead of actually loading a checkpoint from disk."""

    captured: dict = {}

    def __init__(self, checkpoint_path, n_actions, device="cpu"):
        _FakeTrainedAgent.captured = {
            "checkpoint_path": checkpoint_path,
            "n_actions": n_actions,
            "device": device,
        }


@pytest.fixture(autouse=True)
def _stub_player_init(monkeypatch):
    """Player.__init__ (via start_listening=True) tries to start a
    websocket listener -- stub it out so constructing a CheckpointPlayer
    in these tests never touches the network. super().__init__(**kwargs)
    resolves this dynamically at call time, so patching the class
    attribute here is picked up correctly."""
    monkeypatch.setattr(Player, "__init__", lambda self, **kwargs: None)


def test_init_sizes_action_space_from_explicit_battle_format(monkeypatch):
    monkeypatch.setattr(cp_module, "TrainedAgent", _FakeTrainedAgent)
    _FakeTrainedAgent.captured = {}

    CheckpointPlayer(checkpoint_path="fake.pt", device="cpu", battle_format="gen8ou")

    assert _FakeTrainedAgent.captured["n_actions"] == SinglesEnv.get_action_space_size(8)
    assert _FakeTrainedAgent.captured["checkpoint_path"] == "fake.pt"
    assert _FakeTrainedAgent.captured["device"] == "cpu"


def test_init_defaults_to_gen9_action_space_when_no_battle_format_given(monkeypatch):
    monkeypatch.setattr(cp_module, "TrainedAgent", _FakeTrainedAgent)
    _FakeTrainedAgent.captured = {}

    CheckpointPlayer(checkpoint_path="fake.pt")

    assert _FakeTrainedAgent.captured["n_actions"] == SinglesEnv.get_action_space_size(9)


def test_init_passes_through_a_non_default_device(monkeypatch):
    monkeypatch.setattr(cp_module, "TrainedAgent", _FakeTrainedAgent)
    _FakeTrainedAgent.captured = {}

    CheckpointPlayer(checkpoint_path="fake.pt", device="cuda", battle_format="gen9ou")

    assert _FakeTrainedAgent.captured["device"] == "cuda"


def test_player_stores_the_built_agent(monkeypatch):
    monkeypatch.setattr(cp_module, "TrainedAgent", _FakeTrainedAgent)

    player = CheckpointPlayer(checkpoint_path="fake.pt", battle_format="gen9randombattle")

    assert isinstance(player._agent, _FakeTrainedAgent)


# --- choose_move(): wiring from battle -> obs/mask -> agent -> order ----


class _FakeAgent:
    def __init__(self, action_to_return: int):
        self.action_to_return = action_to_return
        self.last_obs = None
        self.last_mask = None

    def act(self, obs, action_mask):
        self.last_obs = obs
        self.last_mask = action_mask
        return self.action_to_return


def test_choose_move_feeds_agent_action_into_action_to_order(monkeypatch):
    # Bypass __init__ entirely (matching test_battle_env_live.py's
    # pattern) since choose_move() only touches self._agent.
    player = CheckpointPlayer.__new__(CheckpointPlayer)
    fake_agent = _FakeAgent(action_to_return=3)
    player._agent = fake_agent

    fake_battle = object()
    fake_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fake_mask = [1, 0, 1, 1]

    monkeypatch.setattr(cp_module, "encode_battle", lambda battle: fake_obs)
    monkeypatch.setattr(SinglesEnv, "get_action_mask", lambda battle: fake_mask)

    order_calls = []

    def fake_action_to_order(action, battle, fake=False, strict=True):
        order_calls.append((action, battle, fake, strict))
        return "FAKE_ORDER"

    monkeypatch.setattr(SinglesEnv, "action_to_order", fake_action_to_order)

    result = player.choose_move(fake_battle)

    assert result == "FAKE_ORDER"
    # The agent must see the freshly encoded obs and the real action
    # mask straight from poke-env -- not something re-derived or stale.
    assert fake_agent.last_obs is fake_obs
    np.testing.assert_array_equal(fake_agent.last_mask, np.array(fake_mask, dtype=np.int64))

    assert len(order_calls) == 1
    action_arg, battle_arg, fake_arg, strict_arg = order_calls[0]
    assert action_arg == 3
    assert battle_arg is fake_battle
    # fake=False, strict=False: an illegal action from the agent should
    # never crash the battle -- see environment/actions.py's own
    # sanitize_action for the equivalent guarantee in the Gym env path.
    assert fake_arg is False
    assert strict_arg is False


def test_choose_move_converts_action_to_numpy_int64(monkeypatch):
    """poke-env's real action_to_order calls .item() on the action --
    plain Python ints don't have that method (see
    environment/actions.py's as_poke_env_action and the regression test
    in tests/test_battle_env_live.py for the same underlying bug)."""
    player = CheckpointPlayer.__new__(CheckpointPlayer)
    player._agent = _FakeAgent(action_to_return=7)  # a plain Python int

    monkeypatch.setattr(cp_module, "encode_battle", lambda battle: np.zeros(3))
    monkeypatch.setattr(SinglesEnv, "get_action_mask", lambda battle: [1, 1, 1, 1, 1, 1, 1, 1])

    seen_action_type = {}

    def fake_action_to_order(action, battle, fake=False, strict=True):
        seen_action_type["type"] = type(action)
        return "ORDER"

    monkeypatch.setattr(SinglesEnv, "action_to_order", fake_action_to_order)

    player.choose_move(object())

    assert seen_action_type["type"] is np.int64