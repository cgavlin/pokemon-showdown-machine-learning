from __future__ import annotations

import numpy as np
import pytest
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player

import agents.checkpoint_player as cp_module
from agents.checkpoint_player import CheckpointPlayer, _gen_from_format


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


class _FakeTrainedAgent:
    captured: dict = {}

    def __init__(self, checkpoint_path, n_actions, device="cpu"):
        _FakeTrainedAgent.captured = {
            "checkpoint_path": checkpoint_path,
            "n_actions": n_actions,
            "device": device,
        }


@pytest.fixture(autouse=True)
def _stub_player_init(monkeypatch):
    monkeypatch.setattr(Player, "__init__", lambda self, **kwargs: None)


def test_init_sizes_action_space_from_explicit_battle_format(monkeypatch):
    monkeypatch.setattr(cp_module, "TrainedAgent", _FakeTrainedAgent)
    _FakeTrainedAgent.captured = {}

    CheckpointPlayer(checkpoint_path="fake.pt", device="cpu", battle_format="gen8ou")

    assert _FakeTrainedAgent.captured["n_actions"] == SinglesEnv.get_action_space_size(8)
    assert _FakeTrainedAgent.captured["checkpoint_path"] == "fake.pt"
    assert _FakeTrainedAgent.captured["device"] == "cpu"


def test_player_stores_the_built_agent(monkeypatch):
    monkeypatch.setattr(cp_module, "TrainedAgent", _FakeTrainedAgent)
    player = CheckpointPlayer(checkpoint_path="fake.pt", battle_format="gen9randombattle")
    assert isinstance(player._agent, _FakeTrainedAgent)


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
    player = CheckpointPlayer.__new__(CheckpointPlayer)
    fake_agent = _FakeAgent(action_to_return=3)
    player._agent = fake_agent
    player._knowledge_base = None

    fake_battle = object()
    fake_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fake_mask = [1, 0, 1, 1]

    monkeypatch.setattr(cp_module, "encode_battle", lambda battle, knowledge_base=None: fake_obs)
    monkeypatch.setattr(SinglesEnv, "get_action_mask", lambda battle: fake_mask)

    order_calls = []

    def fake_action_to_order(action, battle, fake=False, strict=True):
        order_calls.append((action, battle, fake, strict))
        return "FAKE_ORDER"

    monkeypatch.setattr(SinglesEnv, "action_to_order", fake_action_to_order)

    result = player.choose_move(fake_battle)

    assert result == "FAKE_ORDER"
    assert fake_agent.last_obs is fake_obs
    np.testing.assert_array_equal(fake_agent.last_mask, np.array(fake_mask, dtype=np.int64))

    assert len(order_calls) == 1
    action_arg, battle_arg, fake_arg, strict_arg = order_calls[0]
    assert action_arg == 3
    assert battle_arg is fake_battle
    assert fake_arg is False
    assert strict_arg is False


def test_choose_move_converts_action_to_numpy_int64(monkeypatch):
    player = CheckpointPlayer.__new__(CheckpointPlayer)
    player._agent = _FakeAgent(action_to_return=7)
    player._knowledge_base = None

    monkeypatch.setattr(cp_module, "encode_battle", lambda battle, knowledge_base=None: np.zeros(3))
    monkeypatch.setattr(SinglesEnv, "get_action_mask", lambda battle: [1, 1, 1, 1, 1, 1, 1, 1])

    seen_action_type = {}

    def fake_action_to_order(action, battle, fake=False, strict=True):
        seen_action_type["type"] = type(action)
        return "ORDER"

    monkeypatch.setattr(SinglesEnv, "action_to_order", fake_action_to_order)

    player.choose_move(object())

    assert seen_action_type["type"] is np.int64