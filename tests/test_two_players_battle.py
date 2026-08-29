"""
Tests for scripts/two_players_battle.py.

This script had no test coverage at all before -- including the
--learn mode added on top of it. Everything here avoids ever opening a
real network connection:
  - _make_scripted_player/_make_player are tested against fake player
    classes (constructing a real poke-env Player/CheckpointPlayer can
    itself attempt a background connection -- see the note in
    tests/test_pooled_self_play_trainer.py's module docstring for why
    other tests in this project avoid it too).
  - _run_learn is tested against a fake SelfPlayTrainer, matching the
    pattern in tests/test_self_play_train_script.py.
  - main()'s dispatch to _run_learn vs. the scripted/checkpoint
    asyncio path is tested by monkeypatching both branches so neither
    a real battle nor a real training run ever executes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import two_players_battle as tpb


class _FakePlayer:
    """Stands in for a real poke-env Player/CheckpointPlayer without
    ever constructing one -- these can attempt a background connection
    just by being instantiated."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.username = kwargs.get("account_configuration").username if kwargs.get(
            "account_configuration"
        ) else "fake"


# --- _make_scripted_player / _make_player --------------------------------


def test_make_scripted_player_selects_the_requested_class(monkeypatch):
    fake_classes = {"random": _FakePlayer, "max_base_power": _FakePlayer, "heuristic": _FakePlayer}
    monkeypatch.setattr(tpb, "SCRIPTED_PLAYER_CLASSES", fake_classes)

    player = tpb._make_scripted_player("heuristic", "gen9ou", "TestUser")

    assert isinstance(player, _FakePlayer)
    assert player.kwargs["battle_format"] == "gen9ou"
    assert player.kwargs["account_configuration"].username == "TestUser"
    assert player.kwargs["server_configuration"] is tpb.LOCAL_SERVER_CONFIGURATION


def test_make_player_with_checkpoint_builds_checkpoint_player(monkeypatch):
    monkeypatch.setattr(tpb, "CheckpointPlayer", _FakePlayer)

    player = tpb._make_player(
        kind=None,
        checkpoint="runs/some/checkpoint_step10.pt",
        battle_format="gen9randombattle",
        username="Real Human Being",
        device="cpu",
    )

    assert isinstance(player, _FakePlayer)
    assert player.kwargs["checkpoint_path"] == "runs/some/checkpoint_step10.pt"
    assert player.kwargs["device"] == "cpu"
    assert player.kwargs["account_configuration"].username == "Real Human Being"


def test_make_player_without_checkpoint_falls_back_to_scripted(monkeypatch):
    fake_classes = {"random": _FakePlayer, "heuristic": _FakePlayer}
    monkeypatch.setattr(tpb, "SCRIPTED_PLAYER_CLASSES", fake_classes)

    player = tpb._make_player(
        kind="heuristic",
        checkpoint=None,
        battle_format="gen9randombattle",
        username="Not A Robot",
        device="cpu",
    )

    assert isinstance(player, _FakePlayer)
    assert player.kwargs["account_configuration"].username == "Not A Robot"


def test_make_player_defaults_to_random_when_kind_is_none(monkeypatch):
    calls = []

    def fake_make_scripted_player(kind, battle_format, username):
        calls.append(kind)
        return _FakePlayer()

    monkeypatch.setattr(tpb, "_make_scripted_player", fake_make_scripted_player)

    tpb._make_player(kind=None, checkpoint=None, battle_format="gen9randombattle", username="X", device="cpu")

    assert calls == ["random"]


def test_checkpoint_takes_priority_over_scripted_kind(monkeypatch):
    """If both --player1 and --player1-checkpoint are somehow set, the
    checkpoint must win -- matching the docstring's documented
    precedence ('ignored if --playerN-checkpoint is set')."""
    monkeypatch.setattr(tpb, "CheckpointPlayer", _FakePlayer)

    scripted_calls = []
    monkeypatch.setattr(
        tpb, "_make_scripted_player", lambda *a, **k: scripted_calls.append(a) or _FakePlayer()
    )

    player = tpb._make_player(
        kind="random",
        checkpoint="some_checkpoint.pt",
        battle_format="gen9randombattle",
        username="X",
        device="cpu",
    )

    assert isinstance(player, _FakePlayer)
    assert player.kwargs["checkpoint_path"] == "some_checkpoint.pt"
    assert scripted_calls == []


# --- _summarize_self_play_run --------------------------------------------


def _write_metrics(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_summarize_self_play_run_counts_episodes_and_wins(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {"event": "episode_end", "player1_won": True, "player2_won": False},
            {"event": "episode_end", "player1_won": False, "player2_won": True},
            {"event": "episode_end", "player1_won": True, "player2_won": False},
            {"event": "train_progress", "avg_loss": 0.1},  # non-episode lines are ignored
        ],
    )

    summary = tpb._summarize_self_play_run(metrics_path)

    assert summary["episodes"] == 3
    assert summary["player1_wins"] == 2
    assert summary["player2_wins"] == 1
    assert summary["player1_win_rate"] == 2 / 3
    assert summary["player2_win_rate"] == 1 / 3


def test_summarize_self_play_run_handles_zero_episodes_without_dividing_by_zero(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path, [{"event": "train_progress", "avg_loss": 0.1}])

    summary = tpb._summarize_self_play_run(metrics_path)

    assert summary["episodes"] == 0
    assert summary["player1_win_rate"] == 0.0
    assert summary["player2_win_rate"] == 0.0


def test_summarize_self_play_run_treats_missing_won_fields_as_false(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path, [{"event": "episode_end"}])  # e.g. a draw with no won keys

    summary = tpb._summarize_self_play_run(metrics_path)

    assert summary["episodes"] == 1
    assert summary["player1_wins"] == 0
    assert summary["player2_wins"] == 0


# --- _run_learn ------------------------------------------------------------


class _FakeSelfPlayTrainer:
    captured_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeSelfPlayTrainer.captured_kwargs = kwargs
        self.player1_username = kwargs["player1_username"]
        self.player2_username = kwargs["player2_username"]
        self.run_dir = Path(kwargs["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log_path = self.run_dir / "metrics.jsonl"
        _write_metrics(
            self.metrics_log_path,
            [
                {"event": "episode_end", "player1_won": True, "player2_won": False},
                {"event": "episode_end", "player1_won": False, "player2_won": True},
            ],
        )
        self.env = self
        self.closed = False

    def train(self):
        return self.run_dir / "checkpoint_final.pt"

    def close(self):
        self.closed = True


def _learn_args(tmp_path, **overrides) -> argparse.Namespace:
    defaults = dict(
        battle_format="gen9randombattle",
        total_steps=40_000,
        device="cpu",
        run_dir=str(tmp_path),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_learn_builds_trainer_with_expected_config(tmp_path, monkeypatch):
    _FakeSelfPlayTrainer.captured_kwargs = {}
    monkeypatch.setattr(tpb, "SelfPlayTrainer", _FakeSelfPlayTrainer)

    tpb._run_learn(_learn_args(tmp_path, total_steps=123, device="cpu", battle_format="gen9ou"))

    kwargs = _FakeSelfPlayTrainer.captured_kwargs
    assert kwargs["training_config"].total_steps == 123
    assert kwargs["training_config"].device == "cpu"
    assert kwargs["battle_format"] == "gen9ou"
    assert kwargs["player1_username"] == "Real Human Being"
    assert kwargs["player2_username"] == "Not A Robot"


def test_run_learn_closes_env_and_prints_win_rate_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tpb, "SelfPlayTrainer", _FakeSelfPlayTrainer)

    tpb._run_learn(_learn_args(tmp_path))

    captured = capsys.readouterr()
    assert "Self-play learning complete over 2 episode(s)." in captured.out
    assert "Real Human Being won 1/2" in captured.out
    assert "Not A Robot won 1/2" in captured.out
    assert "Final checkpoint:" in captured.out


def test_run_learn_closes_env_even_if_train_raises(tmp_path, monkeypatch):
    class _RaisingTrainer(_FakeSelfPlayTrainer):
        def train(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(tpb, "SelfPlayTrainer", _RaisingTrainer)

    created = {}

    original_init = _RaisingTrainer.__init__

    def capturing_init(self, **kwargs):
        original_init(self, **kwargs)
        created["trainer"] = self

    monkeypatch.setattr(_RaisingTrainer, "__init__", capturing_init)

    try:
        tpb._run_learn(_learn_args(tmp_path))
    except RuntimeError:
        pass

    assert created["trainer"].closed is True


def test_main_dispatches_to_learn_mode_and_skips_asyncio_path(monkeypatch):
    learn_calls = []
    monkeypatch.setattr(tpb, "_run_learn", lambda args: learn_calls.append(args))

    def _must_not_run(coro):
        coro.close()  # avoid "coroutine was never awaited" warning
        raise AssertionError("asyncio.run must not be called in --learn mode")

    monkeypatch.setattr(tpb.asyncio, "run", _must_not_run)

    monkeypatch.setattr(sys, "argv", ["two_players_battle.py", "--learn"])
    tpb.main()

    assert len(learn_calls) == 1
    assert learn_calls[0].learn is True


def test_main_dispatches_to_scripted_battle_when_learn_flag_absent(monkeypatch):
    def _must_not_learn(args):
        raise AssertionError("_run_learn must not be called without --learn")

    monkeypatch.setattr(tpb, "_run_learn", _must_not_learn)

    run_calls = []

    def fake_asyncio_run(coro):
        run_calls.append(coro)
        coro.close()  # never actually execute _run()'s body

    monkeypatch.setattr(tpb.asyncio, "run", fake_asyncio_run)

    monkeypatch.setattr(sys, "argv", ["two_players_battle.py"])
    tpb.main()

    assert len(run_calls) == 1