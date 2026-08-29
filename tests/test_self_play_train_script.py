"""
Tests for scripts/self_play_train.py's CLI wiring.

SelfPlayTrainer itself already validates/loads init_checkpoint (see
tests/test_self_play_trainer.py) and opens a real websocket connection
when constructed for real -- so this only checks that the script's
argparse layer actually reads --init-checkpoint (and the other CLI
flags) and passes them through to SelfPlayTrainer, by monkeypatching
SelfPlayTrainer with a lightweight fake that records its constructor
kwargs instead of connecting to anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import self_play_train as self_play_train_script


class _FakeSelfPlayTrainer:
    captured_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeSelfPlayTrainer.captured_kwargs = kwargs
        self.env = self

    def train(self):
        return Path("fake_checkpoint.pt")

    def close(self):
        pass


def _write_config(tmp_path) -> Path:
    config = {
        "battle_format": "gen9randombattle",
        "reward": {},
        "training": {"total_steps": 10, "device": "cpu"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def test_init_checkpoint_flag_is_passed_through_to_self_play_trainer(tmp_path, monkeypatch):
    _FakeSelfPlayTrainer.captured_kwargs = {}
    monkeypatch.setattr(self_play_train_script, "SelfPlayTrainer", _FakeSelfPlayTrainer)

    config_path = _write_config(tmp_path)
    fake_checkpoint = str(tmp_path / "warm_start.pt")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "self_play_train.py",
            "--config",
            str(config_path),
            "--run-dir",
            str(tmp_path),
            "--init-checkpoint",
            fake_checkpoint,
        ],
    )

    self_play_train_script.main()

    assert _FakeSelfPlayTrainer.captured_kwargs["init_checkpoint"] == fake_checkpoint


def test_no_init_checkpoint_flag_passes_none(tmp_path, monkeypatch):
    _FakeSelfPlayTrainer.captured_kwargs = {}
    monkeypatch.setattr(self_play_train_script, "SelfPlayTrainer", _FakeSelfPlayTrainer)

    config_path = _write_config(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["self_play_train.py", "--config", str(config_path), "--run-dir", str(tmp_path)],
    )

    self_play_train_script.main()

    assert _FakeSelfPlayTrainer.captured_kwargs["init_checkpoint"] is None


def test_player_usernames_default_and_override(tmp_path, monkeypatch):
    _FakeSelfPlayTrainer.captured_kwargs = {}
    monkeypatch.setattr(self_play_train_script, "SelfPlayTrainer", _FakeSelfPlayTrainer)

    config_path = _write_config(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "self_play_train.py",
            "--config",
            str(config_path),
            "--run-dir",
            str(tmp_path),
            "--player1-username",
            "Alice",
            "--player2-username",
            "Bob",
        ],
    )

    self_play_train_script.main()

    assert _FakeSelfPlayTrainer.captured_kwargs["player1_username"] == "Alice"
    assert _FakeSelfPlayTrainer.captured_kwargs["player2_username"] == "Bob"