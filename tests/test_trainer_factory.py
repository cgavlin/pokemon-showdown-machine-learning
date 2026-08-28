"""
Tests for training/trainer_factory.py's build_trainer: verifies that
self_play.enabled/self_play.mode in a config dict actually determine
which trainer class gets built, since before this existed those config
fields were purely documentation that scripts/train.py never read.

Also verifies build_trainer's `init_checkpoint` param -- used by
training/curriculum_runner.py to warm-start each curriculum stage from
the previous stage's checkpoint -- is threaded through to whichever
trainer class gets built, and that a bad path fails fast, before any
opponent/env is constructed (which would otherwise open a real
connection).

These only check *which class* gets built and with *what arguments* --
constructing a real ShowdownBattleEnv-backed trainer would need a live
server, which is exercised manually, not here (see the module-level
note in tests/test_pooled_self_play_trainer.py for why: opponent
construction and the eventual env both open real connections).
"""

from __future__ import annotations

import pytest
import torch

from agents.policy import DuelingQNetwork
from environment.state import observation_size
from training.curriculum import get_stage
from training.pooled_self_play_trainer import PooledSelfPlayTrainer
from training.self_play_trainer import SelfPlayTrainer
from training.trainer import Trainer
from training.trainer_factory import build_trainer


def _write_fake_checkpoint(path, n_actions: int = 26) -> None:
    """A real, loadable checkpoint in this project's format (matching
    what Trainer/SelfPlayTrainer/PooledSelfPlayTrainer's own
    _save_checkpoint produces) -- needed because these trainers'
    init_checkpoint handling calls torch.load for real, unlike the
    lighter FileNotFoundError-only check in build_trainer itself."""
    net = DuelingQNetwork(observation_size(), n_actions)
    torch.save({"q_network": net.state_dict(), "metadata": {}}, path)

BASE_CONFIG = {
    "reward": {},
    "training": {"total_steps": 10, "device": "cpu"},
}


def test_no_self_play_block_builds_plain_trainer(tmp_path, monkeypatch):
    # Trainer's __init__ needs a real env; patch make_env so this stays
    # offline. We only care that Trainer (not a self-play trainer) is
    # the one that gets constructed.
    import training.trainer_factory as factory_module

    class _FakeOpponent:
        format = "gen9randombattle"

    class _FakeEnv:
        action_space = type("Sp", (), {"n": 26})()
        opponent = _FakeOpponent()

        def reset(self, seed=None):
            return None, {}

        def get_action_mask(self):
            return [1] * 26

        def close(self):
            pass

    monkeypatch.setattr(factory_module, "make_env", lambda **kwargs: _FakeEnv())
    monkeypatch.setattr(factory_module, "make_opponent", lambda stage, team=None: object())

    stage = get_stage("stage1_basic_mechanics")
    trainer = build_trainer(dict(BASE_CONFIG), stage, run_dir=tmp_path)
    assert isinstance(trainer, Trainer)


def test_self_play_disabled_explicitly_builds_plain_trainer(tmp_path, monkeypatch):
    import training.trainer_factory as factory_module

    class _FakeOpponent:
        format = "gen9randombattle"

    class _FakeEnv:
        action_space = type("Sp", (), {"n": 26})()
        opponent = _FakeOpponent()

        def reset(self, seed=None):
            return None, {}

        def get_action_mask(self):
            return [1] * 26

        def close(self):
            pass

    monkeypatch.setattr(factory_module, "make_env", lambda **kwargs: _FakeEnv())
    monkeypatch.setattr(factory_module, "make_opponent", lambda stage, team=None: object())

    config = dict(BASE_CONFIG, self_play={"enabled": False})
    stage = get_stage("stage1_basic_mechanics")
    trainer = build_trainer(config, stage, run_dir=tmp_path)
    assert isinstance(trainer, Trainer)


def test_self_play_pooled_mode_builds_pooled_trainer(tmp_path, monkeypatch):
    import training.trainer_factory as factory_module
    import training.pooled_self_play_trainer as pooled_module

    # PooledSelfPlayTrainer's __init__ itself opens a real opponent +
    # env unless we inject env_factory -- but build_trainer doesn't
    # expose that hook (production callers always want the real thing).
    # So here we patch its own OPPONENT_FACTORIES to an inert stand-in
    # and its module-level make_env to an offline fake, mirroring how
    # tests/test_pooled_self_play_trainer.py avoids real connections.
    class _FakeOpponent:
        format = "gen9randombattle"

    class _FakeEnv:
        action_space = type("Sp", (), {"n": 26})()
        opponent = _FakeOpponent()

        def reset(self, seed=None):
            return None, {}

        def get_action_mask(self):
            return [1] * 26

        def close(self):
            pass

    monkeypatch.setattr(pooled_module, "OPPONENT_FACTORIES", {"heuristic": lambda fmt, team: _FakeOpponent()})
    monkeypatch.setattr(pooled_module, "make_env", lambda **kwargs: _FakeEnv())

    config = dict(
        BASE_CONFIG,
        self_play={"enabled": True, "mode": "pooled", "episodes_per_opponent": 5, "max_pool_size": 10},
    )
    stage = get_stage("stage4_competitive_play")
    trainer = build_trainer(config, stage, run_dir=tmp_path)

    assert isinstance(trainer, PooledSelfPlayTrainer)
    assert trainer.episodes_per_opponent == 5
    # No bootstrap_opponent override given -> should default to the
    # stage's own opponent_factory ("heuristic" for stage 4).
    assert trainer.bootstrap_opponent == "heuristic"


def test_self_play_bootstrap_opponent_override_is_respected(tmp_path, monkeypatch):
    import training.pooled_self_play_trainer as pooled_module

    class _FakeOpponent:
        format = "gen9randombattle"

    class _FakeEnv:
        action_space = type("Sp", (), {"n": 26})()
        opponent = _FakeOpponent()

        def reset(self, seed=None):
            return None, {}

        def get_action_mask(self):
            return [1] * 26

        def close(self):
            pass

    monkeypatch.setattr(
        pooled_module,
        "OPPONENT_FACTORIES",
        {"random": lambda fmt, team: _FakeOpponent(), "heuristic": lambda fmt, team: _FakeOpponent()},
    )
    monkeypatch.setattr(pooled_module, "make_env", lambda **kwargs: _FakeEnv())

    config = dict(
        BASE_CONFIG,
        self_play={"enabled": True, "mode": "pooled", "bootstrap_opponent": "random"},
    )
    stage = get_stage("stage4_competitive_play")  # stage default would be "heuristic"
    trainer = build_trainer(config, stage, run_dir=tmp_path)
    assert trainer.bootstrap_opponent == "random"


def test_self_play_mirror_mode_builds_self_play_trainer(tmp_path, monkeypatch):
    import training.self_play_trainer as mirror_module

    class _FakeEnv:
        possible_agents = ["Player1 A", "Player2 B"]
        agents = list(possible_agents)
        action_spaces = {a: type("Sp", (), {"n": 26})() for a in possible_agents}

        def reset(self, seed=None):
            return {}, {}

        def close(self):
            pass

    monkeypatch.setattr(mirror_module, "EncodedSinglesEnv", lambda **kwargs: _FakeEnv())

    config = dict(BASE_CONFIG, self_play={"enabled": True, "mode": "mirror"})
    stage = get_stage("stage4_competitive_play")
    trainer = build_trainer(config, stage, run_dir=tmp_path)
    assert isinstance(trainer, SelfPlayTrainer)


def test_unknown_self_play_mode_raises(tmp_path):
    config = dict(BASE_CONFIG, self_play={"enabled": True, "mode": "not_a_real_mode"})
    stage = get_stage("stage4_competitive_play")
    with pytest.raises(ValueError, match="not_a_real_mode"):
        build_trainer(config, stage, run_dir=tmp_path)


def test_team_pool_dir_builds_a_team_pool_and_threads_it_through(tmp_path):
    import training.trainer_factory as factory_module

    team_dir = tmp_path / "teams"
    team_dir.mkdir()
    (team_dir / "team1.txt").write_text(
        "Landorus-Therian @ Choice Scarf\n"
        "Ability: Intimidate\n"
        "Tera Type: Flying\n"
        "EVs: 252 Atk / 4 SpD / 252 Spe\n"
        "Jolly Nature\n"
        "- Earthquake\n"
        "- U-turn\n"
        "- Stone Edge\n"
        "- Stealth Rock\n"
    )

    class _FakeOpponent:
        format = "gen9randombattle"

    class _FakeEnv:
        action_space = type("Sp", (), {"n": 26})()
        opponent = _FakeOpponent()

        def reset(self, seed=None):
            return None, {}

        def get_action_mask(self):
            return [1] * 26

        def close(self):
            pass

    seen_opponent_team = []
    seen_env_team = []

    def fake_make_opponent(stage, team=None):
        seen_opponent_team.append(team)
        return _FakeOpponent()

    def fake_make_env(**kwargs):
        seen_env_team.append(kwargs.get("team"))
        return _FakeEnv()

    import unittest.mock

    with unittest.mock.patch.multiple(
        factory_module, make_opponent=fake_make_opponent, make_env=fake_make_env
    ):
        config = dict(BASE_CONFIG, team_pool_dir=str(team_dir))
        stage = get_stage("stage1_basic_mechanics")
        build_trainer(config, stage, run_dir=tmp_path)

    from environment.team_pool import TeamPool

    assert len(seen_opponent_team) == 1
    assert isinstance(seen_opponent_team[0], TeamPool)
    assert len(seen_env_team) == 1
    assert seen_env_team[0] is seen_opponent_team[0]  # same pool instance, both sides


def test_no_team_pool_dir_passes_team_none(tmp_path):
    import training.trainer_factory as factory_module

    class _FakeOpponent:
        format = "gen9randombattle"

    class _FakeEnv:
        action_space = type("Sp", (), {"n": 26})()
        opponent = _FakeOpponent()

        def reset(self, seed=None):
            return None, {}

        def get_action_mask(self):
            return [1] * 26

        def close(self):
            pass

    seen_teams = []

    def fake_make_opponent(stage, team=None):
        seen_teams.append(team)
        return _FakeOpponent()

    def fake_make_env(**kwargs):
        seen_teams.append(kwargs.get("team"))
        return _FakeEnv()

    import unittest.mock

    with unittest.mock.patch.multiple(
        factory_module, make_opponent=fake_make_opponent, make_env=fake_make_env
    ):
        stage = get_stage("stage1_basic_mechanics")
        build_trainer(dict(BASE_CONFIG), stage, run_dir=tmp_path)

    assert seen_teams == [None, None]


# --- init_checkpoint threading -------------------------------------------


class _FakeOpponent:
    format = "gen9randombattle"


class _FakeEnv:
    action_space = type("Sp", (), {"n": 26})()
    opponent = _FakeOpponent()

    def reset(self, seed=None):
        return None, {}

    def get_action_mask(self):
        return [1] * 26

    def close(self):
        pass


def test_missing_init_checkpoint_raises_before_any_opponent_or_env_is_built(tmp_path, monkeypatch):
    """A bad init_checkpoint path must fail immediately -- before
    make_opponent/make_env (which open real connections for the
    non-self-play path) are even called. Verified by making both raise
    if called, so the test fails loudly if the ordering ever
    regresses."""
    import training.trainer_factory as factory_module

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("make_opponent/make_env must not run for a bad init_checkpoint")

    monkeypatch.setattr(factory_module, "make_opponent", _must_not_be_called)
    monkeypatch.setattr(factory_module, "make_env", _must_not_be_called)

    stage = get_stage("stage1_basic_mechanics")
    with pytest.raises(FileNotFoundError, match="init_checkpoint not found"):
        build_trainer(
            dict(BASE_CONFIG),
            stage,
            run_dir=tmp_path,
            init_checkpoint=str(tmp_path / "missing.pt"),
        )


def test_valid_init_checkpoint_is_passed_to_plain_trainer(tmp_path, monkeypatch):
    import training.trainer_factory as factory_module

    fake_checkpoint = tmp_path / "checkpoint_step10.pt"
    _write_fake_checkpoint(fake_checkpoint)

    monkeypatch.setattr(factory_module, "make_env", lambda **kwargs: _FakeEnv())
    monkeypatch.setattr(factory_module, "make_opponent", lambda stage, team=None: _FakeOpponent())

    stage = get_stage("stage1_basic_mechanics")
    trainer = build_trainer(dict(BASE_CONFIG), stage, run_dir=tmp_path, init_checkpoint=fake_checkpoint)

    assert isinstance(trainer, Trainer)
    assert trainer.init_checkpoint == fake_checkpoint


def test_valid_init_checkpoint_is_passed_to_pooled_trainer(tmp_path, monkeypatch):
    import training.pooled_self_play_trainer as pooled_module

    fake_checkpoint = tmp_path / "checkpoint_step10.pt"
    _write_fake_checkpoint(fake_checkpoint)

    monkeypatch.setattr(pooled_module, "OPPONENT_FACTORIES", {"heuristic": lambda fmt, team: _FakeOpponent()})
    monkeypatch.setattr(pooled_module, "make_env", lambda **kwargs: _FakeEnv())

    config = dict(BASE_CONFIG, self_play={"enabled": True, "mode": "pooled"})
    stage = get_stage("stage4_competitive_play")
    trainer = build_trainer(config, stage, run_dir=tmp_path, init_checkpoint=fake_checkpoint)

    assert isinstance(trainer, PooledSelfPlayTrainer)
    assert trainer.init_checkpoint == fake_checkpoint


def test_valid_init_checkpoint_is_passed_to_mirror_trainer(tmp_path, monkeypatch):
    import training.self_play_trainer as mirror_module

    fake_checkpoint = tmp_path / "checkpoint_step10.pt"
    _write_fake_checkpoint(fake_checkpoint)

    class _FakeMirrorEnv:
        possible_agents = ["Player1 A", "Player2 B"]
        agents = list(possible_agents)
        action_spaces = {a: type("Sp", (), {"n": 26})() for a in possible_agents}

        def reset(self, seed=None):
            return {}, {}

        def close(self):
            pass

    monkeypatch.setattr(mirror_module, "EncodedSinglesEnv", lambda **kwargs: _FakeMirrorEnv())

    trainer = build_trainer(
        dict(BASE_CONFIG, self_play={"enabled": True, "mode": "mirror"}),
        get_stage("stage4_competitive_play"),
        run_dir=tmp_path,
        init_checkpoint=fake_checkpoint,
    )

    assert isinstance(trainer, SelfPlayTrainer)
    assert trainer.init_checkpoint == fake_checkpoint