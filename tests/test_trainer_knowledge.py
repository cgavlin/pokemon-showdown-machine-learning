from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from environment.rewards import RewardConfig
from environment.state import observation_size
from knowledge.pokemon_knowledge import PokemonKnowledgeBase
from training.trainer import Trainer, TrainingConfig

N_ACTIONS = 5


class _FakeOpponent:
    format = "gen9randombattle"


def _fake_type(name):
    return SimpleNamespace(name=name.upper())


def _fake_pokemon(species):
    return SimpleNamespace(
        species=species, type_1=_fake_type("normal"), type_2=None, ability=None, item=None, moves={}
    )


class _FakeUnderlying:
    """Stands in for ShowdownBattleEnv._underlying: exposes battle1,
    the same access pattern evaluation/benchmarks.py and Trainer both
    use for OUR agent's own perspective on the just-finished battle."""

    def __init__(self):
        self.battle1 = SimpleNamespace(
            team={"p1": _fake_pokemon("Charizard")},
            opponent_team={"o1": _fake_pokemon("Garchomp")},
        )


class _FakeEnv:
    """Deterministically ends an episode every 3 steps, so knowledge-base
    recording can be checked without needing a real battle."""

    def __init__(self, n_actions=N_ACTIONS, episode_len=3):
        self.action_space = type("Sp", (), {"n": n_actions})()
        self.opponent = _FakeOpponent()
        self.n_invalid_actions = 0
        self.n_total_actions = 0
        self._underlying = _FakeUnderlying()
        self._turn = 0
        self._episode_len = episode_len

    def reset(self, seed=None):
        self._turn = 0
        return np.zeros(observation_size(), dtype=np.float32), {}

    def get_action_mask(self):
        return np.ones(N_ACTIONS, dtype=np.int64)

    def step(self, action):
        self._turn += 1
        done = self._turn >= self._episode_len
        return np.zeros(observation_size(), dtype=np.float32), 1.0, done, False, {}

    def close(self):
        pass


def _make_training_config(**overrides) -> TrainingConfig:
    defaults = dict(
        total_steps=12,
        batch_size=4,
        device="cpu",
        learning_starts=1000,  # keep this test focused on recording, not the DQN update itself
        eval_every_steps=6,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_no_knowledge_base_is_a_safe_no_op(tmp_path):
    trainer = Trainer(
        env=_FakeEnv(),
        training_config=_make_training_config(),
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        curriculum_stage_name="stage1_basic_mechanics",
    )
    trainer.train()  # must not raise even though knowledge_base is None

    metadata = json.loads((trainer.run_dir / "run_metadata.json").read_text())
    assert metadata["knowledge_base_path"] is None


def test_knowledge_base_records_every_finished_battle(tmp_path):
    kb = PokemonKnowledgeBase()  # in-memory only, no path
    trainer = Trainer(
        env=_FakeEnv(episode_len=3),
        training_config=_make_training_config(total_steps=12),
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        curriculum_stage_name="stage1_basic_mechanics",
        knowledge_base=kb,
    )
    trainer.train()

    # 12 steps / 3-step episodes = 4 finished battles, each revealing
    # Charizard (own) and Garchomp (opponent).
    assert kb.species_summary("charizard") is not None
    assert kb.species_summary("charizard")["battles_seen"] == 4
    assert kb.species_summary("garchomp")["battles_seen"] == 4


def test_knowledge_base_is_saved_on_the_checkpoint_cadence(tmp_path):
    kb_path = tmp_path / "knowledge.json"
    kb = PokemonKnowledgeBase(path=kb_path)
    trainer = Trainer(
        env=_FakeEnv(episode_len=3),
        training_config=_make_training_config(total_steps=12, eval_every_steps=6),
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        curriculum_stage_name="stage1_basic_mechanics",
        knowledge_base=kb,
    )
    trainer.train()

    assert kb_path.exists()
    reloaded = PokemonKnowledgeBase(path=kb_path)
    assert reloaded.species_summary("charizard") is not None

    metadata = json.loads((trainer.run_dir / "run_metadata.json").read_text())
    assert metadata["knowledge_base_path"] == str(kb_path)


def test_in_memory_knowledge_base_without_a_path_is_never_saved_to_disk(tmp_path):
    """A knowledge_base constructed with no path is a valid, intentional
    choice (e.g. a short-lived analysis session) -- Trainer must not
    try to save it (PokemonKnowledgeBase.save() raises without a path)."""
    kb = PokemonKnowledgeBase()  # no path
    trainer = Trainer(
        env=_FakeEnv(episode_len=3),
        training_config=_make_training_config(total_steps=12, eval_every_steps=6),
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        curriculum_stage_name="stage1_basic_mechanics",
        knowledge_base=kb,
    )
    trainer.train()  # must not raise
    assert not (tmp_path / "knowledge.json").exists()