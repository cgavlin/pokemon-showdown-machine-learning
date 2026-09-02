from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from environment.rewards import RewardConfig
from environment.state import observation_size
from knowledge.pokemon_knowledge import PokemonKnowledgeBase
from training import pooled_self_play_trainer as pspt
from training.trainer import TrainingConfig

N_ACTIONS = 26


class _FakeOpponent:
    pass


def _fake_opponent_factory(battle_format, team):
    return _FakeOpponent()


def _fake_type(name):
    return SimpleNamespace(name=name.upper())


def _fake_pokemon(species):
    return SimpleNamespace(
        species=species, type_1=_fake_type("normal"), type_2=None, ability=None, item=None, moves={}
    )


class _FakeBattle:
    def __init__(self, won: bool):
        self.won = won
        self.lost = not won
        self.team = {"p1": _fake_pokemon("Charizard")}
        self.opponent_team = {"o1": _fake_pokemon("Garchomp")}
        self.available_moves = []
        self.opponent_active_pokemon = None


class _FakeUnderlying:
    def __init__(self):
        self.battle1 = _FakeBattle(won=False)


class MockSingleAgentEnv:
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


def test_pooled_trainer_records_finished_battles_into_knowledge_base(tmp_path, monkeypatch):
    monkeypatch.setattr(pspt, "OPPONENT_FACTORIES", {"random": _fake_opponent_factory})
    kb = PokemonKnowledgeBase()

    created_envs = []

    def env_factory(opponent):
        env = MockSingleAgentEnv(opponent, episode_len=3, win_every_n=2)
        created_envs.append(env)
        return env

    cfg = TrainingConfig(
        total_steps=15,
        batch_size=4,
        replay_capacity=200,
        learning_starts=1000,
        eval_every_steps=1000,
        epsilon_decay_steps=10,
    )
    trainer = pspt.PooledSelfPlayTrainer(
        training_config=cfg,
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        episodes_per_opponent=100,  # avoid opponent swaps complicating the count
        env_factory=env_factory,
        knowledge_base=kb,
    )
    trainer.train()

    # 15 steps / 3-step episodes = 5 finished battles.
    assert kb.species_summary("charizard")["battles_seen"] == 5
    assert kb.species_summary("garchomp")["battles_seen"] == 5


def test_pooled_trainer_saves_knowledge_base_on_checkpoint_cadence(tmp_path, monkeypatch):
    monkeypatch.setattr(pspt, "OPPONENT_FACTORIES", {"random": _fake_opponent_factory})
    kb_path = tmp_path / "knowledge.json"
    kb = PokemonKnowledgeBase(path=kb_path)

    def env_factory(opponent):
        return MockSingleAgentEnv(opponent, episode_len=3, win_every_n=2)

    cfg = TrainingConfig(
        total_steps=15,
        batch_size=4,
        replay_capacity=200,
        learning_starts=1000,
        eval_every_steps=10,
        epsilon_decay_steps=10,
    )
    trainer = pspt.PooledSelfPlayTrainer(
        training_config=cfg,
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        episodes_per_opponent=100,
        env_factory=env_factory,
        knowledge_base=kb,
    )
    trainer.train()

    assert kb_path.exists()
    reloaded = PokemonKnowledgeBase(path=kb_path)
    assert reloaded.species_summary("charizard") is not None