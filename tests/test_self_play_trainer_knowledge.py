from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from environment.rewards import RewardConfig
from environment.state import observation_size
from knowledge.pokemon_knowledge import PokemonKnowledgeBase
from training.self_play_trainer import SelfPlayTrainer
from training.trainer import TrainingConfig


def _fake_type(name):
    return SimpleNamespace(name=name.upper())


def _fake_pokemon(species):
    return SimpleNamespace(
        species=species, type_1=_fake_type("normal"), type_2=None, ability=None, item=None, moves={}
    )


class _FakeBattle:
    def __init__(self, won: bool, own_species: str, opp_species: str):
        self.won = won
        self.team = {"p1": _fake_pokemon(own_species)}
        self.opponent_team = {"o1": _fake_pokemon(opp_species)}


class MockParallelEnv:
    def __init__(self, n_actions: int, obs_size: int, max_turns: int = 3):
        self.possible_agents = ["Player1 A", "Player2 B"]
        self.agents = list(self.possible_agents)
        self.action_spaces = {a: type("S", (), {"n": n_actions})() for a in self.possible_agents}
        self._obs_size = obs_size
        self._n_actions = n_actions
        self._turn = 0
        self._max_turns = max_turns
        self._rng = np.random.default_rng(0)
        self.battle1 = None
        self.battle2 = None

    def _obs_dict(self):
        return {
            a: {
                "observation": self._rng.standard_normal(self._obs_size).astype(np.float32),
                "action_mask": np.ones(self._n_actions, dtype=np.int64),
            }
            for a in self.agents
        }

    def reset(self, seed=None):
        self.agents = list(self.possible_agents)
        self._turn = 0
        return self._obs_dict(), {}

    def step(self, actions):
        self._turn += 1
        done = self._turn >= self._max_turns
        obs = self._obs_dict()
        rewards = {a: 1.0 for a in self.agents}
        terminated = {a: done for a in self.agents}
        truncated = {a: False for a in self.agents}
        if done:
            # Two perspectives on the same battle: player1 sees
            # (own=Charizard, opp=Garchomp); player2 sees the mirror.
            self.battle1 = _FakeBattle(won=True, own_species="Charizard", opp_species="Garchomp")
            self.battle2 = _FakeBattle(won=False, own_species="Garchomp", opp_species="Charizard")
            self.agents = []
        return obs, rewards, terminated, truncated, {}

    def close(self):
        pass


def test_self_play_trainer_records_both_perspectives_into_knowledge_base(tmp_path):
    kb = PokemonKnowledgeBase()
    cfg = TrainingConfig(
        total_steps=9,  # 3 episodes of 3 steps each
        batch_size=4,
        replay_capacity=200,
        learning_starts=1000,
        eval_every_steps=1000,
        epsilon_decay_steps=10,
    )
    mock_env = MockParallelEnv(n_actions=26, obs_size=observation_size(), max_turns=3)
    trainer = SelfPlayTrainer(
        training_config=cfg,
        reward_config=RewardConfig(),
        run_dir=tmp_path,
        env=mock_env,
        knowledge_base=kb,
    )
    trainer.train()

    # Only battle1 is recorded (it already contains BOTH species via
    # team + opponent_team) -- recording battle2 too would double-count
    # every real battle, since it's the same event from the mirrored
    # perspective, not a second battle.
    assert kb.species_summary("charizard")["battles_seen"] == 3
    assert kb.species_summary("garchomp")["battles_seen"] == 3
    # battle2 (won=False, own=Garchomp) must NOT have contributed a
    # second recording -- if it had, battles_seen above would be 6.