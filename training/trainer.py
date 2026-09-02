"""
DQN trainer for the battle agent.

Every run records model version, environment version, reward configuration,
training configuration, opponent pool, team pool, episode/step counts,
evaluation results, seeds, checkpoint locations, and hyperparameters
into a single run directory.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from agents.policy import DuelingQNetwork, epsilon_greedy_action
from agents.value_function import ReplayBuffer, Transition
from environment.battle_env import ShowdownBattleEnv, make_env
from environment.rewards import RewardConfig
from environment.state import observation_size
from training.dqn_update import double_dqn_update

if TYPE_CHECKING:
    from knowledge.pokemon_knowledge import PokemonKnowledgeBase

logger = logging.getLogger(__name__)

# Bumped 0.1.0 -> 0.2.0: environment/state.py's observation now includes
# a knowledge-base-derived block per opponent Pokemon slot (see
# knowledge/pokemon_knowledge.py), changing observation_size(). Any
# checkpoint saved under 0.1.0 has a mismatched input-layer shape and
# will fail to load into a fresh network built against this version.
ENVIRONMENT_VERSION = "0.2.0"  # bump whenever observation/action/reward shape changes


@dataclass
class TrainingConfig:
    total_steps: int = 200_000
    batch_size: int = 256
    replay_capacity: int = 100_000
    learning_starts: int = 5_000
    train_every: int = 4
    target_update_every: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    eval_every_steps: int = 10_000
    eval_battles: int = 200
    seed: int = 0
    device: str = "cpu"


def _epsilon_at(step: int, cfg: TrainingConfig) -> float:
    frac = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


class Trainer:
    def __init__(
        self,
        env: ShowdownBattleEnv,
        training_config: TrainingConfig,
        reward_config: RewardConfig,
        run_dir: str | Path,
        curriculum_stage_name: str,
        team_pool_description: str = "default random team pool",
        init_checkpoint: Optional[str | Path] = None,
        knowledge_base: Optional["PokemonKnowledgeBase"] = None,
    ):
        self.env = env
        self.cfg = training_config
        self.reward_config = reward_config
        self.run_id = f"{curriculum_stage_name}_{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(run_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.curriculum_stage_name = curriculum_stage_name
        self.team_pool_description = team_pool_description
        self.init_checkpoint = init_checkpoint
        # Not owned/constructed here -- trainer_factory.build_trainer
        # builds one shared instance and passes it both to make_env()
        # (so observations already reflect it) and here (so this
        # trainer can grow it as battles finish). May be None, meaning
        # the knowledge feature is simply off for this run.
        self.knowledge_base = knowledge_base

        self.device = torch.device(self.cfg.device)
        obs_size = observation_size()
        n_actions = env.action_space.n

        self.q_network = DuelingQNetwork(obs_size, n_actions).to(self.device)
        if init_checkpoint is not None:
            # Warm-start from a prior run's checkpoint instead of always
            # beginning from random initialization. In particular,
            # training/curriculum_runner.py threads the PREVIOUS
            # curriculum stage's final checkpoint through here, so each
            # stage actually builds on what the last one learned instead
            # of restarting from scratch every stage.
            state_dict = torch.load(init_checkpoint, map_location=self.device)
            if "q_network" not in state_dict:
                raise ValueError(
                    f"{init_checkpoint} loaded but doesn't look like a checkpoint saved "
                    "by this project (missing the 'q_network' key)."
                )
            self.q_network.load_state_dict(state_dict["q_network"])
            logger.info("Warm-started network from %s", init_checkpoint)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.cfg.learning_rate)

        self.buffer = ReplayBuffer(self.cfg.replay_capacity, obs_size, n_actions, seed=self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.metrics_log_path = self.run_dir / "metrics.jsonl"
        self._write_run_metadata()

    def _write_run_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "environment_version": ENVIRONMENT_VERSION,
            "curriculum_stage": self.curriculum_stage_name,
            "reward_config": asdict(self.reward_config),
            "training_config": asdict(self.cfg),
            "team_pool_description": self.team_pool_description,
            "opponent": type(self.env.opponent).__name__,
            "battle_format": getattr(self.env.opponent, "format", "unknown"),
            "init_checkpoint": str(self.init_checkpoint) if self.init_checkpoint else None,
            "knowledge_base_path": (
                str(self.knowledge_base.path)
                if self.knowledge_base is not None and self.knowledge_base.path is not None
                else None
            ),
            "seed": self.cfg.seed,
            "created_at": time.time(),
        }
        (self.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    def _log_metrics(self, **kwargs) -> None:
        kwargs["step"] = kwargs.get("step")
        kwargs["timestamp"] = time.time()
        with open(self.metrics_log_path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")

    def _train_step(self) -> float:
        batch = self.buffer.sample(self.cfg.batch_size)
        return double_dqn_update(
            self.q_network, self.target_network, self.optimizer, batch, self.device, self.cfg.gamma
        )

    def _save_checkpoint(self, step: int) -> Path:
        checkpoint_path = self.run_dir / f"checkpoint_step{step}.pt"
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "metadata": {
                    "step": step,
                    "run_id": self.run_id,
                    "environment_version": ENVIRONMENT_VERSION,
                },
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _record_finished_battle_in_knowledge_base(self) -> None:
        """Called once a battle finishes -- see the `if done:` block in
        train(). Reads the just-finished battle straight off the env
        (same access pattern evaluation/benchmarks.py already uses:
        `env._underlying.battle1` is OUR agent's perspective) and folds
        it into the knowledge base, if one is configured. A no-op when
        knowledge_base is None, so this is always safe to call."""
        if self.knowledge_base is None:
            return
        underlying = getattr(self.env, "_underlying", None)
        battle = getattr(underlying, "battle1", None) if underlying is not None else None
        if battle is not None:
            self.knowledge_base.observe_battle(battle)

    def train(self) -> Path:
        obs, info = self.env.reset(seed=self.cfg.seed)
        episode_reward = 0.0
        episode_count = 0
        losses: list[float] = []

        for step in range(1, self.cfg.total_steps + 1):
            epsilon = _epsilon_at(step, self.cfg)
            action_mask = self.env.get_action_mask()
            action = epsilon_greedy_action(
                self.q_network, obs, action_mask, epsilon, self.device, self.rng
            )

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_mask = self.env.get_action_mask() if not done else np.ones_like(action_mask)

            self.buffer.add(
                Transition(obs, action, reward, next_obs, action_mask, next_mask, done)
            )
            episode_reward += reward
            obs = next_obs

            if done:
                episode_count += 1
                self._record_finished_battle_in_knowledge_base()
                self._log_metrics(
                    step=step,
                    event="episode_end",
                    episode=episode_count,
                    episode_reward=episode_reward,
                    epsilon=epsilon,
                    invalid_action_rate=self.env.n_invalid_actions / max(1, self.env.n_total_actions),
                )
                obs, info = self.env.reset()
                episode_reward = 0.0

            if step >= self.cfg.learning_starts and step % self.cfg.train_every == 0:
                if len(self.buffer) >= self.cfg.batch_size:
                    losses.append(self._train_step())

            if step % self.cfg.target_update_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            if step % self.cfg.eval_every_steps == 0:
                avg_loss = float(np.mean(losses)) if losses else None
                self._log_metrics(step=step, event="train_progress", avg_loss=avg_loss)
                losses = []
                self._save_checkpoint(step)
                # Same cadence as checkpointing: persist the knowledge
                # base's accumulated learning to disk periodically,
                # not every single battle (that would mean an I/O
                # round-trip every few seconds during training).
                if self.knowledge_base is not None and self.knowledge_base.path is not None:
                    self.knowledge_base.save()

        final_checkpoint = self._save_checkpoint(self.cfg.total_steps)
        if self.knowledge_base is not None and self.knowledge_base.path is not None:
            self.knowledge_base.save()
        logger.info("Training complete. Final checkpoint: %s", final_checkpoint)
        return final_checkpoint