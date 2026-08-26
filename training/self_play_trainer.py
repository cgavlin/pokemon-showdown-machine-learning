"""
Self-play trainer: two explicit Player profiles battle each other on a
local Showdown server, and the program learns from both sides of every
battle.

Unlike training/trainer.py (which pits our policy against a fixed,
*non-learning* opponent Player via SingleAgentWrapper), this module
talks directly to poke-env's two-agent `EncodedSinglesEnv`. Both
players -- given their own explicit, distinct Showdown accounts --
share the SAME Dueling DQN weights (mirror self-play): every battle
produces two independent trajectories, one from each player's point of
view, and both get pushed into the replay buffer and used to update the
shared network. This roughly doubles the experience gathered per
battle and is the standard bootstrap for self-play before layering in a pool 
of past checkpoints (training/self_play.py's SelfPlayPool) for opponent diversity.

Both players always connect to the LOCAL server -- never the public ladder.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from poke_env.ps_client import AccountConfiguration

from agents.policy import DuelingQNetwork, epsilon_greedy_action
from agents.value_function import ReplayBuffer, Transition
from environment.actions import as_poke_env_action
from environment.battle_env import LOCAL_SERVER_CONFIGURATION, EncodedSinglesEnv
from environment.rewards import RewardConfig
from environment.state import observation_size
from training.dqn_update import double_dqn_update
from training.trainer import ENVIRONMENT_VERSION, TrainingConfig, _epsilon_at

logger = logging.getLogger(__name__)


class SelfPlayTrainer:
    def __init__(
        self,
        training_config: TrainingConfig,
        reward_config: RewardConfig,
        run_dir: str | Path,
        battle_format: str = "gen9randombattle",
        player1_username: str = "Player1 A",
        player2_username: str = "Player2 B",
        team: str | None = None,
        env=None,
        init_checkpoint: str | Path | None = None,
    ):
        self.cfg = training_config
        self.reward_config = reward_config
        self.battle_format = battle_format
        self.run_id = f"selfplay_{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(run_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.player1_username = player1_username
        self.player2_username = player2_username
        self.init_checkpoint = init_checkpoint

        # Fail fast on a bad checkpoint path -- before opening any real
        # connection below -- rather than deep inside torch.load after
        # a websocket has already been opened and both accounts logged
        # in. A typo'd path is a common mistake (e.g. copying a run
        # directory but not the checkpoint filename inside it).
        if init_checkpoint is not None and not Path(init_checkpoint).is_file():
            raise FileNotFoundError(
                f"init_checkpoint not found: {init_checkpoint}\n"
                "Check the path and that training actually reached a "
                "checkpoint-saving step."
            )

        # Two explicit, distinct Player profiles -- avoids the
        # username-collision login bug that occurs when poke-env has to
        # auto-generate both agents' names from the same class name.
        # `env` can be injected directly (e.g. in tests, to avoid
        # opening a real websocket connection); production callers
        # leave it as None and get a real EncodedSinglesEnv.
        self.env = env or EncodedSinglesEnv(
            reward_config=self.reward_config,
            battle_format=battle_format,
            server_configuration=LOCAL_SERVER_CONFIGURATION,
            account_configuration1=AccountConfiguration(player1_username, None),
            account_configuration2=AccountConfiguration(player2_username, None),
            team=team,
        )

        self.device = torch.device(self.cfg.device)
        obs_size = observation_size()
        self.n_actions = self.env.action_spaces[self.env.possible_agents[0]].n

        self.q_network = DuelingQNetwork(obs_size, self.n_actions).to(self.device)
        if init_checkpoint is not None:
            # Warm-start from an existing checkpoint (e.g. one produced
            # by Trainer, this same class, or PooledSelfPlayTrainer --
            # they all save/load the same {"q_network": state_dict, ...}
            # format) instead of always beginning from random weights.
            state_dict = torch.load(init_checkpoint, map_location=self.device)
            if "q_network" not in state_dict:
                raise ValueError(
                    f"{init_checkpoint} loaded but doesn't look like a checkpoint saved "
                    "by this project (missing the 'q_network' key)."
                )
            self.q_network.load_state_dict(state_dict["q_network"])
            logger.info("Warm-started self-play network from %s", init_checkpoint)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.cfg.learning_rate)

        self.buffer = ReplayBuffer(self.cfg.replay_capacity, obs_size, self.n_actions, seed=self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.n_invalid_actions = 0
        self.n_total_actions = 0

        self.metrics_log_path = self.run_dir / "metrics.jsonl"
        self._write_run_metadata()

    def _write_run_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "environment_version": ENVIRONMENT_VERSION,
            "mode": "self_play_mirror",
            "reward_config": asdict(self.reward_config),
            "training_config": asdict(self.cfg),
            "battle_format": self.battle_format,
            "player1_username": self.player1_username,
            "player2_username": self.player2_username,
            "init_checkpoint": str(self.init_checkpoint) if self.init_checkpoint else None,
            "seed": self.cfg.seed,
            "created_at": time.time(),
        }
        (self.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    def _log_metrics(self, **kwargs) -> None:
        kwargs["timestamp"] = time.time()
        with open(self.metrics_log_path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")

    def _train_step(self) -> float:
        """Double-DQN update against the single shared network both
        players act from."""
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

    def _select_actions(self, observations: dict, agent_ids: list[str], epsilon: float) -> dict:
        actions = {}
        for agent_id in agent_ids:
            obs = observations[agent_id]["observation"]
            mask = observations[agent_id]["action_mask"]
            action = epsilon_greedy_action(self.q_network, obs, mask, epsilon, self.device, self.rng)
            # poke-env's own PokeEnv.step() -> SinglesEnv.action_to_order
            # requires a numpy scalar action -- see
            # environment/actions.py's as_poke_env_action for why.
            actions[agent_id] = as_poke_env_action(action)
        return actions

    def train(self) -> Path:
        observations, infos = self.env.reset(seed=self.cfg.seed)
        agent_ids = list(self.env.agents)
        episode_reward = {a: 0.0 for a in agent_ids}
        episode_count = 0
        losses: list[float] = []

        for step in range(1, self.cfg.total_steps + 1):
            epsilon = _epsilon_at(step, self.cfg)
            actions = self._select_actions(observations, agent_ids, epsilon)
            self.n_total_actions += len(actions)

            next_observations, rewards, terminated, truncated, infos = self.env.step(actions)

            for agent_id in agent_ids:
                done = bool(terminated.get(agent_id, False) or truncated.get(agent_id, False))
                self.buffer.add(
                    Transition(
                        observations[agent_id]["observation"],
                        actions[agent_id],
                        rewards[agent_id],
                        next_observations[agent_id]["observation"],
                        observations[agent_id]["action_mask"],
                        next_observations[agent_id]["action_mask"],
                        done,
                    )
                )
                episode_reward[agent_id] += rewards[agent_id]

            observations = next_observations

            if not self.env.agents:  # poke-env clears .agents once the battle finishes
                episode_count += 1
                # battle1/battle2 still hold the just-finished battle at
                # this point (before reset() reassigns them) -- each is
                # that player's own perspective on the SAME battle, so
                # exactly one of .won is True and the other's .lost is
                # True. Read both so callers (e.g.
                # scripts/two_players_battle.py's --learn summary) can
                # report a real win rate instead of only reward.
                battle1 = getattr(self.env, "battle1", None)
                battle2 = getattr(self.env, "battle2", None)
                player1_won = bool(battle1.won) if battle1 is not None else None
                player2_won = bool(battle2.won) if battle2 is not None else None
                self._log_metrics(
                    step=step,
                    event="episode_end",
                    episode=episode_count,
                    player1_reward=episode_reward.get(agent_ids[0], 0.0),
                    player2_reward=episode_reward.get(agent_ids[1], 0.0),
                    player1_won=player1_won,
                    player2_won=player2_won,
                    epsilon=epsilon,
                )
                observations, infos = self.env.reset()
                agent_ids = list(self.env.agents)
                episode_reward = {a: 0.0 for a in agent_ids}

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

        final_checkpoint = self._save_checkpoint(self.cfg.total_steps)
        logger.info("Self-play training complete. Final checkpoint: %s", final_checkpoint)
        return final_checkpoint