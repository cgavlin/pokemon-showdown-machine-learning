"""
Pooled self-play trainer.

Wires training/self_play.py's SelfPlayPool into actual training:
  - save successful versions of the agent
  - add previous versions to an opponent pool
  - sample opponents from that pool during training
  - periodically evaluate the current agent against older versions
  - keep particularly challenging opponents available

This is different from training/self_play_trainer.py's SelfPlayTrainer
(mirror self-play: the network plays literally itself every battle, so
both sides are always identical and can drift together). Here, only
the LEARNER (agent1, in the normal single-agent ShowdownBattleEnv)
updates its weights. Its opponent is a FROZEN snapshot -- either a
past checkpoint of the learner sampled from the pool, wrapped as a
poke-env Player via agents/checkpoint_player.py's CheckpointPlayer, or
(before any checkpoint exists) a scripted bootstrap opponent.

Every `episodes_per_opponent` episodes:
  1. The learner's win rate over the just-finished window against its
     current opponent is recorded back into the pool
     (SelfPlayPool.update_win_rate), so opponents that beat the learner
     often get pinned as "hard" and easy ones are eligible for eviction.
  2. The learner's current weights are checkpointed and added to the
     pool, so this version becomes available as a future opponent.
  3. A new opponent is sampled from the pool (weighted toward
     historically-challenging checkpoints) and the environment is
     rebuilt against it.

Self-play win rate alone is a non-stationary signal: it's measured
against an opponent pool that is itself made of past versions of the
learner, so "the learner is winning more" can mean either genuine
improvement or that the pool has drifted to be easier, and there's no
way to tell which from that number alone. Every `eval_every_n_swaps`
opponent swaps (disabled by default -- see eval_every_n_swaps below),
this trainer additionally runs the current checkpoint through a short
held-out evaluation against a FIXED scripted opponent (not sampled
from the pool, and not affected by self-play at all), logged as a
separate "eval_vs_fixed_opponent" metric event. That's the number to
actually watch for whether the agent is improving over time.

Always connects to the LOCAL server, never the public ladder.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from poke_env.player import Player

from agents.checkpoint_player import CheckpointPlayer
from agents.inference import TrainedAgent
from agents.policy import DuelingQNetwork, epsilon_greedy_action
from agents.value_function import ReplayBuffer, Transition
from environment.battle_env import ShowdownBattleEnv, make_env
from environment.rewards import RewardConfig
from environment.state import observation_size
from evaluation.benchmarks import evaluate_agent
from training.curriculum import OPPONENT_FACTORIES
from training.dqn_update import double_dqn_update
from training.self_play import PoolEntry, SelfPlayPool
from training.trainer import ENVIRONMENT_VERSION, TrainingConfig, _epsilon_at

logger = logging.getLogger(__name__)


class PooledSelfPlayTrainer:
    def __init__(
        self,
        training_config: TrainingConfig,
        reward_config: RewardConfig,
        run_dir: str | Path,
        battle_format: str = "gen9randombattle",
        episodes_per_opponent: int = 20,
        pool_max_size: int = 20,
        bootstrap_opponent: str = "random",
        pool_dir: Optional[str | Path] = None,
        env_factory: Optional[Callable[[Player], ShowdownBattleEnv]] = None,
        eval_opponent: str = "heuristic",
        eval_every_n_swaps: int = 0,
        eval_battles: int = 20,
        eval_env_factory: Optional[Callable[[Player], ShowdownBattleEnv]] = None,
        team=None,
        init_checkpoint: Optional[str | Path] = None,
    ):
        # Fail fast on a bad init_checkpoint path -- before sampling an
        # opponent or opening the env's real websocket connection below
        # -- rather than deep inside torch.load after a connection has
        # already been made. Mirrors SelfPlayTrainer's own check.
        if init_checkpoint is not None and not Path(init_checkpoint).is_file():
            raise FileNotFoundError(
                f"init_checkpoint not found: {init_checkpoint}\n"
                "Check the path and that training actually reached a "
                "checkpoint-saving step."
            )
        self.init_checkpoint = init_checkpoint

        self.cfg = training_config
        self.reward_config = reward_config
        self.battle_format = battle_format
        self.episodes_per_opponent = max(1, episodes_per_opponent)
        self.bootstrap_opponent = bootstrap_opponent
        # A team (str or poke-env Teambuilder, e.g.
        # environment/team_pool.py's TeamPool) used for every side of
        # every battle this trainer opens: our own env, the scripted
        # bootstrap opponent, sampled CheckpointPlayer opponents, and
        # the fixed-opponent absolute-skill eval. Leave as None for
        # random-battle formats, which get team variety for free from
        # the server; required for non-random formats like gen9ou.
        self.team = team
        # Injectable so tests can swap in a fake env instead of opening
        # a real websocket connection; production callers leave this
        # as None and get a real local ShowdownBattleEnv.
        self._env_factory = env_factory or self._default_env_factory

        # Absolute-skill check config. eval_every_n_swaps=0 (the
        # default) disables this entirely -- opt in explicitly, since
        # each eval opens its own short-lived env/connection on top of
        # training. eval_opponent must be a key in
        # training.curriculum.OPPONENT_FACTORIES.
        self.eval_opponent_kind = eval_opponent
        self.eval_every_n_swaps = max(0, eval_every_n_swaps)
        self.eval_battles = eval_battles
        self._eval_env_factory = eval_env_factory or self._default_eval_env_factory
        self._swap_count = 0

        self.run_id = f"pooled_selfplay_{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(run_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # By default the pool lives inside this run's own directory
        # (isolated, matching every earlier version of this trainer).
        # Pass an explicit pool_dir to instead point at a pool shared
        # across multiple separate training runs/processes -- e.g. to
        # resume self-play later, or run several trainers against a
        # common, steadily-growing pool. SelfPlayPool.add_checkpoint
        # always gives its copies unique filenames, so sharing a
        # pool_dir across runs is safe even if two runs' own step
        # counters happen to produce identically-named checkpoints.
        self.pool = SelfPlayPool(
            pool_dir=Path(pool_dir) if pool_dir else self.run_dir / "self_play_pool",
            max_pool_size=pool_max_size,
        )

        self.device = torch.device(self.cfg.device)
        obs_size = observation_size()

        self._current_opponent_entry: Optional[PoolEntry] = None
        opponent = self._sample_opponent()
        self.env = self._env_factory(opponent)

        n_actions = self.env.action_space.n
        self.q_network = DuelingQNetwork(obs_size, n_actions).to(self.device)
        if self.init_checkpoint is not None:
            # Warm-start the LEARNER's weights from a prior checkpoint
            # (e.g. the previous curriculum stage's final checkpoint,
            # threaded through by training/curriculum_runner.py) instead
            # of always beginning from random initialization. This is
            # independent of the opponent pool: opponents sampled from
            # the pool are already past checkpoints, but until now the
            # learner itself always restarted from scratch.
            state_dict = torch.load(self.init_checkpoint, map_location=self.device)
            if "q_network" not in state_dict:
                raise ValueError(
                    f"{self.init_checkpoint} loaded but doesn't look like a checkpoint "
                    "saved by this project (missing the 'q_network' key)."
                )
            self.q_network.load_state_dict(state_dict["q_network"])
            logger.info("Warm-started pooled self-play learner from %s", self.init_checkpoint)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.cfg.learning_rate)

        self.buffer = ReplayBuffer(self.cfg.replay_capacity, obs_size, n_actions, seed=self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.metrics_log_path = self.run_dir / "metrics.jsonl"
        self._write_run_metadata()

    def _default_env_factory(self, opponent: Player) -> ShowdownBattleEnv:
        return make_env(
            opponent=opponent,
            battle_format=self.battle_format,
            reward_config=self.reward_config,
            local=True,
            team=self.team,
        )

    def _default_eval_env_factory(self, opponent: Player) -> ShowdownBattleEnv:
        return make_env(opponent=opponent, battle_format=self.battle_format, local=True, team=self.team)

    def _sample_opponent(self) -> Player:
        """Sample an opponent from the pool, weighted toward
        historically-challenging checkpoints. Falls back to a scripted
        opponent while the pool is empty (the very first swap window)."""
        entry = self.pool.sample()
        if entry is None:
            self._current_opponent_entry = None
            factory = OPPONENT_FACTORIES[self.bootstrap_opponent]
            opponent = factory(self.battle_format, self.team)
            logger.info(
                "Self-play pool empty -- bootstrapping against scripted opponent: %s",
                type(opponent).__name__,
            )
            return opponent

        self._current_opponent_entry = entry
        logger.info(
            "Sampled opponent from pool: %s (win_rate_vs_current=%.2f, times_sampled=%d, pinned=%s)",
            Path(entry.checkpoint_path).name,
            entry.win_rate_vs_current,
            entry.times_sampled,
            entry.is_pinned_hard,
        )
        return CheckpointPlayer(
            checkpoint_path=entry.checkpoint_path,
            device=str(self.device),
            battle_format=self.battle_format,
            team=self.team,
        )

    def _write_run_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "environment_version": ENVIRONMENT_VERSION,
            "mode": "pooled_self_play",
            "reward_config": asdict(self.reward_config),
            "training_config": asdict(self.cfg),
            "battle_format": self.battle_format,
            "episodes_per_opponent": self.episodes_per_opponent,
            "bootstrap_opponent": self.bootstrap_opponent,
            "eval_opponent": self.eval_opponent_kind,
            "eval_every_n_swaps": self.eval_every_n_swaps,
            "eval_battles": self.eval_battles,
            "init_checkpoint": str(self.init_checkpoint) if self.init_checkpoint else None,
            "seed": self.cfg.seed,
            "created_at": time.time(),
        }
        (self.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    def _log_metrics(self, **kwargs) -> None:
        kwargs["timestamp"] = time.time()
        with open(self.metrics_log_path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")

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

    def _current_opponent_label(self) -> str:
        if self._current_opponent_entry is not None:
            return Path(self._current_opponent_entry.checkpoint_path).name
        return f"scripted:{self.bootstrap_opponent}"

    def _run_fixed_opponent_eval(self, step: int, checkpoint_path: Path) -> None:
        """Absolute-skill check: evaluate the just-saved checkpoint
        against a FIXED scripted opponent (never sampled from the
        pool), so progress is visible even while self-play's own
        internal win rate is a moving target."""
        opponent = OPPONENT_FACTORIES[self.eval_opponent_kind](self.battle_format, self.team)
        eval_env = self._eval_env_factory(opponent)
        try:
            agent = TrainedAgent(checkpoint_path, n_actions=eval_env.action_space.n, device=str(self.device))
            metrics = evaluate_agent(eval_env, agent, n_battles=self.eval_battles)
        finally:
            eval_env.close()

        summary = metrics.summary()
        self._log_metrics(step=step, event="eval_vs_fixed_opponent", opponent=self.eval_opponent_kind, **summary)
        logger.info(
            "Eval vs fixed opponent (%s) at step %d: win_rate=%.2f (%d battles)",
            self.eval_opponent_kind,
            step,
            summary["win_rate"],
            self.eval_battles,
        )

    def _swap_opponent(self, step: int, recent_win_rate: float) -> None:
        # Record how the learner did against whichever opponent it was
        # just playing (skip for the scripted bootstrap, which isn't a
        # pool entry), then checkpoint the learner into the pool so
        # this version becomes available as a future opponent, then
        # sample a new opponent and rebuild the env against it.
        if self._current_opponent_entry is not None:
            self.pool.update_win_rate(
                self._current_opponent_entry.checkpoint_path, win_rate_vs_current=recent_win_rate
            )

        checkpoint_path = self._save_checkpoint(step)
        self.pool.add_checkpoint(checkpoint_path)
        self._swap_count += 1

        if self.eval_every_n_swaps > 0 and self._swap_count % self.eval_every_n_swaps == 0:
            self._run_fixed_opponent_eval(step, checkpoint_path)

        self.env.close()
        opponent = self._sample_opponent()
        self.env = self._env_factory(opponent)
        logger.info(
            "Swapped opponent at step %d (learner win rate vs previous opponent: %.2f); pool size=%d",
            step,
            recent_win_rate,
            len(self.pool.entries),
        )

    def train(self) -> Path:
        obs, info = self.env.reset(seed=self.cfg.seed)
        episode_reward = 0.0
        episode_count = 0
        episodes_since_swap = 0
        wins_since_swap = 0
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
                episodes_since_swap += 1
                battle = self.env._underlying.battle1
                won = bool(battle.won) if battle is not None else False
                wins_since_swap += int(won)

                self._log_metrics(
                    step=step,
                    event="episode_end",
                    episode=episode_count,
                    episode_reward=episode_reward,
                    epsilon=epsilon,
                    won=won,
                    opponent=self._current_opponent_label(),
                )

                if episodes_since_swap >= self.episodes_per_opponent:
                    recent_win_rate = wins_since_swap / episodes_since_swap
                    self._swap_opponent(step, recent_win_rate)
                    episodes_since_swap = 0
                    wins_since_swap = 0

                episode_reward = 0.0
                obs, info = self.env.reset()

            if step >= self.cfg.learning_starts and step % self.cfg.train_every == 0:
                if len(self.buffer) >= self.cfg.batch_size:
                    batch = self.buffer.sample(self.cfg.batch_size)
                    losses.append(
                        double_dqn_update(
                            self.q_network, self.target_network, self.optimizer, batch, self.device, self.cfg.gamma
                        )
                    )

            if step % self.cfg.target_update_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            if step % self.cfg.eval_every_steps == 0:
                avg_loss = float(np.mean(losses)) if losses else None
                self._log_metrics(
                    step=step, event="train_progress", avg_loss=avg_loss, pool_size=len(self.pool.entries)
                )
                losses = []
                self._save_checkpoint(step)

        # Final bookkeeping for whichever opponent we ended the run on.
        if episodes_since_swap > 0 and self._current_opponent_entry is not None:
            self.pool.update_win_rate(
                self._current_opponent_entry.checkpoint_path,
                win_rate_vs_current=wins_since_swap / episodes_since_swap,
            )

        final_checkpoint = self._save_checkpoint(self.cfg.total_steps)
        self.pool.add_checkpoint(final_checkpoint)
        logger.info("Pooled self-play training complete. Final checkpoint: %s", final_checkpoint)
        return final_checkpoint