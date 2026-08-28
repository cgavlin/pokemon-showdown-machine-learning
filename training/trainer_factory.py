"""
Builds the right trainer for a given (config, curriculum stage) pair.

Centralizes the dispatch logic previously duplicated between
scripts/train.py, scripts/self_play_train.py, and
scripts/pooled_self_play_train.py's argument handling: which trainer
class to instantiate depends on the config's `self_play.enabled` /
`self_play.mode` fields, which used to only be documented in
configs/default.yaml's comments and never actually read by anything
(scripts/train.py always ran the fixed-opponent path regardless of
what the yaml said). Used by scripts/train.py directly and by
training/curriculum_runner.py (which needs to build a fresh trainer
per curriculum stage as it advances automatically).

Also builds a shared team pool from config["team_pool_dir"] (if set)
and threads it through to whichever trainer gets built -- see
environment/team_pool.py's TeamPool for why non-random-battle formats
need this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from environment.battle_env import make_env
from environment.rewards import RewardConfig
from environment.team_pool import TeamPool
from training.curriculum import CurriculumStage, make_opponent
from training.pooled_self_play_trainer import PooledSelfPlayTrainer
from training.self_play_trainer import SelfPlayTrainer
from training.trainer import Trainer, TrainingConfig

TrainerUnion = Union[Trainer, SelfPlayTrainer, PooledSelfPlayTrainer]


def _build_team(config: dict, seed: Optional[int] = None):
    """Returns a TeamPool built from config["team_pool_dir"], or None
    (letting poke-env generate a random team, which only works for
    random-battle formats) if that key isn't set."""
    team_pool_dir = config.get("team_pool_dir")
    if not team_pool_dir:
        return None
    return TeamPool.from_directory(
        team_pool_dir, pattern=config.get("team_pool_pattern", "*.txt"), seed=seed
    )


def build_trainer(
    config: dict,
    stage: CurriculumStage,
    run_dir: str | Path,
    init_checkpoint: Optional[str | Path] = None,
) -> TrainerUnion:
    """
    Reads config["self_play"] (if present) to decide which trainer to
    build for this stage:
      - self_play absent or self_play.enabled: false (the default) ->
        Trainer, playing against a single fixed scripted opponent for
        the whole run, per stage.opponent_factory. This is the
        original, unchanged behavior.
      - self_play.enabled: true, self_play.mode: "pooled" (the
        self_play default) -> PooledSelfPlayTrainer, training against
        a rotating pool of the learner's own past checkpoints
        (falling back to a scripted bootstrap opponent -- by default
        stage.opponent_factory itself -- until the pool has entries).
      - self_play.enabled: true, self_play.mode: "mirror" ->
        SelfPlayTrainer, the network playing literally itself.

    Also reads config["team_pool_dir"] (see environment/team_pool.py):
    if set, every side of every battle the built trainer opens (our
    own env, scripted opponents, self-play checkpoint opponents) draws
    from that same team pool instead of poke-env's per-battle random
    team generation, which only exists for random-battle formats.

    `init_checkpoint`, if given, warm-starts the built trainer's
    network from that checkpoint's weights instead of a fresh random
    initialization -- training/curriculum_runner.py passes the
    previous curriculum stage's final checkpoint here, so each stage
    actually builds on what the last one learned rather than
    restarting from scratch every stage. Validated up front, before
    any opponent or env is constructed, so a bad path fails fast
    instead of surfacing after a real connection has already been
    opened (the non-self-play path below opens its env in this very
    function, ahead of the Trainer that would otherwise do the
    validation).
    """
    if init_checkpoint is not None and not Path(init_checkpoint).is_file():
        raise FileNotFoundError(
            f"init_checkpoint not found: {init_checkpoint}\n"
            "Check the path and that training actually reached a "
            "checkpoint-saving step."
        )

    reward_config = RewardConfig(**config["reward"])
    training_config = TrainingConfig(**config["training"])
    self_play_cfg = config.get("self_play") or {}
    team = _build_team(config, seed=training_config.seed)

    if not self_play_cfg.get("enabled", False):
        opponent = make_opponent(stage, team=team)
        env = make_env(
            opponent=opponent,
            battle_format=stage.battle_format,
            reward_config=reward_config,
            local=True,
            team=team,
        )
        return Trainer(
            env=env,
            training_config=training_config,
            reward_config=reward_config,
            run_dir=run_dir,
            curriculum_stage_name=stage.name,
            init_checkpoint=init_checkpoint,
        )

    mode = self_play_cfg.get("mode", "pooled")

    if mode == "pooled":
        # Default the bootstrap opponent to this stage's own scripted
        # opponent (e.g. Stage 4's "heuristic"), so self-play still
        # starts from a sensible baseline instead of always RandomPlayer,
        # unless the config explicitly overrides it.
        bootstrap_opponent = self_play_cfg.get("bootstrap_opponent") or stage.opponent_factory
        return PooledSelfPlayTrainer(
            training_config=training_config,
            reward_config=reward_config,
            run_dir=run_dir,
            battle_format=stage.battle_format,
            episodes_per_opponent=self_play_cfg.get("episodes_per_opponent", 20),
            pool_max_size=self_play_cfg.get("max_pool_size", 20),
            bootstrap_opponent=bootstrap_opponent,
            pool_dir=self_play_cfg.get("pool_dir"),
            eval_opponent=self_play_cfg.get("eval_opponent", "heuristic"),
            eval_every_n_swaps=self_play_cfg.get("eval_every_n_swaps", 0),
            eval_battles=self_play_cfg.get("eval_battles", 20),
            team=team,
            init_checkpoint=init_checkpoint,
        )

    if mode == "mirror":
        return SelfPlayTrainer(
            training_config=training_config,
            reward_config=reward_config,
            run_dir=run_dir,
            battle_format=stage.battle_format,
            team=team,
            init_checkpoint=init_checkpoint,
        )

    raise ValueError(f"Unknown self_play.mode: {mode!r}; expected 'pooled' or 'mirror'")