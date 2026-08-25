#!/usr/bin/env python3
"""
Entry point for pooled self-play training: the learner trains against
opponents sampled from a SelfPlayPool of its own past checkpoints
(falling back to a scripted opponent until the pool has entries),
periodically checkpointing itself back into the pool.

This differs from:
  - scripts/train.py: opponent is a single, fixed, non-learning
    scripted Player for the whole run.
  - scripts/self_play_train.py: the network plays literally itself
    every battle (mirror self-play) -- no pool, no frozen past
    versions, both sides always identical.

Usage:
    python scripts/pooled_self_play_train.py --config configs/default.yaml
    python scripts/pooled_self_play_train.py --config configs/default.yaml \
        --episodes-per-opponent 30 --bootstrap-opponent heuristic

Requires a local Pokemon Showdown server (see README.md). Never
connects anywhere else -- see showdown/integration.py for the
explicitly-gated live-battle path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.rewards import RewardConfig
from training.curriculum import OPPONENT_FACTORIES
from training.pooled_self_play_trainer import PooledSelfPlayTrainer
from training.trainer import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--episodes-per-opponent", type=int, default=20)
    parser.add_argument("--pool-max-size", type=int, default=20)
    parser.add_argument(
        "--bootstrap-opponent", type=str, default="random", choices=list(OPPONENT_FACTORIES)
    )
    parser.add_argument(
        "--eval-opponent",
        type=str,
        default="heuristic",
        choices=list(OPPONENT_FACTORIES),
        help="Fixed scripted opponent used for the periodic absolute-skill check (see --eval-every-n-swaps).",
    )
    parser.add_argument(
        "--eval-every-n-swaps",
        type=int,
        default=0,
        help="Run the absolute-skill check every N opponent swaps. 0 (default) disables it.",
    )
    parser.add_argument("--eval-battles", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = yaml.safe_load(Path(args.config).read_text())
    reward_config = RewardConfig(**config["reward"])
    training_config = TrainingConfig(**config["training"])

    trainer = PooledSelfPlayTrainer(
        training_config=training_config,
        reward_config=reward_config,
        run_dir=args.run_dir,
        battle_format=config.get("battle_format", "gen9randombattle"),
        episodes_per_opponent=args.episodes_per_opponent,
        pool_max_size=args.pool_max_size,
        bootstrap_opponent=args.bootstrap_opponent,
        eval_opponent=args.eval_opponent,
        eval_every_n_swaps=args.eval_every_n_swaps,
        eval_battles=args.eval_battles,
    )

    try:
        checkpoint_path = trainer.train()
        print(f"Pooled self-play training complete. Final checkpoint: {checkpoint_path}")
        print(f"Self-play pool: {len(trainer.pool.entries)} checkpoint(s) at {trainer.pool.pool_dir}")
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
