#!/usr/bin/env python3
"""
Entry point for self-play training: two explicit Player profiles
battle each other on a local Pokemon Showdown server, and a single
shared Dueling DQN learns from both sides of every battle.

This is distinct from scripts/train.py (one learning policy vs. a
fixed, non-learning scripted opponent) and from
scripts/two_players_battle.py (two players battle but nothing learns).
Here, both players ARE the network being trained.

Usage:
    python scripts/self_play_train.py --config configs/default.yaml
    python scripts/self_play_train.py --config configs/default.yaml \
        --player1-username "Player1 A" --player2-username "Player2 B"

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
from training.self_play_trainer import SelfPlayTrainer
from training.trainer import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--player1-username", type=str, default="Player1 A")
    parser.add_argument("--player2-username", type=str, default="Player2 B")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = yaml.safe_load(Path(args.config).read_text())

    reward_config = RewardConfig(**config["reward"])
    training_config = TrainingConfig(**config["training"])

    trainer = SelfPlayTrainer(
        training_config=training_config,
        reward_config=reward_config,
        run_dir=args.run_dir,
        battle_format=config.get("battle_format", "gen9randombattle"),
        player1_username=args.player1_username,
        player2_username=args.player2_username,
    )

    checkpoint_path = None
    try:
        checkpoint_path = trainer.train()
        print(f"Self-play training complete. Final checkpoint: {checkpoint_path}")
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
