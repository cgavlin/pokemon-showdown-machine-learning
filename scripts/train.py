#!/usr/bin/env python3
"""
Entry point for local training runs.

Reads config["self_play"] to decide what kind of training to run (see
training/trainer_factory.py for the exact rules):
  - self_play.enabled: false (or absent, the default) -> train against
    a single fixed scripted opponent, per the curriculum stage.
  - self_play.enabled: true, self_play.mode: "pooled" (the self_play
    default) -> pooled self-play against a rotating pool of the
    learner's own past checkpoints.
  - self_play.enabled: true, self_play.mode: "mirror" -> mirror
    self-play (the network plays literally itself).

Usage:
    python scripts/train.py --config configs/default.yaml

Requires a local Pokemon Showdown server running (see README.md
"Local Showdown server" for setup) -- this script never connects
anywhere else; see showdown/integration.py for the explicitly-gated
live-battle path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.curriculum import get_stage
from training.trainer_factory import build_trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-dir", type=str, default="runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = yaml.safe_load(Path(args.config).read_text())
    stage = get_stage(config["curriculum_stage"])

    trainer = build_trainer(config, stage, run_dir=args.run_dir)

    try:
        checkpoint_path = trainer.train()
        print(f"Training complete. Final checkpoint: {checkpoint_path}")
    finally:
        # Trainer/SelfPlayTrainer/PooledSelfPlayTrainer all expose .env
        # with a .close() method; PooledSelfPlayTrainer additionally
        # already closes its own env at the end of train() (it may have
        # rebuilt it several times over the run), so this is a no-op in
        # that case rather than a double-close of a *different* object.
        trainer.env.close()


if __name__ == "__main__":
    main()

