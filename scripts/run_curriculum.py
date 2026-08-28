#!/usr/bin/env python3
"""
Entry point for automatic curriculum advancement: trains each stage in
CURRICULUM order, evaluates the result against a held-out opponent,
and only proceeds to the next stage if that stage's
min_eval_win_rate_to_advance gate is met. Stops automatically before
stage5_human_opponents -- live battles remain behind the independent,
explicit safety gate in showdown/integration.py.

Each stage's trainer is warm-started from the PREVIOUS stage's final
checkpoint, so the curriculum actually builds on what earlier stages
learned instead of every stage starting from a fresh random network
(see training/curriculum_runner.py). Use --init-checkpoint when
resuming a run with --start-stage to warm-start that first stage from
an existing checkpoint too, instead of starting it random.

Usage:
    python scripts/run_curriculum.py --config configs/default.yaml
    python scripts/run_curriculum.py --config configs/default.yaml \
        --start-stage stage2_tactical_decisions --eval-battles 100
    python scripts/run_curriculum.py --config configs/default.yaml \
        --start-stage stage3_strategic_play \
        --init-checkpoint runs/.../stage2_tactical_decisions_.../checkpoint_step200000.pt

Writes runs/curriculum_progress.jsonl (one line per stage evaluated)
in addition to each stage's own run_metadata.json / metrics.jsonl /
checkpoints under the normal per-stage run directories.

Requires a local Pokemon Showdown server (see README.md). This can be
a long-running process -- each stage trains to config["training"]["total_steps"]
before being evaluated, and there is no time limit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.curriculum_runner import CurriculumRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument(
        "--start-stage",
        type=str,
        default=None,
        help="Resume from this stage name instead of stage1_basic_mechanics.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Warm-start the first stage trained (stage1, or --start-stage "
        "when resuming) from this checkpoint instead of a random network. "
        "Every later stage is always warm-started from the previous "
        "stage's own final checkpoint regardless of this flag.",
    )
    parser.add_argument("--eval-battles", type=int, default=200)
    parser.add_argument("--eval-device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base_config = yaml.safe_load(Path(args.config).read_text())

    runner = CurriculumRunner(
        base_config=base_config,
        run_dir=args.run_dir,
        start_stage=args.start_stage,
        eval_battles=args.eval_battles,
        eval_device=args.eval_device,
        init_checkpoint=args.init_checkpoint,
    )
    runner.run()
    print(f"Curriculum run complete. Progress log: {runner.progress_log_path}")


if __name__ == "__main__":
    main()