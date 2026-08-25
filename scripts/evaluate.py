#!/usr/bin/env python3
"""
Entry point for evaluating a trained checkpoint against held-out
opponents/teams. Never used for training and never touches live
Showdown -- see showdown/integration.py for that explicitly-gated path.

Usage:
    python scripts/evaluate.py --checkpoint runs/.../checkpoint_step200000.pt \
        --opponent heuristic --n-battles 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.inference import TrainedAgent
from environment.battle_env import make_env
from evaluation.benchmarks import evaluate_agent
from training.curriculum import OPPONENT_FACTORIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="heuristic", choices=list(OPPONENT_FACTORIES))
    parser.add_argument("--battle-format", type=str, default="gen9randombattle")
    parser.add_argument("--n-battles", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    opponent = OPPONENT_FACTORIES[args.opponent](args.battle_format, None)
    env = make_env(opponent=opponent, battle_format=args.battle_format, local=True)

    agent = TrainedAgent(args.checkpoint, n_actions=env.action_space.n, device=args.device)

    try:
        results = evaluate_agent(env, agent, n_battles=args.n_battles)
        print(json.dumps(results.summary(), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
