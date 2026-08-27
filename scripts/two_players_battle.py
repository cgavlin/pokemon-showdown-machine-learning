#!/usr/bin/env python3
"""
Demo: two explicit Player profiles battling each other on a local
Pokemon Showdown server.

This is separate from the RL training path (environment/battle_env.py,
scripts/train.py) -- it doesn't touch our Gym env, state encoding, or
reward function at all. It's a quick sanity check that:
  1. the local Showdown server is reachable, and
  2. two independently-named Player agents can log in and battle.

By default both players are scripted (RandomPlayer vs
SimpleHeuristicsPlayer) so this runs with no trained checkpoint
required. Pass --player1-checkpoint/--player2-checkpoint to instead
have one or both sides play from a trained DQN checkpoint via
agents/inference.py.

Usage:
    python scripts/two_players_battle.py --n-battles 5
    python scripts/two_players_battle.py --player1 random --player2 max_base_power
    python scripts/two_players_battle.py --player1-checkpoint runs/.../checkpoint_step200000.pt

Always connects to the LOCAL server (ws://localhost:8000) -- never the
public ladder. See showdown/integration.py for the explicitly-gated
live-battle path if you ever want real opponents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from agents.checkpoint_player import CheckpointPlayer
from environment.battle_env import LOCAL_SERVER_CONFIGURATION
from environment.rewards import RewardConfig
from training.self_play_trainer import SelfPlayTrainer
from training.trainer import TrainingConfig

logger = logging.getLogger(__name__)

SCRIPTED_PLAYER_CLASSES = {
    "random": RandomPlayer,
    "max_base_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


def _make_scripted_player(kind: str, battle_format: str, username: str) -> Player:
    player_class = SCRIPTED_PLAYER_CLASSES[kind]
    return player_class(
        account_configuration=AccountConfiguration(username, None),
        battle_format=battle_format,
        server_configuration=LOCAL_SERVER_CONFIGURATION,
    )


def _make_player(
    kind: Optional[str],
    checkpoint: Optional[str],
    battle_format: str,
    username: str,
    device: str,
) -> Player:
    if checkpoint:
        return CheckpointPlayer(
            checkpoint_path=checkpoint,
            device=device,
            account_configuration=AccountConfiguration(username, None),
            battle_format=battle_format,
            server_configuration=LOCAL_SERVER_CONFIGURATION,
        )
    return _make_scripted_player(kind or "random", battle_format, username)


async def _run(args: argparse.Namespace) -> None:
    player1 = _make_player(
        args.player1, args.player1_checkpoint, args.battle_format, "Real Human Being", args.device
    )
    player2 = _make_player(
        args.player2, args.player2_checkpoint, args.battle_format, "Not A Robot", args.device
    )

    logger.info(
        "Starting %d battle(s): %s vs %s (format=%s)",
        args.n_battles,
        player1.username,
        player2.username,
        args.battle_format,
    )

    await player1.battle_against(player2, n_battles=args.n_battles)

    print(f"{player1.username} won {player1.n_won_battles}/{args.n_battles}")
    print(f"{player2.username} won {player2.n_won_battles}/{args.n_battles}")


def _summarize_self_play_run(metrics_log_path: Path) -> dict:
    """Reads a SelfPlayTrainer run's metrics.jsonl and computes simple
    per-player win-rate stats from its "episode_end" events."""
    episodes = 0
    player1_wins = 0
    player2_wins = 0
    with open(metrics_log_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("event") != "episode_end":
                continue
            episodes += 1
            player1_wins += int(bool(record.get("player1_won")))
            player2_wins += int(bool(record.get("player2_won")))

    return {
        "episodes": episodes,
        "player1_wins": player1_wins,
        "player2_wins": player2_wins,
        "player1_win_rate": player1_wins / episodes if episodes else 0.0,
        "player2_win_rate": player2_wins / episodes if episodes else 0.0,
    }


def _run_learn(args: argparse.Namespace) -> None:
    """--learn mode: instead of a plain scripted/checkpoint battle, run
    mirror self-play (training/self_play_trainer.py's SelfPlayTrainer)
    for the two named players -- a single shared DQN actually learns
    from both sides of every battle -- then print a win-rate summary.
    This is a thin, config-free wrapper around the same trainer
    scripts/self_play_train.py uses; reach for that script instead if
    you need reward/training hyperparameters beyond the defaults.
    """
    training_config = TrainingConfig(total_steps=args.total_steps, device=args.device)
    reward_config = RewardConfig()

    trainer = SelfPlayTrainer(
        training_config=training_config,
        reward_config=reward_config,
        run_dir=args.run_dir,
        battle_format=args.battle_format,
        player1_username="Real Human Being",
        player2_username="Not A Robot",
    )

    logger.info(
        "Starting self-play learning: %s vs %s (format=%s, total_steps=%d)",
        trainer.player1_username,
        trainer.player2_username,
        args.battle_format,
        args.total_steps,
    )

    try:
        checkpoint_path = trainer.train()
    finally:
        trainer.env.close()

    summary = _summarize_self_play_run(trainer.metrics_log_path)
    print(f"Self-play learning complete over {summary['episodes']} episode(s).")
    print(
        f"{trainer.player1_username} won {summary['player1_wins']}/{summary['episodes']} "
        f"({summary['player1_win_rate']:.2%})"
    )
    print(
        f"{trainer.player2_username} won {summary['player2_wins']}/{summary['episodes']} "
        f"({summary['player2_win_rate']:.2%})"
    )
    print(f"Final checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player1", type=str, default="random", choices=list(SCRIPTED_PLAYER_CLASSES),
        help="Scripted policy for player 1 (ignored if --player1-checkpoint is set).",
    )
    parser.add_argument(
        "--player2", type=str, default="heuristic", choices=list(SCRIPTED_PLAYER_CLASSES),
        help="Scripted policy for player 2 (ignored if --player2-checkpoint is set).",
    )
    parser.add_argument("--player1-checkpoint", type=str, default=None)
    parser.add_argument("--player2-checkpoint", type=str, default=None)
    parser.add_argument("--battle-format", type=str, default="gen9randombattle")
    parser.add_argument("--n-battles", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--learn", action="store_true",
        help="Instead of a scripted/checkpoint battle, run mirror "
             "self-play training between the two players (a shared DQN "
             "learns from both sides) and print a win-rate summary.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=40_000,
        help="Total training steps for --learn mode (ignored otherwise).",
    )
    parser.add_argument(
        "--run-dir", type=str, default="runs",
        help="Where to write the self-play run's checkpoints/metrics for --learn mode.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.learn:
        _run_learn(args)
        return

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()