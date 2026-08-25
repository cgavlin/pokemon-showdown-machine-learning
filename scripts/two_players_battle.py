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
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from agents.checkpoint_player import CheckpointPlayer
from environment.battle_env import LOCAL_SERVER_CONFIGURATION

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
