"""
CheckpointPlayer: wraps a trained DQN checkpoint (agents/inference.py's
TrainedAgent) as a poke-env Player, so a trained policy can act as an
ordinary opponent in poke-env battles -- outside our Gym env entirely.

Used by:
  - scripts/two_players_battle.py, to let a checkpoint battle a
    scripted opponent (or another checkpoint) as a quick sanity check.
  - training/pooled_self_play_trainer.py, as a FROZEN (non-learning)
    opponent sampled from a SelfPlayPool of past checkpoints.
"""

from __future__ import annotations

import re

import numpy as np
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player

from agents.inference import TrainedAgent
from environment.state import encode_battle


def _gen_from_format(battle_format: str) -> int:
    """Extract the generation number from a poke-env format string like
    'gen9randombattle' or 'gen9ou'. Defaults to 9 if unparseable."""
    match = re.match(r"gen(\d+)", battle_format)
    return int(match.group(1)) if match else 9


class CheckpointPlayer(Player):
    """A poke-env Player whose moves come from a trained DQN checkpoint
    instead of scripted logic. The checkpoint is frozen -- this player
    never learns or updates weights during battles."""

    def __init__(self, checkpoint_path: str, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        gen = _gen_from_format(kwargs.get("battle_format", "gen9randombattle"))
        n_actions = SinglesEnv.get_action_space_size(gen)
        self._agent = TrainedAgent(checkpoint_path, n_actions=n_actions, device=device)

    def choose_move(self, battle):
        obs = encode_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int64)
        action_index = self._agent.act(obs, mask)

        # Translate the flat action index back into a poke-env BattleOrder
        # using the same layout SinglesEnv.get_action_mask/action_to_order
        # use internally: [switch, move, mega, zmove, dynamax, tera].
        return SinglesEnv.action_to_order(action_index, battle, fake=False, strict=False)
