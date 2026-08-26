"""
Action space for the Pokemon battle RL environment.

The environment should validate actions before applying
them. Invalid actions should not silently become arbitrary legal actions.

poke-env's `SinglesEnv` already encodes actions as a flat Discrete space
(0..N-1) covering moves 1-4 (+ optional terastallize/mega/dynamax
variants) and switches 1-6, and raises/handles illegal actions internally
via `action_to_order`. This module wraps that behavior with:
  1. An explicit, documented mapping description (for logging/inspection).
  2. A hard validation layer used by our own env wrapper so that illegal
     actions produce a well-defined penalty instead of being silently
     remapped to "whatever the default legal action is" without the
     agent (and our logs) knowing a mistake happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Invalid-action penalty applied by ShowdownBattleEnv when the policy
# selects an action index that is not currently legal. This should be
# large enough to strongly discourage illegal actions but must not
# dominate the terminal win/loss reward (see rewards.py).
INVALID_ACTION_PENALTY = -1.0


@dataclass(frozen=True)
class ActionMeaning:
    index: int
    kind: str  # "move" | "switch" | "move+tera" | "move+mega" | "move+dynamax" | "default"
    slot: int  # 0-3 for moves, 0-5 for switches


def describe_action(action_index: int, action_space_size: int) -> ActionMeaning:
    """
    Best-effort human-readable description of a poke-env SinglesEnv action
    index, matching the ordering documented in poke-env's `SinglesEnv`:
    [default, move1..move4, move1+tera..move4+tera, switch1..switch6]
    (exact layout can vary slightly by format; used for logging only --
    never for re-deriving legality, which always goes through
    `env.get_action_mask()` / `action_to_order`).
    """
    if action_index == 0:
        return ActionMeaning(action_index, "default", 0)
    if 1 <= action_index <= 4:
        return ActionMeaning(action_index, "move", action_index - 1)
    if 5 <= action_index <= 8:
        return ActionMeaning(action_index, "move+tera", action_index - 5)
    return ActionMeaning(action_index, "switch", action_index - 9)


def is_action_legal(action_index: int, action_mask) -> bool:
    """action_mask is the boolean/int array returned by env.get_action_mask()."""
    if action_index < 0 or action_index >= len(action_mask):
        return False
    return bool(action_mask[action_index])


def sanitize_action(action_index: int, action_mask) -> tuple[int, bool]:
    """
    Returns (action_to_apply, was_illegal). If the requested action is
    illegal, falls back to the first legal action (index 0, the game's
    default order, which poke-env guarantees is always legal) so the
    battle can proceed -- but callers MUST apply INVALID_ACTION_PENALTY
    and log the event whenever `was_illegal` is True. We never want an
    illegal action to silently look like a normal, rewarded action.
    """
    if is_action_legal(action_index, action_mask):
        return action_index, False
    return 0, True


def as_poke_env_action(action: int) -> np.int64:
    return np.int64(action)