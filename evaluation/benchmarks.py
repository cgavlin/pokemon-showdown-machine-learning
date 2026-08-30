"""
Held-out evaluation: Maintain held-out opponents, teams,
and battle scenarios so improvements can be measured without
overfitting and evaluation should use battles that are not directly
included in the training set.

This module never trains -- it only runs an already-trained (or
random-policy, for baseline) agent against opponents/teams that the
caller configures as held-out (e.g. a different opponent class/team
pool than was used in training/trainer.py), and reports AggregateMetrics.
"""

from __future__ import annotations

import logging

import numpy as np

from agents.inference import TrainedAgent
from environment.actions import describe_action
from environment.battle_env import ShowdownBattleEnv
from environment.state import MAX_TEAM_SIZE
from evaluation.metrics import AggregateMetrics, BattleMetrics

logger = logging.getLogger(__name__)


def _classify_action_effectiveness(battle, action_index: int, action_space_size: int) -> tuple[bool, bool]:
    """
    Returns (is_super_effective, is_ineffective) for a "move"-kind
    action (plain move or move+tera), based on the actual
    type-effectiveness multiplier of that move against the opponent's
    current active Pokemon -- using the same >=2 / <1 cutoffs
    environment/rewards.py's move_effectiveness_reward uses for its
    super_effective_move/ineffective_move reward buckets, so evaluation
    stats and training rewards agree on what "super effective" means.
    Switches and the default action always return (False, False).
    """
    if battle is None:
        return False, False

    meaning = describe_action(action_index, action_space_size)
    if meaning.kind not in ("move", "move+tera"):
        return False, False

    moves = list(battle.available_moves) if battle.available_moves else []
    if meaning.slot >= len(moves):
        return False, False

    move = moves[meaning.slot]
    opponent = battle.opponent_active_pokemon
    if opponent is None or move.type is None:
        return False, False

    try:
        multiplier = opponent.damage_multiplier(move)
    except Exception:
        return False, False

    return multiplier >= 2, multiplier < 1


def evaluate_agent(
    env: ShowdownBattleEnv,
    agent: TrainedAgent,
    n_battles: int,
    seed: int = 0,
) -> AggregateMetrics:
    """
    Runs `n_battles` battles in `env` (which should already be configured
    with a held-out opponent / held-out team pool distinct from training)
    and returns aggregate evaluation metrics.
    """
    aggregate = AggregateMetrics()
    rng = np.random.default_rng(seed)

    for battle_idx in range(n_battles):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False

        n_switches = 0
        n_moves = 0
        n_super_effective = 0
        n_ineffective = 0
        total_reward = 0.0
        turns = 0

        while not done:
            mask = env.get_action_mask()
            action = agent.act(obs, mask)

            # Classify effectiveness using the battle state as it was
            # WHEN the action was chosen (available_moves/opponent's
            # active Pokemon at this turn), before stepping the env
            # changes it for the next turn.
            battle_before_action = env._underlying.battle1
            is_super, is_ineffective = _classify_action_effectiveness(
                battle_before_action, action, len(mask)
            )

            if action >= 9:
                n_switches += 1
            elif action >= 1:
                n_moves += 1
                n_super_effective += int(is_super)
                n_ineffective += int(is_ineffective)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            turns += 1

        battle = env._underlying.battle1  # OUR agent's (agent1's) perspective on the
        # just-finished battle -- NOT battle2, which is the opponent's
        # perspective (its own team vs. ours). Mixing these up silently
        # flips win/loss and damage-dealt/taken for every metric below.
        own_fainted = sum(1 for p in battle.team.values() if p.fainted) if battle else 0
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted) if battle else 0
        own_hp = sum(p.current_hp_fraction or 0.0 for p in battle.team.values()) if battle else 0.0
        opp_hp = sum(p.current_hp_fraction or 0.0 for p in battle.opponent_team.values()) if battle else 0.0

        won = bool(battle.won) if battle else False
        lost = bool(battle.lost) if battle else False
        drew = not won and not lost

        metrics = BattleMetrics(
            won=won,
            lost=lost,
            drew=drew,
            total_reward=total_reward,
            own_fainted=own_fainted,
            opponent_fainted=opp_fainted,
            own_damage_dealt_fraction=max(0.0, MAX_TEAM_SIZE - opp_hp),
            own_damage_taken_fraction=max(0.0, MAX_TEAM_SIZE - own_hp),
            n_switches=n_switches,
            n_moves=n_moves,
            n_super_effective_moves=n_super_effective,
            n_ineffective_moves=n_ineffective,
            turns=turns,
        )
        aggregate.add(metrics)

        if (battle_idx + 1) % 50 == 0:
            logger.info("Evaluated %d/%d battles", battle_idx + 1, n_battles)

    return aggregate
