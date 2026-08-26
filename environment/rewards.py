"""
Reward function for the Pokemon battle RL environment.

Implements this structure:

    total_reward =
        tactical_rewards
        + positional_rewards
        + knockout_rewards
        + win_reward
        - avoidable_loss_penalties
        - battle_loss_penalty

All values below are *starting points*
meant to be tuned through experiments, not fixed requirements. They are
kept in one dataclass so a whole reward configuration can be versioned
and logged.

Guardrails encoded here:
  - win_reward is an order of magnitude larger than any single tactical
    reward, so intermediate farming can never out-value winning.
  - super-effective / knockout rewards are capped relative to win_reward
    so the agent can't learn "trade recklessly" is better than winning.
  - fainting a Pokemon unnecessarily and losing the battle are penalized
    independently of the opponent's fainted-count, so damage-race reward
    hacking doesn't pay off against a well-tuned config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    # Tactical (per-move) rewards
    effective_move: float = 0.15
    super_effective_move: float = 0.3
    extremely_effective_move: float = 0.45
    ineffective_move: float = -0.15
    disadvantageous_move: float = -0.3

    # Positional / strategic rewards
    favorable_switch: float = 0.2
    positional_advantage: float = 0.1
    avoidable_damage_taken: float = -0.1

    # Knockouts
    faint_opponent_pokemon: float = 1.0
    lose_own_pokemon_unnecessarily: float = -1.0

    # Terminal rewards -- must dominate the sum of plausible per-turn
    # tactical/positional/knockout rewards over a typical battle length.
    win_reward: float = 15.0
    loss_penalty: float = -15.0
    draw_reward: float = -1.0

    # Safety cap: no single non-terminal event should be worth more than
    # this fraction of the terminal win reward, to keep the agent from
    # learning "farm effective hits" instead of "win".
    max_non_terminal_fraction_of_terminal: float = 0.5

    def __post_init__(self) -> None:
        cap = self.max_non_terminal_fraction_of_terminal * self.win_reward
        largest_positive_event = max(
            self.effective_move,
            self.super_effective_move,
            self.extremely_effective_move,
            self.favorable_switch,
            self.positional_advantage,
            self.faint_opponent_pokemon,
        )
        if largest_positive_event > cap:
            raise ValueError(
                f"Reward event {largest_positive_event} exceeds "
                f"{self.max_non_terminal_fraction_of_terminal:.0%} of the "
                f"terminal win_reward ({self.win_reward}); tune the config "
                "so winning always dominates farming intermediate reward."
            )


def move_effectiveness_reward(multiplier: float, cfg: RewardConfig) -> float:
    """multiplier is the raw type-effectiveness multiplier (0, 0.25, 0.5, 1, 2, 4)."""
    if multiplier == 0:
        return cfg.disadvantageous_move  # move did nothing -- treat as a mistake
    if multiplier >= 4:
        return cfg.extremely_effective_move
    if multiplier >= 2:
        return cfg.super_effective_move
    if multiplier < 1:
        return cfg.ineffective_move
    return cfg.effective_move


@dataclass
class BattleRewardState:
    """Tracks per-battle counters needed to compute incremental rewards."""

    prev_own_fainted: int = 0
    prev_opp_fainted: int = 0
    prev_own_hp_total: float = 6.0
    prev_opp_hp_total: float = 6.0


def compute_reward(battle, prev_state: BattleRewardState, cfg: RewardConfig) -> tuple[float, BattleRewardState]:
    """
    Incremental reward for the transition that just produced `battle`
    (poke-env AbstractBattle, current state). Returns (reward, new_state)
    so the caller can carry `new_state` into the next step.
    """
    own_fainted = sum(1 for p in battle.team.values() if p.fainted)
    opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
    own_hp_total = sum(p.current_hp_fraction or 0.0 for p in battle.team.values())
    opp_hp_total = sum(p.current_hp_fraction or 0.0 for p in battle.opponent_team.values())

    reward = 0.0

    # Knockout rewards
    new_opp_faints = max(0, opp_fainted - prev_state.prev_opp_fainted)
    new_own_faints = max(0, own_fainted - prev_state.prev_own_fainted)
    reward += new_opp_faints * cfg.faint_opponent_pokemon
    reward += new_own_faints * cfg.lose_own_pokemon_unnecessarily

    # Positional reward: reward net HP swing in our favor this step
    # (a crude proxy for "damage dealt minus damage taken", i.e. avoidable
    # damage is implicitly penalized without needing to attribute blame).
    own_hp_delta = own_hp_total - prev_state.prev_own_hp_total
    opp_hp_delta = opp_hp_total - prev_state.prev_opp_hp_total
    net_hp_swing = opp_hp_delta - own_hp_delta  # negative opp_hp_delta is good for us
    reward += cfg.positional_advantage * max(0.0, -opp_hp_delta)
    if own_hp_delta < 0:
        reward += cfg.avoidable_damage_taken * min(1.0, -own_hp_delta)

    # Terminal rewards
    if battle.finished:
        if battle.won:
            reward += cfg.win_reward
        elif battle.lost:
            reward += cfg.loss_penalty
        else:
            reward += cfg.draw_reward

    new_state = BattleRewardState(
        prev_own_fainted=own_fainted,
        prev_opp_fainted=opp_fainted,
        prev_own_hp_total=own_hp_total,
        prev_opp_hp_total=opp_hp_total,
    )
    return reward, new_state
