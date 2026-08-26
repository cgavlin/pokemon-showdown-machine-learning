"""
Evaluation metrics: win/loss/draw rate, average reward,
average damage dealt/received, Pokemon fainted per battle (both sides), 
switch frequency, move effectiveness, and
(via benchmarks.py) performance against held-out opponents/teams.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BattleMetrics:
    won: bool
    lost: bool
    drew: bool
    total_reward: float
    own_fainted: int
    opponent_fainted: int
    own_damage_dealt_fraction: float
    own_damage_taken_fraction: float
    n_switches: int
    n_moves: int
    n_super_effective_moves: int
    n_ineffective_moves: int
    turns: int


@dataclass
class AggregateMetrics:
    n_battles: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_reward: float = 0.0
    own_fainted: int = 0
    opponent_fainted: int = 0
    damage_dealt_fraction: float = 0.0
    damage_taken_fraction: float = 0.0
    n_switches: int = 0
    n_moves: int = 0
    n_super_effective_moves: int = 0
    n_ineffective_moves: int = 0
    turns: int = 0

    def add(self, battle: BattleMetrics) -> None:
        self.n_battles += 1
        self.wins += int(battle.won)
        self.losses += int(battle.lost)
        self.draws += int(battle.drew)
        self.total_reward += battle.total_reward
        self.own_fainted += battle.own_fainted
        self.opponent_fainted += battle.opponent_fainted
        self.damage_dealt_fraction += battle.own_damage_dealt_fraction
        self.damage_taken_fraction += battle.own_damage_taken_fraction
        self.n_switches += battle.n_switches
        self.n_moves += battle.n_moves
        self.n_super_effective_moves += battle.n_super_effective_moves
        self.n_ineffective_moves += battle.n_ineffective_moves
        self.turns += battle.turns

    def summary(self) -> dict:
        n = max(1, self.n_battles)
        return {
            "n_battles": self.n_battles,
            "win_rate": self.wins / n,
            "loss_rate": self.losses / n,
            "draw_rate": self.draws / n,
            "avg_reward": self.total_reward / n,
            "avg_own_fainted": self.own_fainted / n,
            "avg_opponent_fainted": self.opponent_fainted / n,
            "avg_damage_dealt_fraction": self.damage_dealt_fraction / n,
            "avg_damage_taken_fraction": self.damage_taken_fraction / n,
            "avg_switches_per_battle": self.n_switches / n,
            "move_effectiveness_rate": (
                self.n_super_effective_moves / max(1, self.n_moves)
            ),
            "move_ineffectiveness_rate": (
                self.n_ineffective_moves / max(1, self.n_moves)
            ),
            "avg_turns": self.turns / n,
        }
