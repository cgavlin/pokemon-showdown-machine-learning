"""
Curriculum learning stages: (Stage 1: Basic Mechanics ... Stage 5: Human Opponents).

Each CurriculumStage describes what opponent pool / battle format /
team pool to train against and a rough gate (min win rate over N
evaluation battles) an agent should clear before advancing. Gates are
starting points for experimentation, not hard requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)


@dataclass
class CurriculumStage:
    name: str
    description: str
    battle_format: str
    opponent_factory: str  # key into OPPONENT_FACTORIES
    min_eval_win_rate_to_advance: float
    min_battles_before_eval: int


def _random_opponent(battle_format: str, team: str | None) -> Player:
    return RandomPlayer(battle_format=battle_format, team=team)


def _max_power_opponent(battle_format: str, team: str | None) -> Player:
    return MaxBasePowerPlayer(battle_format=battle_format, team=team)


def _heuristic_opponent(battle_format: str, team: str | None) -> Player:
    return SimpleHeuristicsPlayer(battle_format=battle_format, team=team)


OPPONENT_FACTORIES = {
    "random": _random_opponent,
    "max_base_power": _max_power_opponent,
    "heuristic": _heuristic_opponent,
}


CURRICULUM: list[CurriculumStage] = [
    CurriculumStage(
        name="stage1_basic_mechanics",
        description=(
            "Type matchups, move effectiveness, basic damage, fainting, "
            "switching, status effects, win/loss conditions."
        ),
        battle_format="gen9randombattle",
        opponent_factory="random",
        min_eval_win_rate_to_advance=0.60,
        min_battles_before_eval=2000,
    ),
    CurriculumStage(
        name="stage2_tactical_decisions",
        description=(
            "Move selection, favorable switching, opponent-action "
            "prediction, HP management, avoiding unnecessary sacrifices."
        ),
        battle_format="gen9randombattle",
        opponent_factory="max_base_power",
        min_eval_win_rate_to_advance=0.55,
        min_battles_before_eval=3000,
    ),
    CurriculumStage(
        name="stage3_strategic_play",
        description=(
            "Team composition, hazards, weather, status management, "
            "setup/disruption, resource management, long-term positioning."
        ),
        battle_format="gen9ou",
        opponent_factory="heuristic",
        min_eval_win_rate_to_advance=0.50,
        min_battles_before_eval=5000,
    ),
    CurriculumStage(
        name="stage4_competitive_play",
        description=(
            "Diverse teams, opponent-set uncertainty, prediction, "
            "advanced switching, stronger AI, self-play."
        ),
        battle_format="gen9ou",
        opponent_factory="heuristic",  # self-play pool layered on top; see self_play.py
        min_eval_win_rate_to_advance=0.55,
        min_battles_before_eval=8000,
    ),
    CurriculumStage(
        name="stage5_human_opponents",
        description=(
            "Evaluation against real players. Requires the explicit "
            "live-battle safety gate in showdown/integration.py."
        ),
        battle_format="gen9ou",
        opponent_factory="heuristic",  # placeholder; live games are human, not scripted
        min_eval_win_rate_to_advance=1.0,  # no auto-advance past this stage
        min_battles_before_eval=0,
    ),
]


def get_stage(name: str) -> CurriculumStage:
    for stage in CURRICULUM:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown curriculum stage: {name}")


def make_opponent(stage: CurriculumStage, team: str | None = None) -> Player:
    factory = OPPONENT_FACTORIES[stage.opponent_factory]
    return factory(stage.battle_format, team)
