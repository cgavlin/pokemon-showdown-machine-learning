"""
Cross-battle Pokemon knowledge base.

Unlike environment/state.py's per-battle observation encoding (which
only ever reflects what's revealed *within the current battle*, by
design -- see that module's docstring on the known/unknown boundary),
this module accumulates what the agent has learned across MANY
battles: which moves, abilities, and items a given species has
actually been seen carrying, plus that species' current typing (and
the weaknesses/resistances/immunities derived from it).

Two very different kinds of "knowledge" live here:
  - Typing -> weaknesses/resistances/immunities: fully determined by
    the game's static type chart (TYPE_CHART below) the moment a
    species' type(s) are known. Nothing to learn here; this is a
    lookup, included because it's useful alongside the learned data.
  - Move/ability/item usage: genuinely learned empirically -- the more
    battles the agent plays, the better its picture of what a species
    is likely to be carrying, which is exactly the kind of prior
    real players build up from experience (and unrevealed opponent
    moves/items/abilities are never available witinh a single battle
    otherwise).

TODO: Wire into environment/state.py's observation encoding: 
doing so changes observation_size() and would need an
ENVIRONMENT_VERSION bump, invalidating existing checkpoints.
This module is usable standalone (e.g. for inspection, analysis, or 
a future explicit wiring step) without touching training in the meantime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from environment.state import POKEMON_TYPES

# Standard type chart (attacker -> {defender: multiplier}). Only
# non-1.0 entries are listed; unlisted attacker/defender pairs are
# neutral (1.0), handled by the .get(..., 1.0) lookups below.
TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0,
             "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0,
              "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0,
                 "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5,
              "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0,
              "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0,
            "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5,
                 "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0,
                 "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5,
               "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0,
               "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0,
               "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0,
                "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5,
            "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0,
            "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5,
             "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5,
             "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0,
              "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0,
              "dark": 2.0, "steel": 0.5},
}


def type_effectiveness(attacking_type: str, defending_types: list[str]) -> float:
    """Combined multiplier of `attacking_type` against a (1 or 2)
    defending typing -- multiplicative, matching how dual typing works
    in-game."""
    row = TYPE_CHART.get(attacking_type.lower(), {})
    multiplier = 1.0
    for defending_type in defending_types:
        multiplier *= row.get(defending_type.lower(), 1.0)
    return multiplier


@dataclass
class SpeciesRecord:
    types: list[str] = field(default_factory=list)
    move_counts: dict[str, int] = field(default_factory=dict)
    ability_counts: dict[str, int] = field(default_factory=dict)
    item_counts: dict[str, int] = field(default_factory=dict)
    battles_seen: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SpeciesRecord":
        return cls(**data)


class PokemonKnowledgeBase:
    """
    A growing, on-disk record of what's been observed about each
    species across every battle this knowledge base has seen -- not
    just the current one. Safe to point multiple training runs at the
    same path to build one shared, ever-improving knowledge base (see
    training/self_play.py's SelfPlayPool for the equivalent pattern
    with checkpoints).
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._records: dict[str, SpeciesRecord] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self._records = {species: SpeciesRecord.from_dict(rec) for species, rec in data.items()}

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No path given to save the knowledge base to.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {species: record.to_dict() for species, record in self._records.items()}
        target.write_text(json.dumps(payload, indent=2))

    def observe_battle(self, battle) -> None:
        """
        Call once per FINISHED battle (not every step -- battles_seen
        below counts battles, not turns) with a poke-env AbstractBattle.
        Records every currently-revealed Pokemon on both sides. Only
        reads attributes poke-env actually populates from the real
        protocol stream (same known/unknown boundary environment/
        state.py relies on), so this never records anything that
        wasn't legitimately revealed in-battle.
        """
        for pokemon in list(battle.team.values()) + list(battle.opponent_team.values()):
            self._observe_pokemon(pokemon)

    def _observe_pokemon(self, pokemon) -> None:
        species = getattr(pokemon, "species", None)
        if not species:
            return
        species = species.lower()
        record = self._records.setdefault(species, SpeciesRecord())
        record.battles_seen += 1

        type1 = pokemon.type_1.name.lower() if getattr(pokemon, "type_1", None) else None
        type2 = pokemon.type_2.name.lower() if getattr(pokemon, "type_2", None) else None
        types = [t for t in (type1, type2) if t]
        if types:
            record.types = types  # keep the latest known typing (e.g. after Terastallize)

        ability = getattr(pokemon, "ability", None)
        if ability:
            record.ability_counts[ability] = record.ability_counts.get(ability, 0) + 1

        item = getattr(pokemon, "item", None)
        if item:
            record.item_counts[item] = record.item_counts.get(item, 0) + 1

        for move_id in (getattr(pokemon, "moves", None) or {}):
            record.move_counts[move_id] = record.move_counts.get(move_id, 0) + 1

    def species_summary(self, species: str) -> dict | None:
        """Returns None for a species never observed. Otherwise: known
        typing, derived weaknesses/resistances/immunities, and
        moves/ability/item ranked by how often they've actually been
        seen on this species so far."""
        record = self._records.get(species.lower())
        if record is None:
            return None

        weaknesses, resistances, immunities = {}, {}, []
        for attacking_type in POKEMON_TYPES:
            eff = type_effectiveness(attacking_type, record.types)
            if eff == 0:
                immunities.append(attacking_type)
            elif eff > 1.0:
                weaknesses[attacking_type] = eff
            elif eff < 1.0:
                resistances[attacking_type] = eff

        top_moves = sorted(record.move_counts.items(), key=lambda kv: kv[1], reverse=True)
        top_ability = max(record.ability_counts.items(), key=lambda kv: kv[1])[0] if record.ability_counts else None
        top_item = max(record.item_counts.items(), key=lambda kv: kv[1])[0] if record.item_counts else None

        return {
            "species": species.lower(),
            "types": record.types,
            "battles_seen": record.battles_seen,
            "weaknesses": weaknesses,
            "resistances": resistances,
            "immunities": immunities,
            "known_moves_by_frequency": [move for move, _ in top_moves],
            "move_counts": dict(record.move_counts),
            "most_common_ability": top_ability,
            "most_common_item": top_item,
        }

    def known_species(self) -> list[str]:
        return sorted(self._records)

    def __len__(self) -> int:
        return len(self._records)