"""
State encoding for the Pokemon battle RL environment.

Design principle:
  - Known information: directly observable by the agent (own team fully,
    opponent's revealed Pokemon/moves/items/abilities only).
  - Unknown information: anything poke-env has not revealed through the
    Showdown protocol (e.g. an opponent's un-revealed moveset) MUST NOT
    leak into the observation, even if it happens to be present on the
    underlying simulator/battle object.
  - Derived information: values legitimately computable from known state
    (e.g. type-effectiveness of a known move against a known type) are
    fine to include.
  - Learned information: knowledge/pokemon_knowledge.py's
    PokemonKnowledgeBase accumulates move/ability/item usage patterns
    for each species ACROSS MANY PAST BATTLES -- this is not "unknown
    information leaking in" (nothing about THIS battle's hidden state
    is exposed), it's the same kind of prior a human player builds up
    from experience with the metagame. Only applied to the OPPONENT
    team block: our own team's moves are already fully known from the
    battle object directly, so a historical estimate would only ever
    be redundant there.

poke-env's `AbstractBattle` already enforces the known/unknown boundary
for us: `opponent_team` and `opponent_active_pokemon` only ever contain
what has actually been revealed in the battle log. This encoder is
intentionally written to only read from that battle object's public,
protocol-derived attributes -- never anything injected for debugging/
testing that wouldn't exist in a real battle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    # Type-hint only -- NOT imported at runtime. knowledge/pokemon_knowledge.py
    # itself imports POKEMON_TYPES from this module, so a real top-level
    # import here would be circular. encode_battle() only ever calls
    # `.observation_features(species)` on whatever's passed in, so any
    # duck-typed object with that method works regardless.
    from knowledge.pokemon_knowledge import PokemonKnowledgeBase

# A fixed vocabulary of types keeps the observation vector a stable shape
# across battles/generations we care about (gen 9).
POKEMON_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "dark", "steel", "fairy",
]

STATUS_CONDITIONS = ["brn", "frz", "par", "psn", "slp", "tox", "fnt"]

MAX_MOVES = 4
MAX_TEAM_SIZE = 6

# One-pokemon feature block size, used for both own and opponent mons.
# [hp_frac, fainted, *status_onehot(7), *types_onehot(18)*2 (type1/type2)]
_POKEMON_BLOCK = 1 + 1 + len(STATUS_CONDITIONS) + 2 * len(POKEMON_TYPES)

# Knowledge-base-derived block appended ONLY to opponent Pokemon slots
# (see module docstring): [move_type_coverage(18), confidence(1)] --
# always present and zero-filled when no knowledge_base is given or the
# species has never been observed before, so the overall observation
# shape never depends on whether a knowledge base is in use.
_KNOWLEDGE_BLOCK = len(POKEMON_TYPES) + 1

# Per-move feature block: [base_power_norm, accuracy_norm, pp_frac,
# *type_onehot(18), effectiveness_vs_opponent]
_MOVE_BLOCK = 3 + len(POKEMON_TYPES) + 1

FIELD_EFFECTS = [
    "reflect", "lightscreen", "auroraveil", "spikes", "stealthrock",
    "toxicspikes", "stickyweb",
]

WEATHERS = ["sunnyday", "raindance", "sandstorm", "hail", "snow", "none"]


def _one_hot(value: str | None, vocabulary: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocabulary), dtype=np.float32)
    if value is None:
        return vec
    value = value.lower()
    if value in vocabulary:
        vec[vocabulary.index(value)] = 1.0
    return vec


def _pokemon_features(pokemon) -> np.ndarray:
    """Encode a single, known Pokemon (own or a revealed opponent mon)."""
    if pokemon is None:
        return np.zeros(_POKEMON_BLOCK, dtype=np.float32)

    hp_frac = pokemon.current_hp_fraction if pokemon.current_hp_fraction is not None else 0.0
    fainted = 1.0 if pokemon.fainted else 0.0
    status = pokemon.status.name.lower() if pokemon.status else None
    status_vec = _one_hot(status, STATUS_CONDITIONS)

    type1 = pokemon.type_1.name.lower() if pokemon.type_1 else None
    type2 = pokemon.type_2.name.lower() if pokemon.type_2 else None
    type1_vec = _one_hot(type1, POKEMON_TYPES)
    type2_vec = _one_hot(type2, POKEMON_TYPES)

    return np.concatenate(
        [[hp_frac, fainted], status_vec, type1_vec, type2_vec]
    ).astype(np.float32)


def _opponent_knowledge_features(pokemon, knowledge_base: Optional["PokemonKnowledgeBase"]) -> np.ndarray:
    """
    Historical move-type coverage + confidence for a revealed opponent
    Pokemon, from PokemonKnowledgeBase.observation_features(). Zero-
    filled whenever there's nothing to say: no Pokemon in this slot, no
    knowledge base configured, or a species this knowledge base has
    never seen before (a brand-new knowledge base starts every species
    at all-zero, and confidence only rises as more battles are
    recorded -- see knowledge/pokemon_knowledge.py).
    """
    if pokemon is None or knowledge_base is None:
        return np.zeros(_KNOWLEDGE_BLOCK, dtype=np.float32)

    species = getattr(pokemon, "species", None)
    if not species:
        return np.zeros(_KNOWLEDGE_BLOCK, dtype=np.float32)

    coverage, confidence = knowledge_base.observation_features(species)
    return np.concatenate([coverage, [confidence]]).astype(np.float32)


def _move_features(move, opponent_pokemon) -> np.ndarray:
    if move is None:
        return np.zeros(_MOVE_BLOCK, dtype=np.float32)

    base_power = (move.base_power or 0) / 250.0
    accuracy = move.accuracy if move.accuracy is not None else 1.0
    pp_frac = (move.current_pp / move.max_pp) if getattr(move, "max_pp", 0) else 0.0
    move_type = move.type.name.lower() if move.type else None
    type_vec = _one_hot(move_type, POKEMON_TYPES)

    # Derived information: effectiveness is legitimately computable from
    # known move type + known/revealed opponent type(s).
    effectiveness = 1.0
    if opponent_pokemon is not None and move.type is not None:
        try:
            effectiveness = opponent_pokemon.damage_multiplier(move) / 4.0
        except Exception:
            effectiveness = 0.25  # neutral (1x) normalized, fallback

    return np.concatenate(
        [[base_power, accuracy, pp_frac], type_vec, [effectiveness]]
    ).astype(np.float32)


def _field_features(battle) -> np.ndarray:
    weather_name = None
    if battle.weather:
        # battle.weather is a dict {Weather: turn_started}; take the active one
        weather_name = next(iter(battle.weather)).name.lower()
    weather_vec = _one_hot(weather_name, WEATHERS)

    own_side = np.zeros(len(FIELD_EFFECTS), dtype=np.float32)
    opp_side = np.zeros(len(FIELD_EFFECTS), dtype=np.float32)
    for i, effect in enumerate(FIELD_EFFECTS):
        for cond in battle.side_conditions:
            if effect in cond.name.lower():
                own_side[i] = 1.0
        for cond in battle.opponent_side_conditions:
            if effect in cond.name.lower():
                opp_side[i] = 1.0

    turn_norm = min(battle.turn / 100.0, 1.0)

    return np.concatenate([weather_vec, own_side, opp_side, [turn_norm]]).astype(np.float32)


def observation_size() -> int:
    team_block = MAX_TEAM_SIZE * _POKEMON_BLOCK  # own team
    opp_block = MAX_TEAM_SIZE * (_POKEMON_BLOCK + _KNOWLEDGE_BLOCK)  # opponent team + learned knowledge
    move_block = MAX_MOVES * _MOVE_BLOCK
    field_block = len(WEATHERS) + 2 * len(FIELD_EFFECTS) + 1
    return team_block + opp_block + move_block + field_block


def encode_battle(battle, knowledge_base: Optional["PokemonKnowledgeBase"] = None) -> np.ndarray:
    """
    Encode a poke-env `AbstractBattle` into a fixed-length float32 vector.

    Only reads attributes that poke-env populates from the actual Showdown
    protocol stream, so hidden/unrevealed opponent information is never
    included by construction.

    `knowledge_base`, if given, enriches each revealed OPPONENT Pokemon's
    block with that species' historically-learned move-type coverage
    (see knowledge/pokemon_knowledge.py). Passing None (the default)
    zero-fills that portion of the vector instead -- the output SHAPE
    is identical either way, only the values differ, so callers that
    don't have a knowledge base handy (e.g. existing tests) still get a
    correctly-shaped observation.
    """
    own_team = list(battle.team.values())
    own_team_feats = [
        _pokemon_features(own_team[i]) if i < len(own_team) else np.zeros(_POKEMON_BLOCK, dtype=np.float32)
        for i in range(MAX_TEAM_SIZE)
    ]

    opp_team = list(battle.opponent_team.values())
    opp_team_feats = []
    for i in range(MAX_TEAM_SIZE):
        opp_pokemon = opp_team[i] if i < len(opp_team) else None
        base = _pokemon_features(opp_pokemon)
        knowledge = _opponent_knowledge_features(opp_pokemon, knowledge_base)
        opp_team_feats.append(np.concatenate([base, knowledge]).astype(np.float32))

    active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    moves = list(battle.available_moves) if battle.available_moves else []
    move_feats = [
        _move_features(moves[i], opp_active) if i < len(moves) else np.zeros(_MOVE_BLOCK, dtype=np.float32)
        for i in range(MAX_MOVES)
    ]

    field_feats = _field_features(battle)

    return np.concatenate(
        own_team_feats + opp_team_feats + move_feats + [field_feats]
    ).astype(np.float32)