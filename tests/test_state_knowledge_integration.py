"""
Integration tests between environment/state.py's encode_battle() and
knowledge/pokemon_knowledge.py's PokemonKnowledgeBase -- kept separate
from tests/test_state.py (which deliberately never passes a
knowledge_base, to guard the "same shape with or without one"
contract) and tests/test_pokemon_knowledge.py (which never touches
encode_battle at all).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from environment.state import encode_battle, observation_size, POKEMON_TYPES
from knowledge.pokemon_knowledge import PokemonKnowledgeBase


def _fake_type(name):
    return SimpleNamespace(name=name.upper())


def _fake_move(move_type="fire", base_power=80, accuracy=1.0, current_pp=10, max_pp=16):
    return SimpleNamespace(
        base_power=base_power,
        accuracy=accuracy,
        current_pp=current_pp,
        max_pp=max_pp,
        type=_fake_type(move_type),
    )


def _fake_pokemon(species, hp_frac=1.0, fainted=False, type1="fire", type2=None, moves=None):
    return SimpleNamespace(
        species=species,
        current_hp_fraction=hp_frac,
        fainted=fainted,
        status=None,
        type_1=_fake_type(type1) if type1 else None,
        type_2=_fake_type(type2) if type2 else None,
        moves=moves or {},
    )


def _fake_battle(own_team, opponent_team, moves=None, turn=1):
    return SimpleNamespace(
        team=own_team,
        opponent_team=opponent_team,
        active_pokemon=next(iter(own_team.values())) if own_team else None,
        opponent_active_pokemon=next(iter(opponent_team.values())) if opponent_team else None,
        available_moves=moves or [],
        weather={},
        side_conditions={},
        opponent_side_conditions={},
        turn=turn,
    )


def test_shape_identical_with_and_without_knowledge_base():
    battle = _fake_battle(
        {"p1": _fake_pokemon("Charizard")},
        {"o1": _fake_pokemon("Garchomp", type1="dragon", type2="ground")},
    )
    kb = PokemonKnowledgeBase()
    kb.observe_battle(battle)

    obs_without = encode_battle(battle)
    obs_with = encode_battle(battle, knowledge_base=kb)

    assert obs_without.shape == obs_with.shape == (observation_size(),)


def test_unpopulated_knowledge_base_yields_all_zero_knowledge_block():
    battle = _fake_battle(
        {"p1": _fake_pokemon("Charizard")},
        {"o1": _fake_pokemon("Garchomp", type1="dragon", type2="ground")},
    )
    kb = PokemonKnowledgeBase()  # never observed anything

    obs_none = encode_battle(battle, knowledge_base=None)
    obs_empty_kb = encode_battle(battle, knowledge_base=kb)

    np.testing.assert_array_equal(obs_none, obs_empty_kb)


def test_populated_knowledge_base_changes_the_encoding():
    """The whole point of the feature: once the knowledge base has
    actual history for a species, encoding a FRESH battle against that
    same species (a new individual, e.g. next battle's opponent) must
    differ from encoding it with no knowledge base at all."""
    # Battle 1 (a past battle): Garchomp reveals two Ground moves and
    # one Dragon move -- this is what gets learned.
    past_battle = _fake_battle(
        {},
        {"o1": _fake_pokemon(
            "Garchomp", type1="dragon", type2="ground",
            moves={
                "earthquake": _fake_move("ground"),
                "stealthrock": _fake_move("rock"),
                "outrage": _fake_move("dragon"),
            },
        )},
    )
    kb = PokemonKnowledgeBase()
    kb.observe_battle(past_battle)

    # Battle 2 (the CURRENT battle being encoded): a fresh Garchomp
    # individual, no moves revealed yet this battle.
    current_battle = _fake_battle(
        {"p1": _fake_pokemon("Charizard")},
        {"o1": _fake_pokemon("Garchomp", type1="dragon", type2="ground", moves={})},
    )

    obs_no_history = encode_battle(current_battle, knowledge_base=None)
    obs_with_history = encode_battle(current_battle, knowledge_base=kb)

    assert not np.array_equal(obs_no_history, obs_with_history)


def test_own_team_encoding_is_unaffected_by_knowledge_base():
    """The knowledge block only applies to the OPPONENT team -- our
    own team's block must be byte-identical regardless of what's in
    the knowledge base, since our own moves are already fully known."""
    battle = _fake_battle(
        {"p1": _fake_pokemon("Charizard", moves={})},
        {"o1": _fake_pokemon("Garchomp", type1="dragon", type2="ground")},
    )
    kb = PokemonKnowledgeBase()
    # Feed the knowledge base info about Charizard specifically (our
    # own active mon's species) to make sure it does NOT leak into the
    # own-team block even though the species matches.
    kb.observe_battle(_fake_battle({}, {"o1": _fake_pokemon(
        "Charizard", moves={"flamethrower": _fake_move("fire")}
    )}))

    obs_none = encode_battle(battle, knowledge_base=None)
    obs_kb = encode_battle(battle, knowledge_base=kb)

    # own team occupies the first MAX_TEAM_SIZE * _POKEMON_BLOCK entries;
    # import the same constants encode_battle itself uses.
    from environment.state import MAX_TEAM_SIZE
    own_block_size = len(obs_none) - 0
    # Recompute own-team span the same way observation_size() does.
    import environment.state as state_module
    own_span = state_module.MAX_TEAM_SIZE * state_module._POKEMON_BLOCK
    np.testing.assert_array_equal(obs_none[:own_span], obs_kb[:own_span])