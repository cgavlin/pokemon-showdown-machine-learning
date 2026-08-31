from types import SimpleNamespace

from knowledge.pokemon_knowledge import PokemonKnowledgeBase, type_effectiveness


def _fake_type(name):
    return SimpleNamespace(name=name.upper())


def _fake_pokemon(species, type1="water", type2=None, ability=None, item=None, moves=None):
    return SimpleNamespace(
        species=species,
        type_1=_fake_type(type1) if type1 else None,
        type_2=_fake_type(type2) if type2 else None,
        ability=ability,
        item=item,
        moves=moves or {},
    )


def _fake_battle(own_team, opponent_team):
    return SimpleNamespace(team=own_team, opponent_team=opponent_team)


# --- type_effectiveness ----------------------------------------------------


def test_type_effectiveness_single_type_weakness():
    assert type_effectiveness("electric", ["water"]) == 2.0


def test_type_effectiveness_single_type_resistance():
    assert type_effectiveness("water", ["water"]) == 0.5


def test_type_effectiveness_immunity():
    assert type_effectiveness("normal", ["ghost"]) == 0.0


def test_type_effectiveness_dual_type_multiplies():
    # Water/Ground (Quagsire): Electric is 2.0 vs
    # water but 0.0 vs ground -> immune overall.
    assert type_effectiveness("electric", ["water", "ground"]) == 0.0


def test_type_effectiveness_unlisted_pair_is_neutral():
    assert type_effectiveness("normal", ["normal"]) == 1.0


# --- observe_battle / species_summary ---------------------------------------


def test_unseen_species_summary_is_none():
    kb = PokemonKnowledgeBase()
    assert kb.species_summary("pikachu") is None


def test_observe_battle_records_types_and_derives_weaknesses():
    battle = _fake_battle(
        own_team={},
        opponent_team={"o1": _fake_pokemon("Quagsire", type1="water", type2="ground")},
    )
    kb = PokemonKnowledgeBase()
    kb.observe_battle(battle)

    summary = kb.species_summary("quagsire")
    assert summary["types"] == ["water", "ground"]
    assert summary["battles_seen"] == 1
    # Water/Ground is immune to Electric, not merely resistant.
    assert "electric" in summary["immunities"]
    assert "electric" not in summary["weaknesses"]
    # Grass is super-effective against both Water and Ground.
    assert summary["weaknesses"]["grass"] == 4.0


def test_species_lookup_is_case_insensitive():
    battle = _fake_battle({}, {"o1": _fake_pokemon("Pikachu", type1="electric", type2=None)})
    kb = PokemonKnowledgeBase()
    kb.observe_battle(battle)
    assert kb.species_summary("PIKACHU") is not None
    assert kb.species_summary("pikachu")["types"] == ["electric"]


def test_moves_abilities_items_accumulate_across_multiple_battles():
    """Core requirement: knowledge grows across separate battles, not
    just within one -- distinct observe_battle() calls simulate
    distinct battles."""
    kb = PokemonKnowledgeBase()

    battle1 = _fake_battle(
        {},
        {"o1": _fake_pokemon(
            "Landorus-Therian", type1="ground", type2="flying",
            ability="Intimidate", item="Choice Scarf",
            moves={"earthquake": object(), "uturn": object()},
        )},
    )
    kb.observe_battle(battle1)

    battle2 = _fake_battle(
        {},
        {"o1": _fake_pokemon(
            "Landorus-Therian", type1="ground", type2="flying",
            ability="Intimidate", item="Leftovers",
            moves={"earthquake": object(), "stoneedge": object()},
        )},
    )
    kb.observe_battle(battle2)

    summary = kb.species_summary("landorus-therian")
    assert summary["battles_seen"] == 2
    # earthquake seen in both battles -> count 2; the others once each.
    assert summary["move_counts"]["earthquake"] == 2
    assert summary["move_counts"]["uturn"] == 1
    assert summary["move_counts"]["stoneedge"] == 1
    assert summary["known_moves_by_frequency"][0] == "earthquake"
    assert summary["most_common_ability"] == "Intimidate"
    # Item differed between battles (Scarf vs Leftovers) -- both should
    # be tracked, not just the latest.
    assert set(kb._records["landorus-therian"].item_counts) == {"Choice Scarf", "Leftovers"}


def test_observe_battle_records_both_sides():
    battle = _fake_battle(
        own_team={"p1": _fake_pokemon("Charizard", type1="fire", type2="flying")},
        opponent_team={"o1": _fake_pokemon("Blastoise", type1="water", type2=None)},
    )
    kb = PokemonKnowledgeBase()
    kb.observe_battle(battle)

    assert kb.species_summary("charizard") is not None
    assert kb.species_summary("blastoise") is not None
    assert len(kb) == 2


def test_unrevealed_ability_and_item_are_not_recorded():
    battle = _fake_battle({}, {"o1": _fake_pokemon("Ditto", type1="normal", ability=None, item=None)})
    kb = PokemonKnowledgeBase()
    kb.observe_battle(battle)

    summary = kb.species_summary("ditto")
    assert summary["most_common_ability"] is None
    assert summary["most_common_item"] is None


# --- save / load -------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "knowledge.json"
    kb = PokemonKnowledgeBase(path=path)

    battle = _fake_battle(
        {},
        {"o1": _fake_pokemon(
            "Garchomp", type1="dragon", type2="ground",
            ability="Rough Skin", item="Life Orb",
            moves={"outrage": object()},
        )},
    )
    kb.observe_battle(battle)
    kb.save()

    reloaded = PokemonKnowledgeBase(path=path)
    summary = reloaded.species_summary("garchomp")
    assert summary["types"] == ["dragon", "ground"]
    assert summary["move_counts"] == {"outrage": 1}
    assert summary["most_common_ability"] == "Rough Skin"


def test_loading_a_missing_path_starts_empty(tmp_path):
    kb = PokemonKnowledgeBase(path=tmp_path / "does_not_exist_yet.json")
    assert len(kb) == 0


def test_save_without_any_path_raises():
    kb = PokemonKnowledgeBase()
    try:
        kb.save()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_known_species_returns_sorted_list():
    kb = PokemonKnowledgeBase()
    kb.observe_battle(_fake_battle({}, {
        "o1": _fake_pokemon("Zapdos", type1="electric", type2="flying"),
        "o2": _fake_pokemon("Articuno", type1="ice", type2="flying"),
    }))
    assert kb.known_species() == ["articuno", "zapdos"]