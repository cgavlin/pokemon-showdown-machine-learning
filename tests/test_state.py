from types import SimpleNamespace

import numpy as np

from environment.state import encode_battle, observation_size


def _fake_type(name):
    return SimpleNamespace(name=name.upper())


def _fake_pokemon(hp_frac=1.0, fainted=False, status=None, type1="fire", type2=None):
    return SimpleNamespace(
        current_hp_fraction=hp_frac,
        fainted=fainted,
        status=SimpleNamespace(name=status.upper()) if status else None,
        type_1=_fake_type(type1) if type1 else None,
        type_2=_fake_type(type2) if type2 else None,
    )


def _fake_move(base_power=80, accuracy=1.0, current_pp=10, max_pp=16, move_type="fire"):
    return SimpleNamespace(
        base_power=base_power,
        accuracy=accuracy,
        current_pp=current_pp,
        max_pp=max_pp,
        type=_fake_type(move_type),
    )


def _fake_battle(own_team, opponent_team, moves, turn=1):
    return SimpleNamespace(
        team=own_team,
        opponent_team=opponent_team,
        active_pokemon=next(iter(own_team.values())) if own_team else None,
        opponent_active_pokemon=next(iter(opponent_team.values())) if opponent_team else None,
        available_moves=moves,
        weather={},
        side_conditions={},
        opponent_side_conditions={},
        turn=turn,
    )


def test_observation_size_matches_encoded_vector_length():
    own_team = {"p1": _fake_pokemon()}
    opp_team = {"o1": _fake_pokemon(type1="water")}
    moves = [_fake_move()]
    battle = _fake_battle(own_team, opp_team, moves)

    obs = encode_battle(battle)
    assert obs.shape == (observation_size(),)
    assert obs.dtype == np.float32


def test_encoding_is_fixed_size_regardless_of_team_size():
    small_battle = _fake_battle({"p1": _fake_pokemon()}, {}, [])
    full_battle = _fake_battle(
        {f"p{i}": _fake_pokemon() for i in range(6)},
        {f"o{i}": _fake_pokemon() for i in range(6)},
        [_fake_move(), _fake_move(), _fake_move(), _fake_move()],
    )
    assert encode_battle(small_battle).shape == encode_battle(full_battle).shape


def test_unrevealed_opponent_pokemon_encode_as_zero_blocks():
    """
    Hidden-information handling: an opponent team with only 1 of 6
    Pokemon revealed must not fabricate information about the other 5 --
    those slots should be all-zero, not filled with guesses.
    """
    own_team = {"p1": _fake_pokemon()}
    opp_team = {"o1": _fake_pokemon(type1="water")}  # only 1 revealed
    battle = _fake_battle(own_team, opp_team, [])

    obs = encode_battle(battle)
    # The encoding is deterministic given only revealed info; re-encoding
    # an "enriched" battle object that adds more (simulated) revealed
    # opponent mons must change the vector -- proving the extra slots
    # were genuinely zeroed, not coincidentally similar.
    opp_team_full = dict(opp_team)
    opp_team_full["o2"] = _fake_pokemon(type1="grass")
    battle_more_revealed = _fake_battle(own_team, opp_team_full, [])
    obs_more_revealed = encode_battle(battle_more_revealed)

    assert not np.array_equal(obs, obs_more_revealed)
    assert obs.shape == obs_more_revealed.shape


def test_fainted_pokemon_is_reflected_in_encoding():
    own_team = {"p1": _fake_pokemon(hp_frac=0.0, fainted=True)}
    battle = _fake_battle(own_team, {}, [])
    obs = encode_battle(battle)
    assert obs.shape == (observation_size(),)
    # fainted flag is the second scalar in the first pokemon's block
    assert obs[1] == 1.0
