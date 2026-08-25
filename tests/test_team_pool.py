"""
Tests for environment/team_pool.py's TeamPool.
"""

from __future__ import annotations

import pytest

from environment.team_pool import TeamPool

_TEAM_1 = """Landorus-Therian @ Choice Scarf
Ability: Intimidate
Tera Type: Flying
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- U-turn
- Stone Edge
- Stealth Rock
"""

_TEAM_2 = """Great Tusk @ Leftovers
Ability: Protosynthesis
Tera Type: Water
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Rapid Spin
- Earthquake
- Close Combat
- Ice Spinner
"""


def test_yield_team_returns_packed_format():
    pool = TeamPool([_TEAM_1], seed=0)
    team = pool.yield_team()
    assert isinstance(team, str)
    assert "|" in team  # packed format uses "|" as a field separator


def test_yield_team_samples_from_the_whole_pool():
    pool = TeamPool([_TEAM_1, _TEAM_2], seed=0)
    samples = {pool.yield_team() for _ in range(50)}
    assert len(samples) == 2


def test_len_matches_number_of_teams():
    pool = TeamPool([_TEAM_1, _TEAM_2], seed=0)
    assert len(pool) == 2


def test_empty_team_list_raises():
    with pytest.raises(ValueError, match="at least one team"):
        TeamPool([])


def test_from_directory_loads_one_team_per_file(tmp_path):
    (tmp_path / "a.txt").write_text(_TEAM_1)
    (tmp_path / "b.txt").write_text(_TEAM_2)
    pool = TeamPool.from_directory(tmp_path, seed=0)
    assert len(pool) == 2


def test_from_directory_raises_on_no_matching_files(tmp_path):
    with pytest.raises(ValueError, match="No team files"):
        TeamPool.from_directory(tmp_path)


def test_from_directory_respects_custom_pattern(tmp_path):
    (tmp_path / "a.team").write_text(_TEAM_1)
    (tmp_path / "b.txt").write_text(_TEAM_2)  # should be ignored
    pool = TeamPool.from_directory(tmp_path, pattern="*.team", seed=0)
    assert len(pool) == 1


def test_seed_makes_sampling_deterministic():
    pool_a = TeamPool([_TEAM_1, _TEAM_2], seed=42)
    pool_b = TeamPool([_TEAM_1, _TEAM_2], seed=42)
    sequence_a = [pool_a.yield_team() for _ in range(10)]
    sequence_b = [pool_b.yield_team() for _ in range(10)]
    assert sequence_a == sequence_b
