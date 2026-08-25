"""
TeamPool: a poke-env Teambuilder that samples a random team from a
fixed pool of pre-built teams each time a Player needs one.

Random-battle formats (gen9randombattle, Stage 1-2) get team variety
for free -- poke-env's Showdown server generates a fresh random team
every battle. Non-random formats like gen9ou (Stage 3+) don't: without
an explicit team, a Player either can't battle at all, or (if given a
single fixed team string) always uses the exact same team every
battle, which defeats CLAUDE.md's "varied teams and battle scenarios"
requirement once training moves past Stage 2. TeamPool fixes that by
holding several teams and picking one at random per battle, the same
way a human laddering with a small roster of prepared teams would.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from poke_env.teambuilder import Teambuilder


class TeamPool(Teambuilder):
    """
    A Teambuilder over a fixed pool of teams. Each team is parsed once
    at construction time (accepting either Showdown "export" format --
    the copy-pasteable text from Showdown's own teambuilder -- or
    packed format, auto-detected by the presence of "|"), so
    yield_team() is cheap to call every battle.
    """

    def __init__(self, teams: list[str], seed: Optional[int] = None):
        if not teams:
            raise ValueError("TeamPool requires at least one team.")

        self._teams: list[str] = []
        for team_text in teams:
            mons = (
                self.parse_packed_team(team_text)
                if "|" in team_text
                else self.parse_showdown_team(team_text)
            )
            if not mons:
                raise ValueError(f"Could not parse a non-empty team from:\n{team_text!r}")
            self._teams.append(self.join_team(mons))

        self._rng = random.Random(seed)

    @classmethod
    def from_directory(cls, team_dir: str | Path, pattern: str = "*.txt", seed: Optional[int] = None) -> "TeamPool":
        """
        Loads every file matching `pattern` in `team_dir` as one team
        (Showdown export or packed format, one team per file). Team
        files are the natural way to keep a pool under version control
        -- e.g. `configs/teams/stage3_ou/*.txt` -- separately from
        training code, and to swap in a genuinely held-out set of teams
        for evaluation (CLAUDE.md's "held-out ... teams" requirement).
        """
        team_dir = Path(team_dir)
        paths = sorted(team_dir.glob(pattern))
        if not paths:
            raise ValueError(f"No team files matching {pattern!r} found in {team_dir}")
        teams = [p.read_text() for p in paths]
        return cls(teams, seed=seed)

    def yield_team(self) -> str:
        return self._rng.choice(self._teams)

    def __len__(self) -> int:
        return len(self._teams)
