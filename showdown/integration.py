"""
Live Pokemon Showdown integration -- the ONLY place in this codebase
that is allowed to connect to a non-local server or the public ladder.

Per CLAUDE.md's "Safety and Operational Rules":
  - Never allow an experimental model to automatically enter live
    battles without an explicit evaluation gate.
  - Keep live-battle credentials outside source code and version control.
  - Use environment variables / a secret-management mechanism for creds.
  - Add an explicit configuration flag or equivalent safety gate before
    enabling live battles.
  - Default to local simulation.

This module implements that gate as `LiveBattleConfig.enabled`, which
must be explicitly and manually set to True (there is no default-on
path, no CLI flag default, and no code path that flips it automatically
based on training progress or curriculum stage).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

logger = logging.getLogger(__name__)

SHOWDOWN_PUBLIC_SERVER = ServerConfiguration(
    "wss://sim3.psim.us/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


class LiveBattleNotEnabledError(RuntimeError):
    """Raised whenever code attempts to start a live battle without the
    explicit safety gate being set. This should never be caught and
    silently bypassed."""


@dataclass
class LiveBattleConfig:
    # Hard-required, explicit opt-in. There is intentionally no
    # environment-variable or config-file default of True anywhere in
    # this repository -- a human must pass enabled=True by hand, e.g.
    # from a one-off script, never from a checked-in config file.
    enabled: bool = False

    battle_format: str = "gen9ou"
    max_concurrent_battles: int = 1
    n_battles: int = 1

    # Credentials must come from the environment, never from source
    # control (CLAUDE.md: "Keep live-battle credentials and
    # authentication information outside source code and version
    # control").
    username_env_var: str = "SHOWDOWN_USERNAME"
    password_env_var: str = "SHOWDOWN_PASSWORD"


def _load_account_configuration(cfg: LiveBattleConfig) -> AccountConfiguration:
    username = os.environ.get(cfg.username_env_var)
    password = os.environ.get(cfg.password_env_var)
    if not username:
        raise RuntimeError(
            f"Live battles require a username in the {cfg.username_env_var} "
            "environment variable. Refusing to proceed with a default/"
            "anonymous account for live play."
        )
    return AccountConfiguration(username, password)


def connect_for_live_battles(
    player: Player,
    cfg: LiveBattleConfig,
    require_manual_confirmation: bool = True,
) -> Player:
    """
    The single choke point for connecting an agent to a live Showdown
    server. Every caller (scripts, notebooks, ad-hoc experiments) must
    go through this function -- do not construct a live-server Player
    anywhere else in the codebase.
    """
    if not cfg.enabled:
        raise LiveBattleNotEnabledError(
            "LiveBattleConfig.enabled is False. Live battles are disabled "
            "by default per CLAUDE.md's safety rules. Set enabled=True "
            "explicitly (in a one-off script, not a checked-in default) "
            "to proceed."
        )

    if require_manual_confirmation:
        confirmation = os.environ.get("CONFIRM_LIVE_SHOWDOWN_BATTLES")
        if confirmation != "yes":
            raise LiveBattleNotEnabledError(
                "Set CONFIRM_LIVE_SHOWDOWN_BATTLES=yes in the environment "
                "to confirm you intend to run LIVE battles against real "
                "players. This is a second, independent gate on top of "
                "LiveBattleConfig.enabled."
            )

    account_configuration = _load_account_configuration(cfg)
    logger.warning(
        "Connecting agent %s to the LIVE Pokemon Showdown server for %d "
        "battle(s) in format %s. This is real matchmaking against real "
        "opponents.",
        account_configuration.username,
        cfg.n_battles,
        cfg.battle_format,
    )

    player.account_configuration = account_configuration
    player.server_configuration = SHOWDOWN_PUBLIC_SERVER
    return player


def run_live_evaluation(player: Player, cfg: LiveBattleConfig) -> None:
    """
    Thin convenience wrapper: connects (through the gate above) and
    plays cfg.n_battles ladder games. Intended to be invoked from a
    dedicated, manually-run script -- never from training or CI.
    """
    connect_for_live_battles(player, cfg)
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        player.ladder(cfg.n_battles)
    )
