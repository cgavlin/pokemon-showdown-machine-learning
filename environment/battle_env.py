"""
ShowdownBattleEnv: the RL-facing environment for this project.

This wraps poke-env's `SinglesEnv` (which itself talks to a Pokemon
Showdown server -- local by default, per CLAUDE.md's "train locally
first" rule) and layers on:
  - our own state encoding (environment/state.py)
  - our own action validation + penalty handling (environment/actions.py)
  - our own reward shaping (environment/rewards.py)

so the rest of the codebase (agents/training/evaluation) never has to
know it's ultimately built on poke-env, and could be swapped onto a
different underlying battle simulator later without changing callers.

CLAUDE.md requires a local, deterministic, testable environment before
any live-battle connection. This module defaults to a local Showdown
server address (ws://localhost:8000) and never connects to the public
ladder -- see showdown/integration.py for the explicitly-gated live path.
"""

from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player, RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from environment.actions import INVALID_ACTION_PENALTY, sanitize_action
from environment.rewards import BattleRewardState, RewardConfig, compute_reward
from environment.state import encode_battle, observation_size

logger = logging.getLogger(__name__)

LOCAL_SERVER_CONFIGURATION = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


class EncodedSinglesEnv(SinglesEnv):
    """
    poke-env SinglesEnv subclass that plugs in our observation/reward
    logic via the hooks poke-env expects (`embed_battle`, `calc_reward`).

    NOTE: deliberately does NOT start with an underscore. poke-env
    auto-generates a Showdown username from `self.__class__.__name__`
    whenever no explicit AccountConfiguration is supplied (see
    AccountConfiguration.generate). Showdown's server strips/rejects a
    leading underscore from usernames, which produces a mismatch
    between the username the client thinks it registered and the one
    the server actually assigned -- silently breaking the login
    handshake (`wait_for_login` hangs, then asserts). A class name
    starting with `_` is exactly the bug that caused that failure.
    """

    def __init__(self, reward_config: RewardConfig, **kwargs):
        super().__init__(**kwargs)
        self._reward_config = reward_config
        self._reward_states: dict[str, BattleRewardState] = {}
        self.observation_spaces = {
            agent: Box(low=-10.0, high=10.0, shape=(observation_size(),), dtype=np.float32)
            for agent in self.possible_agents
        }

    def embed_battle(self, battle):
        return encode_battle(battle)

    def calc_reward(self, battle) -> float:
        key = battle.battle_tag
        prev = self._reward_states.get(key, BattleRewardState())
        reward, new_state = compute_reward(battle, prev, self._reward_config)
        self._reward_states[key] = new_state
        if battle.finished:
            self._reward_states.pop(key, None)
        return reward


class ShowdownBattleEnv(gym.Env):
    """
    Single-agent Gym environment: our policy vs. a fixed opponent
    (`opponent_player`), backed by a local Pokemon Showdown server.

    Action validation: every step, the requested discrete action is
    checked against poke-env's live action mask. Illegal actions are
    never silently remapped into an arbitrary legal action without
    consequence -- they incur INVALID_ACTION_PENALTY on top of whatever
    (well-defined) fallback action actually gets sent, and the event is
    logged so policy bugs are detectable (per CLAUDE.md "Action Space").
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Optional[Player] = None,
        battle_format: str = "gen9randombattle",
        reward_config: Optional[RewardConfig] = None,
        server_configuration: ServerConfiguration = LOCAL_SERVER_CONFIGURATION,
        account_configuration: Optional[AccountConfiguration] = None,
        team: Optional[str] = None,
    ):
        super().__init__()
        self.reward_config = reward_config or RewardConfig()

        self._underlying = EncodedSinglesEnv(
            reward_config=self.reward_config,
            battle_format=battle_format,
            server_configuration=server_configuration,
            account_configuration1=account_configuration,
            team=team,
        )
        self.opponent = opponent or RandomPlayer(battle_format=battle_format, team=team)
        self._wrapped = SingleAgentWrapper(self._underlying, self.opponent)

        self.observation_space = self._wrapped.observation_space
        self.action_space = self._wrapped.action_space
        self.n_invalid_actions = 0
        self.n_total_actions = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self._wrapped.reset(seed=seed, options=options)

    def step(self, action: int):
        self.n_total_actions += 1
        mask = self.get_action_mask()
        safe_action, was_illegal = sanitize_action(int(action), mask)

        obs, reward, terminated, truncated, info = self._wrapped.step(safe_action)

        if was_illegal:
            self.n_invalid_actions += 1
            reward += INVALID_ACTION_PENALTY
            info = dict(info)
            info["invalid_action"] = True
            logger.warning(
                "Illegal action %d requested (mask=%s); substituted default "
                "action and applied invalid-action penalty.",
                action,
                mask,
            )

        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        battle = self._underlying.battle1
        # poke-env's SinglesEnv.get_action_mask returns a plain Python
        # list[int]; the rest of this codebase (agents/policy.py's
        # masked_argmax, training/trainer.py) assumes a numpy array
        # (e.g. calls like `.astype(bool)`), so normalize here once.
        return np.array(self._underlying.get_action_mask(battle), dtype=np.int64)

    def close(self):
        self._wrapped.close()

    @property
    def battles_won(self) -> int:
        return self._wrapped.env.agent1.n_won_battles if hasattr(self._wrapped.env, "agent1") else 0


def make_env(
    opponent: Optional[Player] = None,
    battle_format: str = "gen9randombattle",
    reward_config: Optional[RewardConfig] = None,
    local: bool = True,
    team: Optional[str] = None,
) -> ShowdownBattleEnv:
    """
    Factory used by training/evaluation scripts. `local=True` (the
    default and the only mode used anywhere except showdown/integration.py)
    always points at a locally-run Showdown server, per CLAUDE.md's
    "Default to local simulation" safety rule.
    """
    server_config = LOCAL_SERVER_CONFIGURATION if local else None
    return ShowdownBattleEnv(
        opponent=opponent,
        battle_format=battle_format,
        reward_config=reward_config,
        server_configuration=server_config or LOCAL_SERVER_CONFIGURATION,
        team=team,
    )
