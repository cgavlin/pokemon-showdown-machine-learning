"""
ShowdownBattleEnv: the RL-facing environment for this project.

This wraps poke-env's `SinglesEnv` (which itself talks to a Pokemon
Showdown server -- local by default, never the public ladder) and adds:
  - our own state encoding (environment/state.py)
  - our own action validation + penalty handling (environment/actions.py)
  - our own reward shaping (environment/rewards.py)
  - optional cross-battle Pokemon knowledge (knowledge/pokemon_knowledge.py)

so the rest of the codebase (agents/training/evaluation) never has to
know it's ultimately built on poke-env, and could be swapped onto a
different underlying battle simulator later without changing callers.

"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player, RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from environment.actions import INVALID_ACTION_PENALTY, as_poke_env_action, sanitize_action
from environment.rewards import BattleRewardState, RewardConfig, compute_reward
from environment.state import encode_battle, observation_size

if TYPE_CHECKING:
    from knowledge.pokemon_knowledge import PokemonKnowledgeBase

logger = logging.getLogger(__name__)

LOCAL_SERVER_CONFIGURATION = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
# Showdown usernames are capped at 18 chars. These are base names only
# -- see _unique_username() below for why every actual connection gets
# a random suffix appended rather than using these directly.
_LEARNER_USERNAME_BASE = "AshGPT"
_OPPONENT_USERNAME_BASE = "Promptachu"


def _unique_username(base: str) -> str:
    """
    Appends a short random suffix to `base` so every ShowdownBattleEnv
    connection gets its own Showdown username instead of colliding.

    A fixed, hardcoded username reused across every env instance causes
    a `|nametaken|` ShowdownException the moment two connections
    overlap even slightly -- e.g. a previous run's connection hasn't
    finished tearing down yet, or a single process opens several envs
    back-to-back (training/curriculum_runner.py builds one per
    curriculum stage; training/pooled_self_play_trainer.py rebuilds one
    on every opponent swap plus one per absolute-skill eval). None of
    those are errors on our end -- they're just more than one
    connection under the same name existing at overlapping times.
    Suffixing with a few random hex chars (kept short enough to stay
    under Showdown's 18-char username cap even for the longer base
    name) makes every connection's name unique instead.
    """
    suffix = uuid.uuid4().hex[:6]
    return f"{base}{suffix}"[:18]


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

    def __init__(
        self,
        reward_config: RewardConfig,
        knowledge_base: Optional["PokemonKnowledgeBase"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._reward_config = reward_config
        self._knowledge_base = knowledge_base
        self._reward_states: dict[int, BattleRewardState] = {}
        self.observation_spaces = {
            agent: Box(low=-10.0, high=10.0, shape=(observation_size(),), dtype=np.float32)
            for agent in self.possible_agents
        }

    def embed_battle(self, battle):
        return encode_battle(battle, knowledge_base=self._knowledge_base)

    def calc_reward(self, battle) -> float:
        # battle.battle_tag is the SAME string for both sides of a battle
        # (self.battle1 and self.battle2 are two different Battle objects
        # -- agent1's and agent2's own perspectives on one real battle --
        # but poke-env gives them identical tags). calc_reward() is
        # called once per agent per step, so keying incremental reward
        # state by battle_tag alone means agent1's and agent2's calls
        # silently overwrite the SAME BattleRewardState entry every
        # step, corrupting each other's prev_own_hp_total/
        # prev_opp_hp_total tracking (and hence the resulting reward)
        # for both sides. Key by object identity instead -- battle1 and
        # battle2 are stable, distinct objects for the lifetime of a
        # battle, so this cleanly separates the two agents' state while
        # still reusing the same entry turn-over-turn for a given agent.
        key = id(battle)
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
    logged so policy bugs are detectable.

    `knowledge_base`, if given, is exposed as `self.knowledge_base` so
    a trainer can call `.observe_battle(...)` / `.save()` on it after
    each episode, AND is threaded into the underlying encoder so every
    observation this env produces already reflects whatever that
    knowledge base currently knows (see environment/state.py and
    knowledge/pokemon_knowledge.py).
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
        knowledge_base: Optional["PokemonKnowledgeBase"] = None,
    ):
        super().__init__()
        self.reward_config = reward_config or RewardConfig()
        self.knowledge_base = knowledge_base

        self._underlying = EncodedSinglesEnv(
            reward_config=self.reward_config,
            knowledge_base=self.knowledge_base,
            battle_format=battle_format,
            server_configuration=server_configuration,
            account_configuration1=account_configuration
            or AccountConfiguration(_unique_username(_LEARNER_USERNAME_BASE), None),
            account_configuration2=AccountConfiguration(_unique_username(_OPPONENT_USERNAME_BASE), None),
            team=team,
        )
        self.opponent = opponent or RandomPlayer(battle_format=battle_format, team=team)
        self._wrapped = SingleAgentWrapper(self._underlying, self.opponent)

        self.observation_space = self._wrapped.observation_space
        self.action_space = self._wrapped.action_space
        self.n_invalid_actions = 0
        self.n_total_actions = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # poke-env's PokeEnv.reset() (SingleAgentWrapper's underlying
        # two-agent env) always wraps the per-agent observation as
        # {"observation": ..., "action_mask": ...} -- built-in support
        # for action masking that we don't use here (we compute our own
        # mask separately via get_action_mask(), reading straight from
        # the battle object). Unwrap to the flat array the rest of this
        # codebase (Trainer, ReplayBuffer, epsilon_greedy_action) expects.
        obs, info = self._wrapped.reset(seed=seed, options=options)
        return obs["observation"], info

    def step(self, action: int):
        self.n_total_actions += 1
        mask = self.get_action_mask()
        safe_action, was_illegal = sanitize_action(int(action), mask)

        # poke-env's own SinglesEnv.action_to_order (called internally by
        # SingleAgentWrapper.step -> PokeEnv.step) requires a numpy
        # scalar action -- see environment/actions.py's
        # as_poke_env_action for why.
        obs, reward, terminated, truncated, info = self._wrapped.step(as_poke_env_action(safe_action))

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

        # Same unwrapping as reset() -- see the comment there.
        return obs["observation"], reward, terminated, truncated, info

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
    knowledge_base: Optional["PokemonKnowledgeBase"] = None,
) -> ShowdownBattleEnv:
    """
    Factory used by training/evaluation scripts. `local=True` (the
    default and the only mode used anywhere except showdown/integration.py)
    always points at a locally-run Showdown server
    """
    server_config = LOCAL_SERVER_CONFIGURATION if local else None
    return ShowdownBattleEnv(
        opponent=opponent,
        battle_format=battle_format,
        reward_config=reward_config,
        server_configuration=server_config or LOCAL_SERVER_CONFIGURATION,
        team=team,
        knowledge_base=knowledge_base,
    )