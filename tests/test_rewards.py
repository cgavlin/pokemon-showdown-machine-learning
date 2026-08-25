import pytest

from environment.rewards import (
    BattleRewardState,
    RewardConfig,
    move_effectiveness_reward,
)


def test_reward_config_defaults_are_valid():
    cfg = RewardConfig()
    assert cfg.win_reward > 0
    assert cfg.loss_penalty < 0


def test_reward_config_rejects_dominating_non_terminal_event():
    with pytest.raises(ValueError):
        RewardConfig(win_reward=1.0, faint_opponent_pokemon=5.0)


def test_win_reward_dominates_plausible_tactical_sum():
    """
    A ~20-turn battle where every turn scores the single largest
    plausible tactical reward should still be worth less than winning,
    so the agent can't out-value winning by farming effective hits.
    """
    cfg = RewardConfig()
    plausible_turns = 20
    max_per_turn = max(
        cfg.effective_move,
        cfg.super_effective_move,
        cfg.extremely_effective_move,
        cfg.favorable_switch,
        cfg.positional_advantage,
    )
    assert plausible_turns * max_per_turn < cfg.win_reward


@pytest.mark.parametrize(
    "multiplier,expected_attr",
    [
        (0.0, "disadvantageous_move"),
        (0.25, "ineffective_move"),
        (0.5, "ineffective_move"),
        (1.0, "effective_move"),
        (2.0, "super_effective_move"),
        (4.0, "extremely_effective_move"),
    ],
)
def test_move_effectiveness_reward_buckets(multiplier, expected_attr):
    cfg = RewardConfig()
    reward = move_effectiveness_reward(multiplier, cfg)
    assert reward == getattr(cfg, expected_attr)


def test_battle_reward_state_defaults():
    state = BattleRewardState()
    assert state.prev_own_fainted == 0
    assert state.prev_opp_fainted == 0
    assert state.prev_own_hp_total == 6.0
    assert state.prev_opp_hp_total == 6.0
