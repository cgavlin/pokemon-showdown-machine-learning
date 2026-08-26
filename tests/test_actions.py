import numpy as np

from environment.actions import (
    INVALID_ACTION_PENALTY,
    describe_action,
    is_action_legal,
    sanitize_action,
)


def test_is_action_legal_true_for_masked_in_action():
    mask = np.array([1, 0, 1, 0], dtype=bool)
    assert is_action_legal(0, mask) is True
    assert is_action_legal(2, mask) is True


def test_is_action_legal_false_for_masked_out_action():
    mask = np.array([1, 0, 1, 0], dtype=bool)
    assert is_action_legal(1, mask) is False
    assert is_action_legal(3, mask) is False


def test_is_action_legal_false_for_out_of_range_index():
    mask = np.array([1, 0, 1, 0], dtype=bool)
    assert is_action_legal(-1, mask) is False
    assert is_action_legal(10, mask) is False


def test_sanitize_action_passes_through_legal_action():
    mask = np.array([1, 1, 0, 0], dtype=bool)
    action, was_illegal = sanitize_action(1, mask)
    assert action == 1
    assert was_illegal is False


def test_sanitize_action_never_silently_substitutes_without_flagging():
    """
    Core safety property: an illegal action must never
    look identical to a legal one downstream -- the caller MUST be able
    to detect it happened (via the returned flag) even though a
    well-defined fallback action is produced so the battle can continue.
    """
    mask = np.array([1, 0, 0, 0], dtype=bool)
    action, was_illegal = sanitize_action(2, mask)
    assert was_illegal is True
    assert is_action_legal(action, mask)  # fallback itself must be legal


def test_invalid_action_penalty_is_negative():
    assert INVALID_ACTION_PENALTY < 0


def test_describe_action_covers_default_move_and_switch_ranges():
    size = 15  # matches poke-env's typical singles layout used here
    assert describe_action(0, size).kind == "default"
    assert describe_action(1, size).kind == "move"
    assert describe_action(4, size).kind == "move"
    assert describe_action(5, size).kind == "move+tera"
    assert describe_action(9, size).kind == "switch"
