import os
from types import SimpleNamespace

import pytest

from showdown.integration import (
    LiveBattleConfig,
    LiveBattleNotEnabledError,
    connect_for_live_battles,
)


def _fake_player():
    return SimpleNamespace(account_configuration=None, server_configuration=None)


def test_live_battles_disabled_by_default():
    cfg = LiveBattleConfig()
    assert cfg.enabled is False


def test_connect_raises_when_not_enabled():
    cfg = LiveBattleConfig(enabled=False)
    with pytest.raises(LiveBattleNotEnabledError):
        connect_for_live_battles(_fake_player(), cfg)


def test_connect_raises_without_confirmation_env_var(monkeypatch):
    monkeypatch.delenv("CONFIRM_LIVE_SHOWDOWN_BATTLES", raising=False)
    monkeypatch.setenv("SHOWDOWN_USERNAME", "test_user")
    cfg = LiveBattleConfig(enabled=True)
    with pytest.raises(LiveBattleNotEnabledError):
        connect_for_live_battles(_fake_player(), cfg)


def test_connect_raises_without_username_even_if_confirmed(monkeypatch):
    monkeypatch.setenv("CONFIRM_LIVE_SHOWDOWN_BATTLES", "yes")
    monkeypatch.delenv("SHOWDOWN_USERNAME", raising=False)
    cfg = LiveBattleConfig(enabled=True)
    with pytest.raises(RuntimeError):
        connect_for_live_battles(_fake_player(), cfg)


def test_connect_succeeds_with_both_gates_and_credentials(monkeypatch):
    monkeypatch.setenv("CONFIRM_LIVE_SHOWDOWN_BATTLES", "yes")
    monkeypatch.setenv("SHOWDOWN_USERNAME", "test_user")
    monkeypatch.setenv("SHOWDOWN_PASSWORD", "test_pass")
    cfg = LiveBattleConfig(enabled=True)
    player = connect_for_live_battles(_fake_player(), cfg)
    assert player.account_configuration.username == "test_user"
