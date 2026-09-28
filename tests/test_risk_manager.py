import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from risk.risk_manager import RiskManager


def test_trading_allowed_by_default():
    rm = RiskManager()
    assert rm.is_trading_allowed() is True


def test_kill_switch_locks_trading():
    rm = RiskManager()
    rm.lock_trading("test")
    assert rm.is_trading_allowed() is False
    assert rm.get_state()["lock_reason"] == "test"


def test_unlock_requires_explicit_call():
    rm = RiskManager()
    rm.lock_trading("test")
    rm.unlock_trading()
    assert rm.is_trading_allowed() is True
