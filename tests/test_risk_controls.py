import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from execution.base import OrderRequest
from paper_trading.paper_execution import PaperExecutionManager
from risk.risk_manager import RiskLimits, RiskManager


def manager(tmp_path, **overrides) -> RiskManager:
    defaults = {
        "max_loss_per_trade_pct": 1.0,
        "max_daily_loss_pct": 2.0,
        "max_consecutive_losses": 2,
        "max_trades_per_day": 3,
        "paper_account_balance": 1000.0,
    }
    defaults.update(overrides)
    return RiskManager(
        limits=RiskLimits(**defaults),
        state_path=tmp_path / "risk.json",
    )


def paper_order(*, amount: float = 10.0, direction: str = "CALL") -> OrderRequest:
    return OrderRequest(
        asset="EUR/USD",
        direction=direction,
        amount=amount,
        expiration_seconds=60,
        signal_confidence=0.8,
        reasons=["paper test"],
    )


def test_risk_manager_caps_order_size_tracks_loss_streak_and_persists_lock(tmp_path):
    risk = manager(tmp_path, max_daily_loss_pct=5.0)
    paper = PaperExecutionManager(risk)

    assert paper.submit_order(paper_order(amount=10)).status == "SIMULATED"
    assert paper.submit_order(paper_order(amount=10)).status == "SIMULATED"
    for order in paper.get_open_positions():
        paper.settle_order(order["order_id"], -1)

    assert risk.is_trading_allowed() is False
    assert "consecutivas" in risk.get_state()["lock_reason"]
    assert RiskManager(
        limits=RiskLimits(1, 5, 2, 3, 1000),
        state_path=tmp_path / "risk.json",
    ).is_trading_allowed() is False


def test_daily_loss_max_and_explicit_unlock(tmp_path):
    risk = manager(tmp_path)
    risk.record_trade_result(-1.0)
    risk.record_trade_result(-1.0)

    assert risk.is_trading_allowed() is False
    assert "diaria" in risk.get_state()["lock_reason"]
    risk.unlock_trading()
    assert risk.is_trading_allowed() is True


def test_daily_rollover_does_not_silently_unlock_locked_risk(tmp_path):
    risk = manager(tmp_path)
    risk.lock_trading("manual stop")
    next_day = datetime(2026, 9, 28, tzinfo=timezone.utc)

    state = risk.get_state(now=next_day)

    assert state["trading_day"] == "2026-09-28"
    assert state["trades_today"] == 0
    assert state["trading_locked"] is True
    assert risk.is_trading_allowed(now=next_day) is False


def test_paper_manager_rejects_invalid_order_and_kill_switch_clears_open_orders(tmp_path):
    risk = manager(tmp_path)
    paper = PaperExecutionManager(risk)
    assert paper.submit_order(paper_order(amount=10.01)).status == "REJECTED_BY_RISK_MANAGER"
    assert paper.submit_order(paper_order(direction="HOLD")).status == "REJECTED_INVALID_ORDER"
    accepted = paper.submit_order(paper_order())
    assert accepted.status == "SIMULATED"

    paper.kill_switch()

    assert paper.get_open_positions() == []
    assert risk.is_trading_allowed() is False


def test_risk_state_corruption_locks_instead_of_resetting(tmp_path):
    state_path = tmp_path / "risk.json"
    state_path.write_text('{"trading_locked":"false"}', encoding="utf-8")

    risk = manager(tmp_path)

    assert risk.is_trading_allowed() is False
    assert "inválido" in risk.get_state()["lock_reason"]


def test_risk_limits_reject_invalid_configuration():
    with pytest.raises(ValueError, match="límites"):
        RiskLimits(0, 5, 3, 10, 1000)


def test_paper_settlement_cannot_exceed_order_stake(tmp_path):
    risk = manager(tmp_path)
    paper = PaperExecutionManager(risk)
    order = paper.submit_order(paper_order(amount=10))

    with pytest.raises(ValueError, match="importe máximo"):
        paper.settle_order(order.order_id, -1.1)

    assert len(paper.get_open_positions()) == 1
