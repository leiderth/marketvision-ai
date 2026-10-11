import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ml.continuous_learning import ContinuousLearningManager
from paper_trading.autopilot import PaperAutopilot
from paper_trading.paper_execution import PaperExecutionManager
from risk.risk_manager import RiskLimits, RiskManager


def build_snapshot(timestamp: datetime, *, open_price: float, close: float) -> dict:
    return {
        "status": {"connected": True, "fresh": True, "quality_accepted": True},
        "candles": [
            {
                "symbol": "EUR/USD",
                "timestamp": timestamp.isoformat(),
                "interval_seconds": 60,
                "open": open_price,
                "high": max(open_price, close),
                "low": min(open_price, close),
                "close": close,
                "volume": 100.0,
            }
        ],
        "indicators": {
            "values": {
                "sma_20": 101.0,
                "sma_50": 100.0,
                "ema_20": 101.0,
                "ema_50": 100.0,
                "macd": {"histogram": 0.1},
                "atr_14": 1.0,
                "rsi_14": 60.0,
                "vwap_session": 99.0,
            },
            "as_of": timestamp.isoformat(),
        },
        "market_regime": {"regime": "trending"},
    }


def build_autopilot(tmp_path, signal):
    risk = RiskManager(
        limits=RiskLimits(1, 5, 3, 20, 1000, max_trades_per_hour=5),
        state_path=tmp_path / "risk.json",
    )
    execution = PaperExecutionManager(risk)
    learning = ContinuousLearningManager(
        tmp_path / "learning.sqlite3",
        tmp_path / "training.csv",
    )
    autopilot = PaperAutopilot(execution, risk, learning, lambda _snapshot: signal[0])
    return autopilot, execution, risk, learning


def valid_signal(direction="CALL"):
    return {
        "signal": direction,
        "confidence": 0.8,
        "should_trade": True,
        "reasons": ["test signal"],
        "explainability": {"ml_model": None},
    }


def test_autopilot_is_disabled_by_default_and_requires_paper_mode(tmp_path):
    signal = [valid_signal()]
    autopilot, execution, _, _ = build_autopilot(tmp_path, signal)
    first_time = datetime.now(timezone.utc)
    snapshot = build_snapshot(first_time, open_price=100, close=100)

    autopilot.process_once(mode="ANALYSIS_MODE", market_snapshot=snapshot)
    assert execution.get_open_positions() == []
    assert autopilot.get_status()["enabled"] is False

    autopilot.configure(enabled=True, amount=10, payout_pct=80)
    autopilot.prime(snapshot)
    autopilot.process_once(mode="ANALYSIS_MODE", market_snapshot=snapshot)
    assert execution.get_open_positions() == []
    assert autopilot.get_status()["enabled"] is False


def test_autopilot_opens_one_position_per_new_candle_and_never_duplicates(tmp_path):
    signal = [valid_signal()]
    autopilot, execution, _, _ = build_autopilot(tmp_path, signal)
    previous = datetime.now(timezone.utc) - timedelta(seconds=70)
    first_snapshot = build_snapshot(previous, open_price=100, close=100)
    autopilot.configure(enabled=True, amount=10, payout_pct=80)
    autopilot.prime(first_snapshot)

    next_snapshot = build_snapshot(previous + timedelta(seconds=60), open_price=100, close=101)
    autopilot.process_once(mode="PAPER_TRADING_MODE", market_snapshot=next_snapshot)
    autopilot.process_once(mode="PAPER_TRADING_MODE", market_snapshot=next_snapshot)

    positions = execution.get_open_positions()
    assert len(positions) == 1
    assert positions[0]["strategy"] == "paper-autopilot"
    assert positions[0]["direction"] == "CALL"
    assert positions[0]["amount"] == 10


def test_autopilot_settles_using_expiry_price_and_records_learning_sample(tmp_path):
    signal = [valid_signal()]
    autopilot, execution, risk, learning = build_autopilot(tmp_path, signal)
    previous = datetime.now(timezone.utc) - timedelta(seconds=70)
    first_snapshot = build_snapshot(previous, open_price=100, close=100)
    autopilot.configure(enabled=True, amount=10, payout_pct=80)
    autopilot.prime(first_snapshot)

    entry_snapshot = build_snapshot(previous + timedelta(seconds=60), open_price=100, close=100)
    autopilot.process_once(mode="PAPER_TRADING_MODE", market_snapshot=entry_snapshot)
    opened = execution.get_open_positions()[0]
    expiry_time = datetime.fromisoformat(opened["opened_at"]) + timedelta(seconds=60)
    signal[0] = {
        **valid_signal(),
        "signal": "NO_TRADE",
        "should_trade": False,
    }
    expiry_snapshot = build_snapshot(expiry_time, open_price=101, close=102)

    autopilot.process_once(mode="PAPER_TRADING_MODE", market_snapshot=expiry_snapshot)

    assert execution.get_open_positions() == []
    assert risk.get_state()["daily_pnl_pct"] == pytest.approx(0.8)
    dashboard = learning.get_dashboard()
    assert dashboard["closed_trades"] == 1
    assert dashboard["by_asset"]["EUR/USD"]["win_rate"] == 1
    assert dashboard["dataset_samples"] == 1


def test_autopilot_stops_if_risk_manager_rejects_the_configured_amount(tmp_path):
    signal = [valid_signal()]
    autopilot, execution, risk, _ = build_autopilot(tmp_path, signal)
    now = datetime.now(timezone.utc)
    snapshot = build_snapshot(now, open_price=100, close=100)
    autopilot.configure(enabled=True, amount=10, payout_pct=80)
    autopilot.prime(snapshot)
    risk.lock_trading("test risk lock")
    next_snapshot = build_snapshot(now + timedelta(seconds=60), open_price=100, close=100)

    autopilot.process_once(mode="PAPER_TRADING_MODE", market_snapshot=next_snapshot)

    assert autopilot.get_status()["enabled"] is False
    assert execution.get_open_positions() == []
