import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ml.continuous_learning import ContinuousLearningManager
from ml.schema import FEATURE_NAMES


def build_manager(tmp_path):
    return ContinuousLearningManager(
        tmp_path / "trades.sqlite3",
        tmp_path / "paper_training.csv",
    )


def build_order():
    return {
        "asset": "EUR/USD",
        "direction": "CALL",
        "amount": 10.0,
        "expiration_seconds": 60,
        "signal_confidence": 0.8,
        "reasons": [],
        "timeframe": "60s",
        "indicator_snapshot": {
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
            "latest_close": 100.0,
        },
        "market_regime": "trending",
        "strategy": "signal-engine",
        "model_version": None,
    }


def test_closed_paper_trade_persists_labeled_sample_and_updates_learning(tmp_path):
    manager = build_manager(tmp_path)
    order_id = str(uuid4())
    opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manager.record_open_trade(order_id, build_order(), opened_at.isoformat())

    result = manager.record_closed_trade(
        {
            "order_id": order_id,
            **build_order(),
            "opened_at": opened_at.isoformat(),
            "settled_at": (opened_at + timedelta(seconds=60)).isoformat(),
            "duration_seconds": 60.0,
            "pnl_pct": -0.5,
            "payout_pct": 85.0,
        }
    )

    dashboard = manager.get_dashboard()
    rows = list(csv.DictReader((tmp_path / "paper_training.csv").open(encoding="utf-8")))
    assert result == {"order_id": order_id, "outcome": "loss", "label": "PUT"}
    assert dashboard["closed_trades"] == 1
    assert dashboard["dataset_samples"] == 1
    assert dashboard["by_asset"]["EUR/USD"]["win_rate"] == 0
    assert dashboard["online_learning"]["updates"] == 1
    assert rows[0]["label"] == "PUT"
    assert float(rows[0]["realized_return_pct"]) == -100
    assert set(FEATURE_NAMES).issubset(rows[0])


def test_open_paper_trade_survives_database_reopen(tmp_path):
    manager = build_manager(tmp_path)
    order_id = str(uuid4())
    opened_at = datetime.now(timezone.utc).isoformat()
    order = build_order()
    manager.record_open_trade(order_id, order, opened_at)

    restored = build_manager(tmp_path).get_open_trades()

    assert restored == [{"order_id": order_id, "opened_at": opened_at, "order": order}]


def test_cancelled_trade_is_not_restored_or_learned(tmp_path):
    manager = build_manager(tmp_path)
    order_id = str(uuid4())
    opened_at = datetime.now(timezone.utc).isoformat()
    manager.record_open_trade(order_id, build_order(), opened_at)

    assert manager.cancel_open_trade(order_id, datetime.now(timezone.utc).isoformat()) is True
    assert build_manager(tmp_path).get_open_trades() == []
    assert manager.get_dashboard()["dataset_samples"] == 0


def test_winning_trade_uses_direction_and_payout_for_sample(tmp_path):
    manager = build_manager(tmp_path)
    order_id = str(uuid4())
    opened_at = datetime.now(timezone.utc)
    manager.record_open_trade(order_id, build_order(), opened_at.isoformat())
    manager.record_closed_trade(
        {
            "order_id": order_id,
            **build_order(),
            "opened_at": opened_at.isoformat(),
            "settled_at": (opened_at + timedelta(seconds=60)).isoformat(),
            "duration_seconds": 60.0,
            "pnl_pct": 0.85,
            "payout_pct": 85.0,
        }
    )

    row = next(csv.DictReader((tmp_path / "paper_training.csv").open(encoding="utf-8")))
    assert row["label"] == "CALL"
    assert float(row["realized_return_pct"]) == 85


def test_closing_a_trade_without_a_matching_open_row_is_rejected(tmp_path):
    manager = build_manager(tmp_path)
    trade = {
        "order_id": str(uuid4()),
        **build_order(),
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "settled_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": 0,
        "pnl_pct": 0,
        "payout_pct": None,
    }
    with pytest.raises(KeyError, match="no abierta"):
        manager.record_closed_trade(trade)
