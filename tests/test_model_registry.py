import csv
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from analysis.signal_engine import SignalEngine
from backtesting.walk_forward import optimize_confidence_thresholds, run_walk_forward
from ml.model_registry import (
    MLModelManager,
    feature_vector,
    train_and_register,
)
from ml.schema import FEATURE_NAMES, ModelRegistryError


def write_labeled_dataset(path, *, count: int = 150) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8", newline="") as dataset:
        writer = csv.DictWriter(
            dataset,
            fieldnames=[
                "timestamp",
                "label_end_timestamp",
                *FEATURE_NAMES,
                "label",
                "realized_return_pct",
            ],
        )
        writer.writeheader()
        for index in range(count):
            label = "CALL" if index % 2 == 0 else "PUT"
            direction = 1.0 if label == "CALL" else -1.0
            timestamp = start + timedelta(minutes=index)
            row = {
                "timestamp": timestamp.isoformat(),
                "label_end_timestamp": (timestamp + timedelta(seconds=1)).isoformat(),
                "label": label,
                "realized_return_pct": direction * 0.1,
            }
            row.update({name: direction * (1 + index * 0.0001) for name in FEATURE_NAMES})
            writer.writerow(row)


def test_training_registers_an_inactive_model_and_loads_linear_shap(tmp_path):
    dataset = tmp_path / "labeled.csv"
    registry = tmp_path / "registry"
    write_labeled_dataset(dataset)

    registered = train_and_register(dataset, registry)

    assert registered["registered"] is True
    assert registered["active"] is False
    assert registered["metrics"]["test_balanced_accuracy"] >= 0.55
    manager = MLModelManager(
        registry,
        registered["version"],
        drift_state_dir=tmp_path / "drift",
    )
    assert manager.get_status()["model_loaded"] is True
    result = manager.predict(
        {
            "sma_20": 101,
            "sma_50": 100,
            "ema_20": 101,
            "ema_50": 100,
            "macd": {"histogram": 0.1},
            "atr_14": 1,
            "rsi_14": 60,
            "vwap_session": 99,
        },
        100,
        as_of=datetime.now(timezone.utc).isoformat(),
    )
    assert result["available"] is True
    assert result["prediction"] == "CALL"
    explanation = result["explainability"]
    assert explanation["method"] == "linear_interventional_shap"
    assert explanation["expected_value"] + sum(
        item["contribution"] for item in explanation["contributions"]
    ) == pytest.approx(result["logit"])


def test_bad_or_unconfigured_model_is_never_silently_activated(tmp_path):
    unconfigured = MLModelManager(tmp_path)
    assert unconfigured.predict({}, None) == {"configured": False}

    invalid = MLModelManager(tmp_path, "missing-model")
    assert invalid.get_status()["model_loaded"] is False
    assert invalid.predict({}, None)["available"] is False


def test_training_rejects_small_datasets_and_cross_boundary_label_windows(tmp_path):
    dataset = tmp_path / "too-small.csv"
    write_labeled_dataset(dataset, count=80)
    with pytest.raises(ModelRegistryError, match="al menos 100"):
        train_and_register(dataset, tmp_path / "registry")

    write_labeled_dataset(dataset)
    with dataset.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows[:90]:
        row["label_end_timestamp"] = rows[90]["timestamp"]
    with dataset.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ModelRegistryError, match="purgado"):
        train_and_register(dataset, tmp_path / "purged-registry")


def test_model_failing_holdout_threshold_is_not_registered(tmp_path):
    dataset = tmp_path / "unlearnable.csv"
    registry = tmp_path / "registry"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with dataset.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["timestamp", "label_end_timestamp", *FEATURE_NAMES, "label"],
        )
        writer.writeheader()
        for index in range(150):
            timestamp = start + timedelta(minutes=index)
            writer.writerow(
                {
                    "timestamp": timestamp.isoformat(),
                    "label_end_timestamp": (timestamp + timedelta(seconds=1)).isoformat(),
                    **dict.fromkeys(FEATURE_NAMES, 0),
                    "label": "CALL" if index % 2 == 0 else "PUT",
                }
            )

    with pytest.raises(ModelRegistryError, match="no se registró"):
        train_and_register(dataset, registry)
    assert not registry.exists()


def test_registry_rejects_corrupt_model_artifact(tmp_path):
    dataset = tmp_path / "labeled.csv"
    registry = tmp_path / "registry"
    write_labeled_dataset(dataset)
    registered = train_and_register(dataset, registry)
    artifact = registry / f"{registered['version']}.json"
    content = artifact.read_text(encoding="utf-8")
    artifact.write_text(content.replace('"intercept":', '"intercept": 987654, "old_intercept":'), encoding="utf-8")

    manager = MLModelManager(registry, registered["version"])

    assert manager.get_status()["model_loaded"] is False
    assert "integridad" in manager.get_status()["error"]


def test_active_model_cannot_override_a_deterministic_no_trade(tmp_path):
    dataset = tmp_path / "labeled.csv"
    registry = tmp_path / "registry"
    write_labeled_dataset(dataset)
    registered = train_and_register(dataset, registry)
    model_manager = MLModelManager(
        registry,
        registered["version"],
        drift_state_dir=tmp_path / "drift",
    )

    result = SignalEngine().generate(
        {
            "status": {"connected": False, "fresh": False, "quality_accepted": False},
            "indicators": {"ready": False, "reasons": [], "values": {}},
            "market_regime": {
                "ready": False,
                "trade_blocked": True,
                "regime": "unknown",
                "reasons": [],
            },
            "candles": [],
        },
        ml_prediction=model_manager.predict({}, None),
    )

    assert result["signal"] == "NO_TRADE"
    assert result["should_trade"] is False


def test_feature_vector_rejects_missing_or_invalid_market_inputs():
    with pytest.raises(ModelRegistryError, match="insuficientes"):
        feature_vector({"sma_20": float("nan")}, None)


def test_walk_forward_backtest_reports_sequential_costed_directional_proxy(tmp_path):
    dataset = tmp_path / "labeled.csv"
    write_labeled_dataset(dataset, count=150)

    result = run_walk_forward(dataset, folds=3, transaction_cost_pct=0.01)

    assert len(result["folds"]) == 3
    assert result["accepted_predictions"] > 0
    assert result["win_rate"] == 1.0
    assert result["max_drawdown_pct"] == 0
    assert result["transaction_cost_pct_per_trade"] == 0.01
    assert "proxy direccional" in result["limitations"][0]


def test_threshold_optimization_uses_final_fold_only_for_holdout(tmp_path):
    dataset = tmp_path / "labeled.csv"
    write_labeled_dataset(dataset, count=240)

    result = optimize_confidence_thresholds(
        dataset,
        folds=3,
        minimum_tuning_trades=5,
    )

    assert result["selected_on_folds"] == [1, 2]
    assert result["final_holdout_fold"] == 3
    assert result["suggested_threshold"] in {0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9}
    assert result["activated_automatically"] is False
    assert result["holdout"]["trades"] > 0
