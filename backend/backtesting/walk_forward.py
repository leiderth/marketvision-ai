"""Chronological walk-forward evaluation for direction-labeled market data."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml.schema import FEATURE_NAMES, MIN_TRAINING_ROWS, ModelRegistryError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def run_walk_forward(
    dataset_path: str | Path,
    *,
    folds: int = 3,
    min_confidence: float = 0.65,
    transaction_cost_pct: float = 0.0,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("Se requieren al menos dos ventanas walk-forward.")
    if not math.isfinite(min_confidence) or not 0.5 < min_confidence <= 1:
        raise ValueError("min_confidence debe estar entre 0.5 y 1.")
    if not math.isfinite(transaction_cost_pct) or transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct debe ser finito y no negativo.")

    frame = pd.read_csv(dataset_path)
    required_columns = {
        "timestamp",
        "label_end_timestamp",
        "label",
        "realized_return_pct",
        *FEATURE_NAMES,
    }
    if not required_columns.issubset(frame.columns):
        raise ModelRegistryError("El CSV de backtest no cumple el esquema documentado.")
    frame = frame[["timestamp", "label_end_timestamp", *FEATURE_NAMES, "label", "realized_return_pct"]]
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["label_end_timestamp"] = pd.to_datetime(
        frame["label_end_timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["realized_return_pct"] = pd.to_numeric(
        frame["realized_return_pct"],
        errors="coerce",
    )
    if (
        frame["timestamp"].isna().any()
        or frame["timestamp"].duplicated().any()
        or frame["label_end_timestamp"].isna().any()
        or (frame["label_end_timestamp"] <= frame["timestamp"]).any()
    ):
        raise ModelRegistryError("El backtest contiene timestamps inválidos o duplicados.")
    feature_frame = frame.loc[:, FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(feature_frame.to_numpy(dtype=float)).all() or not np.isfinite(
        frame["realized_return_pct"].to_numpy(dtype=float)
    ).all():
        raise ModelRegistryError("El backtest contiene valores no numéricos o no finitos.")
    if not set(frame["label"].unique()).issubset({"CALL", "PUT"}):
        raise ModelRegistryError("Las etiquetas de backtest deben ser CALL o PUT.")
    label_matches_return = (
        ((frame["label"] == "CALL") & (frame["realized_return_pct"] > 0))
        | ((frame["label"] == "PUT") & (frame["realized_return_pct"] < 0))
    )
    if not label_matches_return.all():
        raise ModelRegistryError(
            "Las etiquetas deben corresponder al signo del retorno futuro observado."
        )
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    rows = len(frame)
    minimum_test_size = max(1, math.ceil(rows * 0.1))
    if rows < MIN_TRAINING_ROWS + folds * minimum_test_size:
        raise ModelRegistryError("El dataset es demasiado pequeño para las ventanas solicitadas.")
    first_test = int(rows * 0.7)
    test_size = (rows - first_test) // folds
    if test_size < 1:
        raise ModelRegistryError("No hay suficientes filas de test por ventana.")

    all_results = []
    all_predictions = []
    fold_metrics = []
    for fold_number in range(folds):
        test_start_index = first_test + fold_number * test_size
        test_end_index = rows if fold_number == folds - 1 else test_start_index + test_size
        test_rows = frame.iloc[test_start_index:test_end_index]
        test_start = test_rows["timestamp"].iloc[0]
        train_rows = frame.iloc[:test_start_index]
        train_rows = train_rows[train_rows["label_end_timestamp"] < test_start]
        if len(train_rows) < MIN_TRAINING_ROWS:
            raise ModelRegistryError("La purga temporal deja insuficientes datos de entrenamiento.")
        if set(train_rows["label"].unique()) != {"CALL", "PUT"}:
            raise ModelRegistryError("Cada ventana de entrenamiento debe contener ambas clases.")
        if set(test_rows["label"].unique()) != {"CALL", "PUT"}:
            raise ModelRegistryError("Cada fold de test debe contener ambas clases.")

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_rows.loc[:, FEATURE_NAMES].to_numpy(dtype=float))
        test_x = scaler.transform(test_rows.loc[:, FEATURE_NAMES].to_numpy(dtype=float))
        train_y = (train_rows["label"].to_numpy() == "CALL").astype(int)
        test_y = (test_rows["label"].to_numpy() == "CALL").astype(int)
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
            solver="lbfgs",
        )
        classifier.fit(train_x, train_y)
        probabilities = classifier.predict_proba(test_x)[:, 1]
        predicted_y = (probabilities >= 0.5).astype(int)
        confidences = np.maximum(probabilities, 1 - probabilities)
        active = confidences >= min_confidence
        fold_results = []
        for local_index, (_, row) in enumerate(test_rows.iterrows()):
            direction = 1 if predicted_y[local_index] else -1
            gross_return = float(row["realized_return_pct"]) * direction
            net_return = gross_return - transaction_cost_pct
            prediction_result = {
                "timestamp": row["timestamp"].isoformat(),
                "fold": fold_number + 1,
                "prediction": "CALL" if direction > 0 else "PUT",
                "label": row["label"],
                "confidence_score": float(confidences[local_index]),
                "would_trade": bool(active[local_index]),
                "gross_directional_return_pct": gross_return,
                "net_directional_return_pct": net_return,
            }
            all_predictions.append(prediction_result)
            if active[local_index]:
                fold_results.append(prediction_result)
        fold_metrics.append(
            {
                "fold": fold_number + 1,
                "training_rows": len(train_rows),
                "test_rows": len(test_rows),
                "accuracy": float(accuracy_score(test_y, predicted_y)),
                "balanced_accuracy": float(balanced_accuracy_score(test_y, predicted_y)),
                "coverage": float(active.mean()),
                "accepted_predictions": int(active.sum()),
            }
        )
        all_results.extend(fold_results)

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for result in all_results:
        factor = 1 + result["net_directional_return_pct"] / 100
        if factor <= 0:
            equity = 0.0
        else:
            equity *= factor
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    wins = sum(result["net_directional_return_pct"] > 0 for result in all_results)
    gross_profit = sum(
        result["net_directional_return_pct"]
        for result in all_results
        if result["net_directional_return_pct"] > 0
    )
    gross_loss = -sum(
        result["net_directional_return_pct"]
        for result in all_results
        if result["net_directional_return_pct"] < 0
    )
    return {
        "method": "expanding_walk_forward_logistic",
        "dataset_rows": rows,
        "folds": fold_metrics,
        "accepted_predictions": len(all_results),
        "coverage": len(all_results) / sum(item["test_rows"] for item in fold_metrics),
        "win_rate": wins / len(all_results) if all_results else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "cumulative_directional_return_pct": (equity - 1) * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "transaction_cost_pct_per_trade": transaction_cost_pct,
        "min_confidence_score": min_confidence,
        "results": all_results,
        "all_predictions": all_predictions,
        "limitations": [
            "El retorno es un proxy direccional close-to-close del subyacente, no payoff de opciones binarias.",
            "No incluye latencia, spread variable, slippage, rechazos ni disponibilidad de broker.",
            "Las puntuaciones de clase no están calibradas como probabilidades.",
        ],
    }


def optimize_confidence_thresholds(
    dataset_path: str | Path,
    *,
    candidates: tuple[float, ...] = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
    folds: int = 4,
    transaction_cost_pct: float = 0.0,
    minimum_tuning_trades: int = 5,
) -> dict[str, Any]:
    if len(candidates) < 2 or any(
        not math.isfinite(candidate) or not 0.5 < candidate <= 1
        for candidate in candidates
    ):
        raise ValueError("Se requieren al menos dos umbrales únicos en (0.5, 1].")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Los umbrales de confianza deben ser únicos.")
    if folds < 3:
        raise ValueError("La optimización requiere al menos tres folds.")
    if minimum_tuning_trades <= 0:
        raise ValueError("minimum_tuning_trades debe ser positivo.")
    evaluation = run_walk_forward(
        dataset_path,
        folds=folds,
        min_confidence=0.5000001,
        transaction_cost_pct=transaction_cost_pct,
    )
    predictions = evaluation["all_predictions"]
    tuning = [row for row in predictions if row["fold"] < folds]
    holdout = [row for row in predictions if row["fold"] == folds]
    candidates_results = []
    for threshold in candidates:
        selected = [
            row for row in tuning if row["confidence_score"] >= threshold
        ]
        if len(selected) < minimum_tuning_trades:
            continue
        metrics = _return_metrics(selected)
        candidates_results.append(
            {
                "threshold": threshold,
                "tuning_trades": len(selected),
                **metrics,
                "selection_score": metrics["cumulative_return_pct"]
                / max(metrics["max_drawdown_pct"], 0.01),
            }
        )
    if not candidates_results:
        raise ModelRegistryError(
            "No hay suficientes operaciones aceptadas para comparar umbrales."
        )
    best = max(
        candidates_results,
        key=lambda result: (result["selection_score"], result["threshold"]),
    )
    holdout_results = [
        row for row in holdout if row["confidence_score"] >= best["threshold"]
    ]
    return {
        "method": "tuning_folds_then_final_holdout",
        "suggested_threshold": best["threshold"],
        "selected_on_folds": list(range(1, folds)),
        "final_holdout_fold": folds,
        "tuning_candidates": candidates_results,
        "holdout": {
            "trades": len(holdout_results),
            "coverage": len(holdout_results) / len(holdout) if holdout else 0.0,
            **_return_metrics(holdout_results),
        },
        "activated_automatically": False,
        "limitations": [
            "Umbrales aplican a puntuaciones de clase no calibradas.",
            "El resultado es proxy direccional close-to-close, no payoff de opciones binarias.",
            "El umbral sugerido requiere revisión humana y no se modifica en producción.",
        ],
    }


def _return_metrics(results: list[dict[str, Any]]) -> dict[str, float | None]:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for result in results:
        factor = 1 + result["net_directional_return_pct"] / 100
        equity = equity * factor if factor > 0 else 0.0
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak)
    gains = sum(
        row["net_directional_return_pct"]
        for row in results
        if row["net_directional_return_pct"] > 0
    )
    losses = -sum(
        row["net_directional_return_pct"]
        for row in results
        if row["net_directional_return_pct"] < 0
    )
    return {
        "win_rate": (
            sum(row["net_directional_return_pct"] > 0 for row in results) / len(results)
            if results
            else None
        ),
        "profit_factor": gains / losses if losses > 0 else None,
        "cumulative_return_pct": (equity - 1) * 100,
        "max_drawdown_pct": drawdown * 100,
    }
