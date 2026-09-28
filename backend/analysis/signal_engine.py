"""Deterministic, rule-based market analysis with additive explanations."""
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, ClassVar


class SignalEngine:
    MIN_DIRECTIONAL_EVIDENCE = 3
    MIN_CONFIDENCE = 0.65

    _WEIGHTS: ClassVar[dict[str, float]] = {
        "moving_average_alignment": 2.0,
        "macd_momentum": 1.5,
        "rsi_momentum": 1.0,
        "price_structure": 1.0,
        "session_vwap": 0.5,
    }

    def generate(
        self,
        snapshot: Mapping[str, Any],
        *,
        ml_prediction: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        status = snapshot.get("status")
        indicators = snapshot.get("indicators")
        regime = snapshot.get("market_regime")
        if not isinstance(status, Mapping) or not isinstance(indicators, Mapping) or not isinstance(
            regime, Mapping
        ):
            return self._no_trade(
                "Snapshot de mercado incompleto; NO_TRADE.",
                market_regime=None,
                as_of=None,
                attributions=[],
            )

        market_regime = regime.get("regime") if isinstance(regime.get("regime"), str) else None
        as_of = indicators.get("as_of")
        if not status.get("connected") or not status.get("fresh"):
            reasons.append("Feed desconectado o desactualizado; NO_TRADE.")
        if not status.get("quality_accepted"):
            reasons.append("Datos no aprobados por DataQualityGate; NO_TRADE.")
            reasons.extend(self._string_reasons(status.get("quality_reasons")))
        if not indicators.get("ready"):
            reasons.extend(self._string_reasons(indicators.get("reasons")))
            if not indicators.get("reasons"):
                reasons.append("Indicadores no disponibles; NO_TRADE.")
        if not regime.get("ready") or regime.get("trade_blocked"):
            reasons.extend(self._string_reasons(regime.get("reasons")))
            if not regime.get("reasons"):
                reasons.append("Régimen de mercado no operativo; NO_TRADE.")

        candles = snapshot.get("candles")
        latest_close = None
        if isinstance(candles, list) and candles and isinstance(candles[-1], Mapping):
            latest_close = self._number(candles[-1].get("close"))
        values = indicators.get("values")
        if not isinstance(values, Mapping):
            values = {}
        reasons.extend(self._validate_features(values))
        if latest_close is None and self._number(values.get("vwap_session")) is not None:
            reasons.append("Cierre de mercado inválido para contrastar el VWAP; NO_TRADE.")
        attributions = self._evaluate_features(values, latest_close)
        if reasons:
            return self._no_trade(
                *reasons,
                market_regime=market_regime,
                as_of=as_of if isinstance(as_of, str) else None,
                attributions=attributions,
            )

        bullish = [item for item in attributions if item["direction"] == "CALL"]
        bearish = [item for item in attributions if item["direction"] == "PUT"]
        internal_conflicts = [
            item for item in attributions if item["direction"] == "NEUTRAL"
        ]
        if (bullish and bearish) or internal_conflicts:
            reasons.append("Evidencia técnica contradictoria; NO_TRADE.")
            return self._no_trade(
                *reasons,
                market_regime=market_regime,
                as_of=as_of if isinstance(as_of, str) else None,
                attributions=attributions,
            )

        directional = bullish or bearish
        score = sum(item["contribution"] for item in attributions)
        confidence = min(abs(score) / sum(self._WEIGHTS.values()), 1.0)
        if len(directional) < self.MIN_DIRECTIONAL_EVIDENCE:
            reasons.append(
                "Evidencia direccional insuficiente: se requieren al menos "
                f"{self.MIN_DIRECTIONAL_EVIDENCE} factores; NO_TRADE."
            )
        if confidence < self.MIN_CONFIDENCE:
            reasons.append(
                f"Consenso analítico inferior al umbral {self.MIN_CONFIDENCE:.0%}; NO_TRADE."
            )
        if reasons:
            return self._no_trade(
                *reasons,
                market_regime=market_regime,
                as_of=as_of if isinstance(as_of, str) else None,
                attributions=attributions,
                confidence=confidence,
                score=score,
            )

        direction = "CALL" if score > 0 else "PUT"
        explanation = self._explanation(attributions, score)
        if ml_prediction and ml_prediction.get("configured"):
            if not ml_prediction.get("available"):
                return self._no_trade(
                    str(ml_prediction.get("reason") or "Inferencia ML no disponible; NO_TRADE."),
                    market_regime=market_regime,
                    as_of=as_of if isinstance(as_of, str) else None,
                    attributions=attributions,
                    confidence=confidence,
                    score=score,
                )
            drift = ml_prediction.get("drift")
            if isinstance(drift, Mapping) and (
                drift.get("drifted") or not drift.get("available")
            ):
                blocked = self._no_trade(
                    str(drift.get("reason") or "Monitor de drift no disponible; NO_TRADE."),
                    market_regime=market_regime,
                    as_of=as_of if isinstance(as_of, str) else None,
                    attributions=attributions,
                    confidence=confidence,
                    score=score,
                )
                blocked["explainability"]["ml_model"] = self._model_explanation(ml_prediction)
                self._mark_model_shap(blocked["explainability"], ml_prediction)
                return blocked
            model_direction = ml_prediction.get("prediction")
            if model_direction != direction:
                blocked = self._no_trade(
                    "El modelo ML contradice el candidato determinista; NO_TRADE.",
                    market_regime=market_regime,
                    as_of=as_of if isinstance(as_of, str) else None,
                    attributions=attributions,
                    confidence=confidence,
                    score=score,
                )
                blocked["explainability"]["ml_model"] = self._model_explanation(ml_prediction)
                self._mark_model_shap(blocked["explainability"], ml_prediction)
                return blocked
            if not ml_prediction.get("confident"):
                blocked = self._no_trade(
                    "Puntuación de clase del modelo ML inferior al umbral; NO_TRADE.",
                    market_regime=market_regime,
                    as_of=as_of if isinstance(as_of, str) else None,
                    attributions=attributions,
                    confidence=confidence,
                    score=score,
                )
                blocked["explainability"]["ml_model"] = self._model_explanation(ml_prediction)
                self._mark_model_shap(blocked["explainability"], ml_prediction)
                return blocked
            explanation["ml_model"] = self._model_explanation(ml_prediction)
            self._mark_model_shap(explanation, ml_prediction)
        explanation["summary"] = (
            "Factores técnicos direccionales alineados."
            if direction == "CALL"
            else "Factores técnicos direccionales alineados a la baja."
        )
        return {
            "signal": direction,
            "confidence": confidence,
            "market_regime": market_regime,
            "risk_level": None,
            "should_trade": True,
            "reasons": [
                (
                    f"{len(directional)} factores direccionales coinciden; "
                    f"consenso analítico {confidence:.0%}."
                )
            ],
            "as_of": as_of if isinstance(as_of, str) else None,
            "explainability": explanation,
        }

    def _evaluate_features(
        self,
        values: Mapping[str, Any],
        latest_close: float | None,
    ) -> list[dict[str, Any]]:
        evidence: list[tuple[str, int, str, Any]] = []
        sma_20 = self._number(values.get("sma_20"))
        sma_50 = self._number(values.get("sma_50"))
        ema_20 = self._number(values.get("ema_20"))
        ema_50 = self._number(values.get("ema_50"))
        if None not in (sma_20, sma_50, ema_20, ema_50):
            sma_direction = 1 if sma_20 > sma_50 else -1 if sma_20 < sma_50 else 0
            ema_direction = 1 if ema_20 > ema_50 else -1 if ema_20 < ema_50 else 0
            if sma_direction and sma_direction == ema_direction:
                evidence.append(
                    (
                        "moving_average_alignment",
                        sma_direction,
                        "SMA(20/50) y EMA(20/50) alineadas",
                        {"sma_20": sma_20, "sma_50": sma_50, "ema_20": ema_20, "ema_50": ema_50},
                    )
                )
            elif sma_direction and ema_direction:
                evidence.append(
                    (
                        "moving_average_alignment",
                        0,
                        "SMA y EMA presentan direcciones distintas",
                        {"sma_20": sma_20, "sma_50": sma_50, "ema_20": ema_20, "ema_50": ema_50},
                    )
                )

        macd = values.get("macd")
        atr = self._number(values.get("atr_14"))
        if isinstance(macd, Mapping):
            histogram = self._number(macd.get("histogram"))
            if histogram is not None and atr is not None and atr > 0:
                normalized_histogram = histogram / atr
                if math.isfinite(normalized_histogram) and abs(normalized_histogram) >= 0.02:
                    evidence.append(
                        (
                            "macd_momentum",
                            self._sign(normalized_histogram),
                            "Histograma MACD relativo a ATR",
                            normalized_histogram,
                        )
                    )

        rsi = self._number(values.get("rsi_14"))
        if rsi is not None:
            if rsi >= 55:
                evidence.append(("rsi_momentum", 1, "RSI(14) en zona de momentum alcista", rsi))
            elif rsi <= 45:
                evidence.append(("rsi_momentum", -1, "RSI(14) en zona de momentum bajista", rsi))

        price_action = values.get("price_action")
        if isinstance(price_action, Mapping):
            structure = price_action.get("structure")
            if structure == "higher_highs_and_lows":
                evidence.append(
                    ("price_structure", 1, "Estructura de máximos y mínimos crecientes", structure)
                )
            elif structure == "lower_highs_and_lows":
                evidence.append(
                    ("price_structure", -1, "Estructura de máximos y mínimos decrecientes", structure)
                )

        vwap = self._number(values.get("vwap_session"))
        if latest_close is not None and vwap is not None and latest_close != vwap:
            evidence.append(
                (
                    "session_vwap",
                    1 if latest_close > vwap else -1,
                    "Precio de cierre respecto al VWAP de sesión",
                    {"close": latest_close, "vwap": vwap},
                )
            )

        return [
            {
                "feature": feature,
                "direction": "CALL" if direction > 0 else "PUT" if direction < 0 else "NEUTRAL",
                "weight": self._WEIGHTS[feature],
                "contribution": self._WEIGHTS[feature] * direction,
                "description": description,
                "value": value,
            }
            for feature, direction, description, value in evidence
        ]

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    def _validate_features(self, values: Mapping[str, Any]) -> list[str]:
        invalid = []
        for name in ("sma_20", "sma_50", "ema_20", "ema_50", "rsi_14", "atr_14"):
            if self._number(values.get(name)) is None:
                invalid.append(f"Indicador {name} ausente o inválido; NO_TRADE.")
        rsi = self._number(values.get("rsi_14"))
        if rsi is not None and not 0 <= rsi <= 100:
            invalid.append("RSI fuera de rango; NO_TRADE.")
        atr = self._number(values.get("atr_14"))
        if atr is not None and atr <= 0:
            invalid.append("ATR no positivo; NO_TRADE.")

        macd = values.get("macd")
        if not isinstance(macd, Mapping):
            invalid.append("MACD ausente o inválido; NO_TRADE.")
        else:
            macd_line = self._number(macd.get("line"))
            macd_signal = self._number(macd.get("signal"))
            histogram = self._number(macd.get("histogram"))
            if macd_line is None or macd_signal is None or histogram is None:
                invalid.append("MACD ausente o inválido; NO_TRADE.")
            elif not math.isclose(
                macd_line - macd_signal,
                histogram,
                rel_tol=1e-6,
                abs_tol=1e-12,
            ):
                invalid.append("Histograma MACD inconsistente; NO_TRADE.")

        price_action = values.get("price_action")
        if not isinstance(price_action, Mapping) or price_action.get("structure") not in {
            "higher_highs_and_lows",
            "lower_highs_and_lows",
            "mixed",
            "insufficient_swings",
        }:
            invalid.append("Estructura de precio ausente o inválida; NO_TRADE.")

        vwap = values.get("vwap_session")
        if vwap is not None and self._number(vwap) is None:
            invalid.append("VWAP de sesión inválido; NO_TRADE.")
        return invalid

    @staticmethod
    def _sign(value: float) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    @staticmethod
    def _string_reasons(value: Any) -> list[str]:
        return [reason for reason in value if isinstance(reason, str)] if isinstance(value, list) else []

    def _explanation(
        self,
        attributions: list[dict[str, Any]],
        score: float,
    ) -> dict[str, Any]:
        return {
            "method": "deterministic_additive_attribution",
            "baseline_score": 0.0,
            "score": score,
            "contributions": attributions,
            "shap_available": False,
            "shap_note": (
                "SHAP se muestra solo cuando una versión de modelo lineal "
                "revisada está activa en el Model Registry."
            ),
            "summary": "",
        }

    @staticmethod
    def _model_explanation(prediction: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "version": prediction["version"],
            "prediction": prediction["prediction"],
            "class_score_call": prediction["class_score_call"],
            "class_score_put": prediction["class_score_put"],
            "minimum_confidence": prediction["minimum_confidence"],
            "shap": prediction["explainability"],
        }

    @staticmethod
    def _mark_model_shap(
        explanation: dict[str, Any],
        prediction: Mapping[str, Any],
    ) -> None:
        model_explanation = prediction.get("explainability")
        if isinstance(model_explanation, Mapping):
            explanation["shap_available"] = True
            explanation["shap_note"] = str(model_explanation.get("note", ""))

    def _no_trade(
        self,
        *reasons: str,
        market_regime: str | None,
        as_of: str | None,
        attributions: list[dict[str, Any]],
        confidence: float = 0.0,
        score: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "signal": "NO_TRADE",
            "confidence": confidence,
            "market_regime": market_regime,
            "risk_level": None,
            "should_trade": False,
            "reasons": list(dict.fromkeys(reason for reason in reasons if reason)),
            "as_of": as_of,
            "explainability": self._explanation(attributions, score),
        }
