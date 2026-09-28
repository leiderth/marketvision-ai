# BacktestingEngine (Fase 8)

`python -m backtesting.run C:\ruta\dataset.csv --folds 3
--min-confidence 0.65 --transaction-cost-pct 0.01` ejecuta evaluación
expanding walk-forward con scaler y clasificador ajustados solo con la ventana
de entrenamiento de cada fold. El CSV exige timestamps de observación y fin de
etiqueta, las cinco características de `ml.schema.FEATURE_NAMES`,
`label` (CALL/PUT) y `realized_return_pct` observado durante el horizonte. Cada
etiqueta se purga si su horizonte se cruza con el comienzo de test; también se
comprueba que la etiqueta concuerde con el signo del retorno realizado.

Las salidas incluyen accuracy/balanced accuracy por fold, cobertura con
abstención por umbral, win rate, profit factor, retorno acumulado y drawdown
con costo fijo configurable. **El retorno es únicamente un proxy direccional
close-to-close del subyacente, no el payoff de opciones binarias ni una
simulación de bróker.** No modela payout, latencia, spread variable, slippage,
rechazos ni expiración específica del producto; las puntuaciones no están
calibradas. Sin dataset válido no se produce evaluación y nunca se generan
datos sintéticos para reportar resultados.

## Optimización de umbral (Fase 14)

`python -m backtesting.optimize C:\ruta\dataset.csv --folds 4
--transaction-cost-pct 0.01` compara los umbrales predefinidos en folds de
ajuste y reserva exclusivamente el último fold como holdout final. Requiere al
menos tres folds, suficientes operaciones de ajuste y los mismos datos reales
con etiquetas/horizonte auditables. Devuelve una recomendación y métricas del
holdout; **no cambia `ML_MIN_CONFIDENCE`, no activa modelos ni ajusta el sistema
automáticamente**. Los resultados se deben revisar y validar con más periodos
antes de cualquier cambio manual.
