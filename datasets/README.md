# Datos históricos para entrenamiento y backtesting

No se incluyen datos inventados. Aporta datos con licencia y procedencia
verificables en CSV antes de entrenar o reportar métricas.

El comando `python -m ml.train` requiere estas columnas:

- `timestamp`: instante UTC de observación de las características.
- `label_end_timestamp`: instante UTC hasta el cual se midió el resultado de
  la etiqueta. Se usa para purgar observaciones que cruzan límites temporales.
- Las cinco características de `backend/ml/schema.py`.
- `label`: `CALL` o `PUT`, observado según una regla de resultado documentada.

El backtest (`python -m backtesting.run`) requiere además `realized_return_pct`,
el retorno observado del subyacente durante exactamente el horizonte de la
etiqueta. Su resultado es un proxy direccional, no el payout de opciones
binarias. No mezcles etiquetas de distintos horizontes, datos futuros ni
señales generadas por el mismo modelo. Incluye fuente, zona horaria, símbolo,
intervalo, periodo y definición de etiquetas al entregar el dataset para poder
auditarlo antes de entrenar.
