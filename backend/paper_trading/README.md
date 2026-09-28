# Paper trading (Fase 9)

`PaperExecutionManager` es la única implementación operativa de ejecución.
Se habilita con confirmación explícita en la UI y crea órdenes simuladas solo
si el SignalEngine presenta un candidato y el RiskManager autoriza el tamaño.
El tamaño máximo es `RISK_MAX_LOSS_PER_TRADE_PCT` del balance virtual
`RISK_PAPER_ACCOUNT_BALANCE`. La liquidación es explícita y manual en esta
fase; el porcentaje PnL paper introducido actualiza el estado de riesgo, no es
un resultado observado del mercado ni evidencia de rentabilidad.

El kill switch cancela órdenes abiertas y deja persistido el RiskManager en
estado bloqueado. Solo el usuario puede desbloquear explícitamente desde UI.
Paper orders no persisten tras un reinicio; no uses esta fase como registro
contable. No hay conexión de broker ni operación con dinero real.
