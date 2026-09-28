# TechnicalIndicatorEngine (Fase 4) y MarketRegimeDetector (Fase 5)

Fase 4 calcula SMA(20/50), EMA(20/50), RSI(14), MACD(12/26/9), ATR(14), VWAP
de sesión cuando hay volumen informado, estructura de precio, patrones básicos
de velas e Ichimoku como confirmación descriptiva. Requiere un feed conectado, datos
aprobados por DataQualityGate y al menos 52 velas; de lo contrario marca los
indicadores como no disponibles/NO_TRADE. Los indicadores son descriptivos:
no producen señales de operación ni reemplazan el RiskManager.

La UI dibuja SMA(20/50) sobre el gráfico y muestra RSI, MACD y ATR. Ningún
indicador individual basta para decidir.

`MarketRegimeDetector` (Fase 5) clasifica tendencias, transición, lateralidad,
ruido y volatilidad extrema a partir de estructura direccional, eficiencia del
precio y ATR. También detecta momentum contradictorio, rupturas falsas, spread
alto cuando el proveedor suministra ese dato y posibles problemas de liquidez
cuando hay volumen comparable. La ausencia de spread/volumen se informa como
desconocida, no se inventa; al ser datos necesarios para evaluar la operativa,
su ausencia bloquea el estado operativo. Umbrales configurables:
`MARKET_REGIME_EXTREME_ATR_PCT` y `MARKET_REGIME_MAX_SPREAD_PCT`.

`EconomicCalendarManager` exige un adaptador de proveedor explícito, eventos
con zona horaria y una actualización reciente. Un evento de alto impacto
dentro de la ventana configurada bloquea el resultado operativo; sin proveedor
o con calendario vencido, régimen = `NO_TRADE`. El contrato está listo para un
proveedor autorizado. El adaptador `TradingEconomicsCalendarProvider` consulta
el endpoint oficial por país y rango de fechas cuando se configura
`ECONOMIC_CALENDAR_API_KEY`. `ECONOMIC_CALENDAR_CURRENCIES` limita los países
consultados y define la cobertura explícita; un símbolo FX que requiera una
divisa no incluida mantiene `NO_TRADE`. Los niveles oficiales de importancia
1/2/3 se mapean a bajo/medio/alto. Las fechas ISO sin offset documentadas por
el proveedor se normalizan a UTC; registros sin hora, impacto o divisa
confiables se rechazan. El backend actualiza el calendario periódicamente y
ante error conserva el último conjunto solo hasta que expire su vigencia.

## SignalEngine (Fase 6)

`SignalEngine` combina alineación SMA/EMA, MACD normalizado por ATR, RSI,
estructura del precio y posición respecto al VWAP. Solo presenta CALL/PUT si
el feed, los indicadores y el régimen pasan todos sus vetos, no hay evidencia
contradictoria, coinciden al menos tres factores y el consenso ponderado supera
65 %. En cualquier otro caso devuelve `NO_TRADE`. El campo `confidence`
representa cobertura de consenso respecto al peso total configurado; no es
probabilidad ni está calibrado como resultado futuro.

La explicación expone cada factor, dirección, peso y contribución, con un
puntaje aditivo que reconstruye exactamente la regla. Cuando existe un modelo
ML explícitamente activado, también expone SHAP lineal interventional exacto
en espacio log-odds; se aplica solo al modelo lineal y bajo el supuesto de
independencia interventional. Sin modelo activo, SHAP no está disponible.
`should_trade` solo significa que hay un candidato analítico: no autoriza
ejecución. Cualquier motor de paper trading debe consultar el `RiskManager`
como veto independiente y aquí no se implementa ejecución real.
