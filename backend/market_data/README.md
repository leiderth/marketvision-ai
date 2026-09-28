# MarketDataManager (Fase 2)

`MarketDataManager` coordina adaptadores que implementan `MarketDataProvider`.
`TwelveDataProvider` obtiene las velas iniciales por REST y agrega los ticks
del WebSocket oficial en velas OHLCV intradía. Configura `MARKET_DATA_PROVIDER`,
`MARKET_DATA_API_KEY` y un solo `MARKET_DATA_SYMBOL` en `.env`; `MARKET_DATA_INTERVAL`
admite `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h` y `8h`.
El tamaño de la historia inicial se define con `MARKET_DATA_HISTORY_SIZE`
(1–5000). La clave se mantiene en el backend, nunca en la UI. No se inicia la
conexión si faltan configuración o clave. No implementar scraping agresivo ni
evasión de controles de seguridad o términos de servicio.

El WebSocket de Twelve Data entrega precios, no volumen por tick; la integración
no interpreta `day_volume` como volumen de vela. Las velas en vivo declaran
volumen cero, por lo que VWAP solo está disponible cuando la ventana histórica
contiene volumen positivo de la fuente. Los ticks se agregan por el timestamp
UNIX del proveedor y el intervalo configurado. Se envía el heartbeat oficial
cada 10 segundos. El acceso al feed WebSocket depende del plan y permisos de
Twelve Data; verifica la disponibilidad y licencia de los símbolos en tu cuenta.
Referencias oficiales: [Time Series](https://twelvedata.com/docs/market-data/time-series),
[WebSocket overview](https://twelvedata.com/docs/websocket/ws-overview) y
[Real-time price](https://twelvedata.com/docs/websocket/ws-real-time-price).

`DataQualityGate` valida lotes de velas OHLCV antes de que cualquier dato
llegue al motor de análisis: timestamps con zona horaria duplicados/fuera de
orden/futuros, estructura OHLCV, cambios de precio superiores al umbral configurable
(`MARKET_DATA_MAX_PRICE_CHANGE_PCT`, por defecto 20 %) y gaps mayores al
intervalo de la vela. Un feed desconectado, lote vacío o dato inválido es
rechazado y fuerza `NO_TRADE`.

El gestor marca también el feed como no actualizado (`NO_TRADE`) si el timestamp
de la última vela aceptada supera dos intervalos, con un mínimo de tolerancia de
30 segundos.

El endpoint `/api/status/` expone la conexión y el resultado de calidad, y
`/api/signals/latest` conserva `NO_TRADE` si los datos no están aprobados. Sin
adaptador configurado, la aplicación reporta feed desconectado; no consulta
ningún proveedor externo ni usa datos de muestra.

El WebSocket `/api/market-data/ws` entrega una instantánea inicial y después
instantáneas actualizadas en cada lote aceptado. El histórico en memoria está
limitado a 240 velas por defecto. Los lotes rechazados no se dibujan y
actualizan el estado de calidad enviado a los clientes. La UI se reconecta
automáticamente al WebSocket si se interrumpe la API.

El WebSocket de la UI incluye SMA(20/50), EMA(20/50), RSI(14), MACD(12/26/9),
ATR(14), VWAP de sesión (cuando existe volumen), estructura de precio,
patrones básicos de velas e Ichimoku. Estos indicadores descriptivos solo se
declaran listos con feed conectado, datos recientes aprobados y 52 o más velas;
no generan señales ni autorizan operaciones. `ChartVisionModule` sigue siendo
opcional y no se implementa en esta fase.

Fase 5 añade `MarketRegimeDetector`, disponible en `/api/market-data/regime`
y en las instantáneas WebSocket. Clasifica tendencia alcista/bajista,
transición, lateralidad, ruido y volatilidad extrema. Expone de forma separada
flags de ruptura falsa, momentum contradictorio, spread elevado y liquidez
baja; spread o volumen ausentes se etiquetan como desconocidos y bloquean el
estado operativo por información insuficiente. Volatilidad
extrema, contradicciones, rupturas falsas, spread alto, baja liquidez y eventos
económicos de alto impacto bloquean el estado operativo.

`EconomicCalendarManager` proporciona `/api/market-data/calendar`, comprueba
la cobertura de divisas del símbolo, la ventana de eventos de alto impacto y
la vigencia de datos. `TradingEconomicsCalendarProvider` consulta el endpoint
oficial de calendario por país y rango de fechas. Configura
`ECONOMIC_CALENDAR_API_KEY` y las divisas cubiertas en
`ECONOMIC_CALENDAR_CURRENCIES`; la credencial permanece en el backend. Las
actualizaciones ocurren cada `ECONOMIC_CALENDAR_REFRESH_MINUTES`; errores,
cobertura incompleta o datos vencidos fuerzan `NO_TRADE`. Sin credenciales, el
calendario se mantiene explícitamente no disponible. `ECONOMIC_CALENDAR_EVENT_WINDOW_MINUTES`
define la ventana anterior/posterior y `ECONOMIC_CALENDAR_MAX_AGE_HOURS` su
vigencia máxima.

Referencias oficiales: [Economic Calendar API](https://docs.tradingeconomics.com/economic_calendar/),
[consulta por país y fechas](https://docs.tradingeconomics.com/economic_calendar/country/).
