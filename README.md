# MarketVision AI

Copiloto de análisis de mercado asistido por IA para operaciones de corto plazo.

> ⚠️ **Aviso de riesgo y estado del proyecto**
> Los instrumentos de opciones binarias / plataformas de expiración corta (p. ej. Quotex) están
> prohibidos o restringidos para clientes minoristas en varias jurisdicciones (por ejemplo la Unión
> Europea vía ESMA). Antes de usar este software para operar con dinero real, verifica la legalidad
> de este tipo de producto en tu país y el de tus usuarios, y considera asesoría legal.
>
> Este repositorio es una arquitectura de **análisis y simulación**. El `ExecutionManager`
> (`backend/execution/`) se entrega como **interfaz abstracta sin implementación de ejecución real
> contra ningún bróker**. La app arranca siempre en `ANALYSIS_MODE`. Ningún modelo, por bueno que
> sea, puede garantizar el éxito en la mayoría de las operaciones ni eliminar el riesgo de pérdida.

## ¿Qué es MarketVision AI?

Un agente de análisis de mercado en tiempo real — no un "sistema de señales mágicas". Combina
múltiples fuentes de evidencia (tendencia, momentum, volatilidad, price action, multi-timeframe) en
una decisión explicable, trata **NO OPERAR** como el resultado más frecuente y valioso del sistema
(no como una falla), y aprende de sus resultados de forma controlada (train → validation → test →
shadow deployment → producción), nunca de forma autónoma sin supervisión.

## Stack

- **Frontend**: React + TypeScript + Vite + TailwindCSS + Tauri
- **Backend**: Python + FastAPI + NumPy + Pandas + scikit-learn + PyTorch
- **Base de datos**: PostgreSQL (prod) / SQLite (dev), TimescaleDB opcional
- **Comunicación**: REST + WebSocket

## Estructura del proyecto

Ver `docs/` (a completar) y la sección 23 de la especificación técnica original para el detalle
completo de cada módulo.

## Modos de operación

| Modo | Qué hace | Estado en este repo |
|---|---|---|
| `ANALYSIS_MODE` | Solo observa y muestra señales | ✅ implementado (esqueleto) |
| `PAPER_TRADING_MODE` | Simula ejecución sin dinero real | ✅ implementado (esqueleto) |
| `AUTONOMOUS_EXECUTION_MODE` | Ejecuta operaciones reales sin confirmación por operación | ⛔ **no implementado** — ver `backend/execution/README.md` |

## Cómo empezar

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
uvicorn api.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Instalador de Windows (Tauri + API local)

El instalador NSIS incluye la interfaz y el backend FastAPI empaquetado como
recursos de la app. El backend escucha solo en `127.0.0.1:18731`; al cerrar la
app, Tauri detiene el proceso. El estado local se guarda en `%LOCALAPPDATA%\MarketVision AI`.
No se requieren credenciales de proveedor de mercado. La lectura visual comienza
después de calibrar localmente el área del gráfico; hasta entonces el resultado
es `NO OPERAR`.

Para generar el instalador x64 desde Windows con Rust/Cargo, Visual Studio C++
Build Tools y WebView2 instalados:

```powershell
cd frontend
npm install
npm run package:windows
```

El script crea un entorno Python de empaquetado aislado en `%TEMP%` e instala
las dependencias declaradas del backend; no reutiliza paquetes Python ajenos
instalados en el sistema.

El instalador `.exe` se genera bajo
`desktop\tauri\src-tauri\target\release\bundle\nsis\`. El ejecutable no está firmado
si no se configura un certificado de firma; Windows puede mostrar una advertencia
de SmartScreen. Las credenciales configuradas se guardan como texto en
`%LOCALAPPDATA%\MarketVision AI\settings.env`; protege ese archivo y no lo
compartas ni lo subas a control de versiones. La actualización automática y la
ejecución con bróker no están incluidas.

El backend Windows se empaqueta en modo `--noconsole`; sus logs y los errores
del ciclo de vida del proceso se guardan en
`%LOCALAPPDATA%\MarketVision AI\logs\backend.log`. El diario paper, las
migraciones SQLite y el CSV de muestras se guardan en
`%LOCALAPPDATA%\MarketVision AI\data\`.
En Windows, Tauri mantiene el sidecar en un Job Object con
`KILL_ON_JOB_CLOSE`, además de terminarlo y esperarlo en el cierre normal; si no
se puede asignar al Job Object, el arranque falla en vez de dejar un proceso sin
supervisión.

## Diario y aprendizaje paper

Cada orden paper aceptada se registra en SQLite con activo, timeframe,
dirección, snapshot de indicadores, régimen y confianza. Al liquidarla se
guardan el resultado, el payout opcional y la duración, y se añade una muestra
etiquetada a `paper_training.csv`. El perfil incremental de pesos y umbral es
informativo: no altera el `SignalEngine`, los límites de riesgo ni autoriza
operaciones. El panel muestra winrate por activo y la importancia incremental
de features.

El CSV puede revisarse con las herramientas offline existentes
(`backend/ml/train.py` y `backend/backtesting/walk_forward.py`). El
reentrenamiento automático por lotes, la promoción automática y el rollback
por degradación aún no están habilitados: un artefacto registrado permanece
inactivo y requiere evaluación/activación explícita. El retorno direccional
exportado para el backtest es un proxy calculado a partir del resultado paper
y del payout, no un precio de mercado observado.

El RiskManager aplica además `RISK_MAX_TRADES_PER_HOUR` (predeterminado: 5);
los límites y el estado de bloqueo siguen siendo independientes del perfil
aprendido.

`AUTO PAPER` es una simulación opcional, desactivada por defecto. Solo puede
activarse con `PAPER_TRADING_MODE` ya habilitado; evalúa la señal al cambiar
la vela validada, permite como máximo una posición abierta y la liquida con
el precio de apertura de la primera vela con timestamp igual o posterior al
vencimiento. El payout es un parámetro de simulación introducido por el
usuario; `KILL SWITCH`, el cambio a análisis y los límites de riesgo detienen
la apertura de nuevas operaciones y cancelan las posiciones paper abiertas.
No mueve el cursor ni interactúa con Quotex ni con otra ventana. Sin una
fuente verificable del estado del broker, la ejecución real permanece
deshabilitada.

El límite por operación usa `RISK_MAX_LOSS_PER_TRADE_PCT` del balance paper.
También se detiene al alcanzar `RISK_MAX_DAILY_PROFIT_PCT` (5% por defecto,
0 para deshabilitar ese objetivo), la pérdida diaria, la racha consecutiva,
las operaciones por hora o el máximo diario. El bloqueo persiste hasta un
desbloqueo explícito con confirmación en la UI. El motor paper rechaza una
nueva orden mientras haya una posición abierta.

## Visión local de pantalla

La consola captura el escritorio virtual localmente con MSS en un bucle
configurable de 1–15 FPS (`VISION_CAPTURE_FPS`, por defecto 5), conserva la
geometría y el DPI de cada monitor y solo acepta cada región calibrada si cabe
por completo en un monitor. El sidecar usa conciencia DPI PerMonitorV2; si
Windows no permite activarla o cambia la distribución/resolución/DPI, bloquea
la lectura y pide recalibrar. Ninguna captura se envía a un servicio externo.

Pulsa **DETECTAR GRÁFICO** para que una heurística local proponga regiones con
velas de colores; verifica y corrige el recuadro manualmente, ya que no es una
identificación infalible de la interfaz del broker. También puedes pulsar
**CALIBRAR ÁREA** y seleccionar el gráfico a mano. Después
selecciona en el menú y marca, también arrastrando, las regiones de OCR del
activo, temporizador, payout y monto. Indica el activo esperado y el intervalo
de vela y guarda. El activo leído se compara con ese valor; una discrepancia,
un timer/payout/monto ilegible o una lectura visual de baja calidad produce
**NO OPERAR** y puede solicitar recalibración. El análisis incluye el cuerpo y
las mechas estimados de las velas y una puntuación de calidad.

El OCR se ejecuta cada segundo y requiere lecturas repetidas coherentes antes
de validar activo, temporizador, payout y monto; hasta entonces la lectura queda
bloqueada. Las velas detectadas se guardan en la tabla SQLite
`vision_pixel_candles` (migración `005_vision_pixel_candles.sql`) y están
disponibles en `GET /api/vision/candles`. Sus OHLC son coordenadas normalizadas
derivadas de los píxeles, alineadas aproximadamente usando el temporizador y el
timeframe calibrado: **no son cotizaciones reales ni precios OHLC de mercado**.

### Fase 2: etiquetas de la próxima vela y dataset visual

Al pulsar **REGISTRAR CANDIDATO DEMO** se guarda una fotografía versionada de
features de las velas visibles y se prepara una observación para la siguiente
vela cuyo inicio se estima con el timer OCR y el timeframe. Cuando esa vela ya
cerró, el sistema exige calidad mínima de 70/100 y dos observaciones separadas
por al menos un segundo con la misma dirección visual. Solo entonces guarda
CALL/PUT/FLAT en `vision_series_samples`; lecturas insuficientes o direcciones
inestables permanecen sin etiqueta, no se inventa un resultado.

Este dataset y el journal WIN/LOSS de operaciones son independientes. Una
etiqueta visual mide únicamente si el color/cuerpo detectado coincide con la
dirección candidata; **no es el resultado de una operación Quotex ni una
cotización de mercado**. La clasificación visual FLAT se conserva por separado
y no se fuerza a WIN/LOSS. Features, capturas y etiquetas se guardan en SQLite
local (`vision_series_samples`, `vision_series_outcome_watch`); la migración
correspondiente es `006_vision_series_labels.sql`. La API de diagnóstico es
`GET /api/vision/series/dataset-status` y `GET /api/vision/series/samples`.

El pipeline de features versionado incluye la dirección/tamaño de velas,
mechas, sesgo de las últimas 5/10 velas, retornos visuales, rango, tendencia,
momentum, puntuaciones heurísticas, calidad y contexto OCR. Se conserva la base
`normalized_chart_pixels` para evitar interpretarlas como precios. En esta fase
el dataset se recolectaba y el pipeline se probaba; el entrenamiento
supervisado de esas etiquetas se describe a continuación.

El empaquetado de Windows necesita Tesseract OCR v5 de 64 bits instalado en la
máquina de compilación (por defecto `C:\Program Files\Tesseract-OCR`) o la
variable `TESSERACT_HOME`. El script incorpora `tesseract.exe`, sus DLL,
`eng.traineddata`, la configuración TSV y la licencia al instalador. La app
instalada no requiere que el usuario final instale Tesseract aparte. OCR corre
localmente en regiones pequeñas y se actualiza con menor frecuencia que la
captura visual.

Las lecturas de colores y formas son heurísticas, no OHLCV ni probabilidades.
El score de calidad y el OCR son filtros de legibilidad, no una garantía de que
la interpretación sea correcta; revisa los campos leídos antes de usar DEMO.
La tasa CALL/PUT de la UI se estima solo con el 80% cronológicamente anterior
del historial que coincide exactamente en activo, timeframe, régimen visual y
hora UTC; sus intervalos son Wilson al 95%. La decisión sigue bloqueada hasta
tener al menos 30 muestras de cada dirección en ese contexto, 100 etiquetadas
por dirección para la validación temporal global, un holdout global que supere
al baseline, payout legible y calidad de lectura de al menos 70/100. Además,
el límite inferior de confianza de la dirección candidata debe superar el
breakeven `100 / (100 + payout_pct)` más el margen configurable (5 puntos por
defecto), y debe separarse del límite superior de la dirección opuesta por al
menos 15 puntos (configurable). Desde Fase 3, el requisito de probabilidad es
además como mínimo el umbral adaptativo de confianza, que sube ante pérdidas y
solo baja gradualmente con resultados favorables. Si una condición falla,
muestra `NO OPERAR` y la razón. Estas tasas no predicen resultados futuros ni
garantizan rentabilidad.

La consola permite registrar el candidato visual y etiquetar manualmente el
resultado realmente observado en DEMO; no introduzcas resultados supuestos.
Solo se admite una decisión por activo, timeframe y ventana de vela, para
evitar inflar artificialmente el tamaño de muestra repitiendo la misma señal.

### Fase 3: aprendizaje visual supervisado

La región OCR `Resultado cerrado (opcional)` permite vigilar una operación que
el usuario abrió manualmente en DEMO. Calibra esa región sobre el estado neutral
del campo de resultado, registra el candidato, ejecuta como máximo esa operación
DEMO manualmente e indica su duración esperada antes de pulsar `VIGILAR OCR
DEMO`. El sistema espera que venza esa duración y etiqueta solo cuando dos
lecturas consecutivas de confianza alta muestran un resultado inequívoco
(`WIN/LOSS`, textos equivalentes o un P&L con signo explícito). Si no reconoce un
cierre inequívoco, no inventa una etiqueta: usa WIN/LOSS manual y corrige luego
con `CORREGIR`; `CANCELAR` detiene una vigilancia sin etiquetar. Las
correcciones quedan auditadas y reconstruyen los pesos.
Esta vigilancia no ejecuta operaciones y admite una sola decisión pendiente.

Cada resultado actualiza pesos ligeros y umbral de confianza; las pérdidas
aumentan el margen contextual sobre breakeven y las ganancias solo lo reducen
gradualmente hasta el mínimo configurado. El reentrenamiento batch requiere al
menos 100 etiquetas, evalúa un split cronológico 60/20/20 con purga por fin del
horizonte, y se programa cada 50 etiquetas nuevas o una vez al día. Un modelo
versionado se promueve solo si mejora el Brier del modelo vigente y del baseline
al menos 0.002 en el holdout. Si el modelo activo empeora más de 0.02 frente al
baseline, se revierte a la versión anterior o al baseline. Dataset y artefactos
se guardan como `vision_training.csv` y `vision_registry/` junto a los datos
locales de la app; SQLite añade las tablas de la migración
`backend/database/migrations/004_vision_learning.sql` y la columna de origen de
etiqueta. La UI presenta winrate por activo/contexto, pesos, evolución del
modelo y calibración del holdout.

Los modelos batch de esta fase son observacionales y no sustituyen la regla de
decisión contextual de la Fase 2 ni habilitan clics reales. El OCR de resultados
depende del idioma/layout del broker y de que la región calibrada contenga un
estado final legible; valida las primeras etiquetas manualmente en DEMO.

#### Modelo visual de coincidencia de la siguiente vela

Es un modelo independiente del aprendizaje WIN/LOSS paper: solo aprende de
`vision_series_samples` confirmadas, excluye FLAT y evalúa si la etiqueta visual
CALL/PUT coincidió con la siguiente vela estimada. Requiere 200 muestras válidas,
usa un split cronológico purgado 60/20/20, calibra con la ventana de validación
y calcula Brier, log-loss, error de calibración e intervalos de fiabilidad. Solo
promueve al champion si mejora fuera de muestra al baseline y al modelo vigente;
una degradación del champion activa rollback. Los intentos de entrenamiento
fallidos se difieren hasta recibir datos nuevos, evitando reintentos cada minuto.

Estado, evaluaciones y artefactos viven en SQLite y `vision_series_registry/`;
la migración es `backend/database/migrations/007_vision_series_model.sql`.
Desde **APRENDIZAJE VISUAL** se solicita evaluación manual al superar las 200
muestras. La UI muestra la probabilidad calibrada de coincidencia visual, su
IC95% cuando hay soporte, la versión y métricas OOS. No representa probabilidad
de ganar una operación, precio real ni señal de trading, y no habilita clics.

### Overlay de escritorio (Fase 4)

En Windows, la ventana transparente y siempre visible se expande al escritorio
virtual que forman todos los monitores conectados, incluidos los que tienen
coordenadas negativas. El panel queda anclado arriba a la izquierda; solo su
rectángulo recibe entrada cuando click-through está activo, y el resto de la
ventana deja interactuar con las aplicaciones que hay debajo. El panel conserva
scroll para pantallas pequeñas.

Los atajos globales son **F9** (mostrar/ocultar), **F10** (alternar
click-through) y **F12** (parada: desactiva click-through y pausa la captura);
**F8** pausa/reanuda la captura desde el panel. La parada también está disponible
como botón. Si otra aplicación ya registró una tecla, MarketVision lo anota en
`logs\backend.log`; se pueden seguir usando las acciones de la UI.

El overlay muestra señal, probabilidad contextual y motivo, calidad de lectura,
estado del RiskManager, P&L paper del día, historial y cuenta regresiva de una
operación DEMO bajo seguimiento OCR. No detecta automáticamente operaciones de
Quotex ni sigue la geometría de una ventana concreta del navegador: la captura
calibrada y el navegador deben permanecer alineados en el escritorio. Los
monitores con escalados DPI distintos comparten el escalado Windows de la
ventana overlay; revisa el panel en cada monitor y recalibra la captura si
cambia la disposición, resolución o DPI. El modo click-through no ejecuta
órdenes y la ejecución real permanece bloqueada.

### Calibración de botones (Fase 5, prueba segura)

Pulsa **CALIBRAR CALL/PUT**, marca el centro de cada botón en la captura del
escritorio y guarda ambos puntos. Las coordenadas globales se validan contra
los monitores/DPI actuales y se escriben en `screen_vision.json`; un cambio de
layout o DPI bloquea la prueba hasta recalibrar. **PROBAR CURSOR CALL/PUT** solo
mueve el puntero a la coordenada almacenada: no envía clic ni abre operaciones.
Cada prueba queda registrada en `logs\backend.log`.

La ejecución real continúa bloqueada. El sistema aún no identifica con
fiabilidad la ventana concreta de Quotex, no detecta una operación abierta ni
puede confirmar en pantalla que una orden se abrió; por tanto no arma ni envía
clics y no se puede usar el movimiento de prueba como verificación de ejecución.
No hay migración SQL para esta fase: los puntos usan el archivo JSON de
calibración existente.

### Replay y recuperación (Fase 6)

En la consola, pulsa **GUARDAR CAPTURA LOCAL** cuando haya una lectura reciente
para conservar el recorte del gráfico en
`%LOCALAPPDATA%\MarketVision AI\data\vision_replay\`. Selecciónalo y pulsa
**REPRODUCIR · PAPER VISUAL** para volver a ejecutar la extracción visual sin
MSS, OCR en vivo, feed, red ni operaciones. El replay muestra tendencia, velas,
calidad y puntuaciones visuales heurísticas; siempre devuelve **NO_TRADE** y
no reproduce la probabilidad histórica ni inicia clics. Los archivos se guardan
localmente y no hay migración SQL.

El proyecto ya empaqueta el sidecar Python con PyInstaller `--noconsole`,
redirecciona sus logs a `%LOCALAPPDATA%\MarketVision AI\logs\backend.log` y
Tauri detiene el proceso al cerrar la app. Los instaladores NSIS de Windows se
generan con `cd frontend; npm run package:windows` y quedan bajo
`desktop\tauri\src-tauri\target\release\bundle\nsis\`. Para revisar o iniciar
la instalación de forma segura usa
`powershell -ExecutionPolicy Bypass -File scripts\recover_windows.ps1`; si la
ruta no aparece en el registro, pásala con `-ApplicationPath`. El script nunca
finaliza procesos ni borra la base de datos; si la interfaz está abierta pero
el backend no responde, primero pide cerrarla normalmente.

Las pruebas unitarias existentes del motor determinista y RiskManager, junto
con las de replay visual, se ejecutan desde la raíz con:

```powershell
$env:PYTHONPATH = (Resolve-Path backend).Path
python -m pytest tests\test_signal_engine.py tests\test_risk_controls.py tests\test_vision_replay.py -q
```

## Reglas invariables del sistema (no negociables)

1. Si no hay suficiente información → NO TRADE.
2. Si hay señales contradictorias → NO TRADE.
3. Si el modelo tiene baja confianza → NO TRADE.
4. Si hay error en los datos → NO TRADE.
5. Si el feed se desconecta → NO TRADE.
6. Si el sistema no puede explicar la señal → NO TRADE.
7. El `RiskManager` es inmutable: ningún modelo de IA puede modificar, ignorar o sortear sus límites.
8. Debe existir siempre un kill switch accesible para detener toda ejecución.
9. Tras pérdidas consecutivas máximas o pérdida diaria máxima, el sistema se detiene solo.
10. El escalado de capital es siempre gradual y nunca automático hacia capital completo.

## Roadmap (fases)

- [x] Fase 1 — Interfaz de escritorio (overlay + onboarding) + estructura del proyecto
- [x] Fase 2 — Conexión de mercado + DataQualityGate (Twelve Data configurable)
- [x] Fase 3 — Gráfico en tiempo real (WebSocket y velas OHLCV validadas de Twelve Data)
- [x] Fase 4 — Indicadores técnicos (tendencia, momentum, volatilidad, acción del precio, patrones, VWAP e Ichimoku)
- [x] Fase 5 — Market Regime Detection + calendario económico Trading Economics (fail-closed y cobertura de divisas)
- [x] Fase 6 — Signal Engine determinista + explicación aditiva
- [x] Fase 7 — Modelo supervisado logistic, registro JSON local, evaluación temporal y SHAP lineal (modelo inactivo hasta activación explícita)
- [x] Fase 8 — Evaluación clasificadora walk-forward (proxy direccional, no payoff binario; requiere CSV histórico)
- [x] Fase 9 — Paper Trading manual y explícito (sin dinero real, nunca ejecuta automáticamente)
- [x] Fase 10 — Risk Management con límites persistentes, bloqueo por pérdidas y kill switch manual
- [x] Fase 11 — Monitor fail-closed y perfil incremental informativo de resultados paper
- [x] Fase 12 — Health y métricas Prometheus de solo lectura
- [ ] Fase 13 — Ejecución asistida *(requiere decisión explícita y verificación legal — ver aviso arriba)*
- [x] Fase 14 — Recomendación offline de umbral con tuning folds y holdout (nunca autoactiva)
- [ ] Fase 15 — Instalador Windows x64 generado sin firma; firma y auto-actualización pendientes
