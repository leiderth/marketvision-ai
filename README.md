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
La primera ejecución crea `settings.env` en esa carpeta con campos vacíos para
las credenciales de los proveedores. No se incluyen credenciales ni modelos
entrenados. Sin feed válido, el resultado sigue siendo `NO_TRADE`.

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
- [x] Fase 11 — Monitor fail-closed de cambio de media de características (sin aprendizaje online)
- [x] Fase 12 — Health y métricas Prometheus de solo lectura
- [ ] Fase 13 — Ejecución asistida *(requiere decisión explícita y verificación legal — ver aviso arriba)*
- [x] Fase 14 — Recomendación offline de umbral con tuning folds y holdout (nunca autoactiva)
- [ ] Fase 15 — Instalador Windows x64 generado sin firma; firma y auto-actualización pendientes
