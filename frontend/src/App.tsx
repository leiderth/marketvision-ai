import { useEffect, useState } from "react";
import MarketChart from "./components/MarketChart";
import {
  getStatus,
  getLatestSignal,
  setPaperMode,
  activateKillSwitch,
  unlockPaperRisk,
  subscribeMarketData,
  type MarketDataSnapshot,
  type SignalResponse,
  type StatusResponse,
} from "./services/api";

const signalColor: Record<string, string> = {
  CALL: "text-mv-up",
  PUT: "text-mv-down",
  NO_TRADE: "text-mv-neutral",
};

const signalLabel: Record<string, string> = {
  CALL: "🟢 ALZA",
  PUT: "🔴 BAJA",
  NO_TRADE: "⚪ NO OPERAR",
};

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [signal, setSignal] = useState<SignalResponse | null>(null);
  const [marketData, setMarketData] = useState<MarketDataSnapshot | null>(null);
  const [marketSocketConnected, setMarketSocketConnected] = useState(false);
  const [compact, setCompact] = useState(false);
  const [controlError, setControlError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setStatus(await getStatus());
        setSignal(await getLatestSignal());
      } catch {
        // Backend aún no corriendo — esperable en Fase 1 sin uvicorn activo.
      }
    };
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(
    () => subscribeMarketData(setMarketData, setMarketSocketConnected),
    [],
  );

  const changePaperMode = async () => {
    const enabling = status?.mode !== "PAPER_TRADING_MODE";
    const confirmation = enabling
      ? window.confirm("¿Activar PAPER TRADING? Solo simula; no usa dinero real.")
      : window.confirm("¿Detener paper trading y activar el kill switch?");
    if (!confirmation) return;
    try {
      await setPaperMode(enabling);
      setControlError(null);
      setStatus(await getStatus());
    } catch (error) {
      setControlError(error instanceof Error ? error.message : "No se pudo cambiar el modo.");
    }
  };

  const stopTrading = async () => {
    if (!window.confirm("¿Activar kill switch y cancelar las órdenes paper abiertas?")) return;
    try {
      await activateKillSwitch();
      setControlError(null);
      setStatus(await getStatus());
    } catch (error) {
      setControlError(error instanceof Error ? error.message : "No se pudo activar el kill switch.");
    }
  };

  const unlockRisk = async () => {
    if (!window.confirm("¿Desbloquear paper trading? Revisa el motivo y acepta el riesgo.")) return;
    try {
      await unlockPaperRisk();
      setControlError(null);
      setStatus(await getStatus());
    } catch (error) {
      setControlError(error instanceof Error ? error.message : "No se pudo desbloquear riesgo.");
    }
  };

  return (
    <div className="min-h-screen bg-mv-bg text-white font-mono p-3 select-none">
      <div className="border border-mv-border rounded-md bg-mv-panel/90 backdrop-blur-sm max-w-sm">
        <div className="flex items-center justify-between px-3 py-2 border-b border-mv-border">
          <span className="text-xs tracking-widest text-mv-neutral">MARKETVISION AI</span>
          <button
            onClick={() => setCompact((c) => !c)}
            className="text-xs text-mv-neutral hover:text-white"
          >
            {compact ? "⧉" : "—"}
          </button>
        </div>

        <div className="px-3 py-2 text-xs text-mv-neutral border-b border-mv-border flex justify-between">
          <span>MODO: {status?.mode ?? "..."}</span>
          <span>
            FEED: {status?.data_feed.connected && status.data_feed.quality_accepted ? "🟢" : "⚪"}
          </span>
        </div>
        <div className="px-3 py-2 border-b border-mv-border flex gap-2">
          <button
            onClick={changePaperMode}
            className="flex-1 rounded border border-mv-border px-2 py-1 text-[10px] text-mv-neutral hover:text-white"
          >
            {status?.mode === "PAPER_TRADING_MODE" ? "DETENER PAPER" : "ACTIVAR PAPER"}
          </button>
          <button
            onClick={stopTrading}
            className="rounded border border-mv-down/60 px-2 py-1 text-[10px] text-mv-down"
          >
            KILL SWITCH
          </button>
        </div>
        {status?.risk.trading_locked && (
          <div className="px-3 py-2 border-b border-mv-border text-[10px] text-mv-down">
            <div>RIESGO BLOQUEADO: {status.risk.lock_reason}</div>
            <button onClick={unlockRisk} className="mt-2 rounded border border-mv-border px-2 py-1">
              DESBLOQUEO EXPLÍCITO
            </button>
          </div>
        )}
        {controlError && (
          <div role="alert" className="px-3 py-2 text-[10px] text-mv-down">
            {controlError}
          </div>
        )}

        {!compact && (
          <>
            <MarketChart
              candles={marketData?.candles ?? []}
              feedConnected={marketData?.status.connected ?? false}
              qualityAccepted={marketData?.status.quality_accepted ?? false}
              socketConnected={marketSocketConnected}
              indicators={marketData?.indicators ?? {
                status: "NO_TRADE",
                ready: false,
                as_of: null,
                samples: 0,
                required_samples: 52,
                values: {
                  sma_20: null,
                  sma_50: null,
                  ema_20: null,
                  ema_50: null,
                  rsi_14: null,
                  macd: null,
                  atr_14: null,
                  vwap_session: null,
                  price_action: null,
                  candle_pattern: null,
                  ichimoku: null,
                },
                reasons: ["Esperando datos de mercado."],
              }}
              marketRegime={marketData?.market_regime}
              economicCalendar={marketData?.economic_calendar}
            />
            <div className="px-3 py-4 text-center">
              <div className={`text-lg font-semibold ${signalColor[signal?.signal ?? "NO_TRADE"]}`}>
                {signalLabel[signal?.signal ?? "NO_TRADE"]}
              </div>
              <div className="mt-2 text-xs text-mv-neutral">
                Consenso analítico (no probabilidad): {((signal?.confidence ?? 0) * 100).toFixed(0)}%
              </div>
              <div className="mt-1 text-[10px] text-mv-neutral">
                Candidato analítico; no autoriza ejecución.
              </div>
              {status?.active_model_version && (
                <div className="mt-1 text-[10px] text-mv-neutral">
                  ML activo: {status.active_model_version}
                </div>
              )}
              {signal?.explainability && (
                <div className="mt-3 text-left text-[10px] text-mv-neutral">
                  <div>
                    {signal.explainability.summary || signal.explainability.shap_note}
                  </div>
                  {signal.explainability.contributions.slice(0, 4).map((item) => (
                    <div key={item.feature} className="mt-1">
                      · {item.description}: {item.direction}
                    </div>
                  ))}
                  {signal.explainability.ml_model && (
                    <div className="mt-2">
                      Modelo {signal.explainability.ml_model.version}: puntuación de clase{" "}
                      {(
                        Math.max(
                          signal.explainability.ml_model.class_score_call,
                          signal.explainability.ml_model.class_score_put,
                        ) * 100
                      ).toFixed(0)}
                      % (no calibrada)
                    </div>
                  )}
                </div>
              )}
              <ul className="mt-3 text-left text-[11px] text-mv-neutral space-y-1">
                {(signal?.reasons ?? []).map((r, i) => (
                  <li key={i}>· {r}</li>
                ))}
              </ul>
            </div>
          </>
        )}

        <div className="px-3 py-2 border-t border-mv-border text-[10px] text-mv-neutral">
          Análisis y paper trading únicamente. Ninguna señal garantiza resultado.
        </div>
      </div>
    </div>
  );
}
