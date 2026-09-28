import type {
  EconomicCalendarStatus,
  MarketDataCandle,
  MarketRegimeStatus,
  TechnicalIndicators,
} from "../services/api";

type MarketChartProps = {
  candles: MarketDataCandle[];
  feedConnected: boolean;
  qualityAccepted: boolean;
  socketConnected: boolean;
  indicators: TechnicalIndicators;
  marketRegime?: MarketRegimeStatus;
  economicCalendar?: EconomicCalendarStatus;
};

const width = 340;
const height = 176;
const plotTop = 10;
const plotBottom = 150;
const plotLeft = 4;
const plotRight = 278;

function formatPrice(value: number): string {
  return value.toLocaleString(undefined, { maximumSignificantDigits: 7 });
}

export default function MarketChart({
  candles,
  feedConnected,
  qualityAccepted,
  socketConnected,
  indicators,
  marketRegime,
  economicCalendar,
}: MarketChartProps) {
  const visibleCandles = candles.slice(-48);
  const minimum = Math.min(...visibleCandles.map((candle) => candle.low));
  const maximum = Math.max(...visibleCandles.map((candle) => candle.high));
  const range = maximum - minimum || Math.max(Math.abs(maximum) * 0.01, 1);
  const y = (price: number) => plotBottom - ((price - minimum) / range) * (plotBottom - plotTop);
  const slotWidth = (plotRight - plotLeft) / Math.max(visibleCandles.length, 1);
  const bodyWidth = Math.max(2, Math.min(8, slotWidth * 0.58));
  const movingAverage = (period: number) =>
    visibleCandles.map((_, index) => {
      const end = candles.length - visibleCandles.length + index + 1;
      if (end < period) return null;
      const average = candles.slice(end - period, end).reduce((sum, candle) => sum + candle.close, 0) / period;
      return `${plotLeft + slotWidth * (index + 0.5)},${y(average)}`;
    }).filter((point): point is string => point !== null).join(" ");

  const emptyMessage = !socketConnected
    ? "API DE MERCADO DESCONECTADA"
    : !feedConnected
      ? "ESPERANDO FEED AUTORIZADO"
      : visibleCandles.length === 0
        ? "ESPERANDO VELAS"
        : !qualityAccepted
          ? "ÚLTIMO LOTE RECHAZADO · DATOS ANTERIORES"
          : null;

  return (
    <section className="border-b border-mv-border px-3 py-3" aria-label="Gráfico de mercado en tiempo real">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] tracking-widest text-mv-neutral">MERCADO · VELAS</span>
        <span className={`text-[9px] ${feedConnected && qualityAccepted ? "text-mv-up" : "text-mv-neutral"}`}>
          {feedConnected && qualityAccepted ? "EN VIVO" : "SIN DATOS VÁLIDOS"}
        </span>
      </div>
      <div className="mb-2 rounded border border-mv-border bg-black/20 px-2 py-1.5 text-[9px]">
        <div className="flex items-center justify-between text-mv-neutral">
          <span>RÉGIMEN</span>
          <span className={marketRegime?.trade_blocked ? "text-mv-down" : "text-mv-up"}>
            {marketRegime?.regime.replace(/_/g, " ").toUpperCase() ?? "DESCONOCIDO"}
            {" · "}
            {marketRegime?.trade_blocked ? "BLOQUEADO" : "SIN BLOQUEOS"}
          </span>
        </div>
        <div className="mt-1 text-mv-neutral">
          Calendario:{" "}
          {economicCalendar?.available
            ? `${economicCalendar.active_events.length} eventos activos · ${economicCalendar.upcoming_events.length} próximos`
            : "no disponible · NO TRADE"}
        </div>
        {economicCalendar?.active_events.slice(0, 2).map((event) => (
          <div key={`${event.timestamp}-${event.name}`} className="mt-1 text-mv-down">
            · {event.currency} · {event.name} · {event.minutes_until > 0 ? "en" : "hace"}{" "}
            {Math.abs(event.minutes_until)} min
          </div>
        ))}
        {economicCalendar?.upcoming_events.slice(0, 2).map((event) => (
          <div key={`${event.timestamp}-${event.name}`} className="mt-1 text-mv-neutral">
            · Próximo: {event.currency} · {event.name} · en {event.minutes_until} min
          </div>
        ))}
        {marketRegime?.reasons.slice(0, 2).map((reason) => (
          <div key={reason} className="mt-1 text-mv-down">· {reason}</div>
        ))}
      </div>

      <svg
        className="block h-auto w-full"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={
          visibleCandles.length
            ? `Gráfico de velas con ${visibleCandles.length} velas`
            : "Gráfico de mercado vacío"
        }
      >
        {[0, 1, 2, 3].map((line) => {
          const fraction = line / 3;
          const price = maximum - range * fraction;
          const lineY = plotTop + (plotBottom - plotTop) * fraction;
          return (
            <g key={line}>
              <line
                x1={plotLeft}
                x2={plotRight}
                y1={lineY}
                y2={lineY}
                stroke="rgb(55 65 81 / 65%)"
                strokeDasharray="3 4"
              />
              {visibleCandles.length > 0 && (
                <text x={plotRight + 6} y={lineY + 3} fill="#9ca3af" fontSize="8">
                  {formatPrice(price)}
                </text>
              )}
            </g>
          );
        })}

        {visibleCandles.map((candle, index) => {
          const centerX = plotLeft + slotWidth * (index + 0.5);
          const rising = candle.close >= candle.open;
          const color = rising ? "#22c55e" : "#ef4444";
          const bodyTop = Math.min(y(candle.open), y(candle.close));
          const bodyHeight = Math.max(1, Math.abs(y(candle.open) - y(candle.close)));
          return (
            <g key={`${candle.timestamp}-${index}`}>
              <line
                x1={centerX}
                x2={centerX}
                y1={y(candle.high)}
                y2={y(candle.low)}
                stroke={color}
                strokeWidth="1"
              />
              <rect
                x={centerX - bodyWidth / 2}
                y={bodyTop}
                width={bodyWidth}
                height={bodyHeight}
                fill={color}
                rx="0.5"
              />
            </g>
          );
        })}

        {indicators.ready && visibleCandles.length > 0 && (
          <>
            <polyline points={movingAverage(20)} fill="none" stroke="#facc15" strokeWidth="1" />
            <polyline points={movingAverage(50)} fill="none" stroke="#38bdf8" strokeWidth="1" />
          </>
        )}

        {emptyMessage && (
          <text
            x={(plotLeft + plotRight) / 2}
            y={(plotTop + plotBottom) / 2}
            fill="#9ca3af"
            fontSize="9"
            textAnchor="middle"
          >
            {emptyMessage}
          </text>
        )}

        {visibleCandles.length > 0 && (
          <>
            <text x={plotLeft} y={height - 7} fill="#9ca3af" fontSize="8">
              {new Date(visibleCandles[0].timestamp).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </text>
            <text x={plotRight} y={height - 7} fill="#9ca3af" fontSize="8" textAnchor="end">
              {new Date(visibleCandles[visibleCandles.length - 1].timestamp).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </text>
          </>
        )}
      </svg>

      {visibleCandles.length > 0 && (
        <div className="mt-1 flex justify-between text-[9px] text-mv-neutral">
          <span>{visibleCandles[visibleCandles.length - 1].symbol}</span>
          <span>Cierre: {formatPrice(visibleCandles[visibleCandles.length - 1].close)}</span>
        </div>
      )}
      <div className="mt-2 grid grid-cols-3 gap-y-1 text-[9px] text-mv-neutral">
        <span>RSI 14</span>
        <span>MACD 12/26/9</span>
        <span>ATR 14</span>
        {indicators.ready ? (
          <>
            <span>{indicators.values.rsi_14?.toFixed(1)}</span>
            <span>{indicators.values.macd?.histogram.toFixed(5)}</span>
            <span>{indicators.values.atr_14?.toFixed(5)}</span>
          </>
        ) : (
          <span className="col-span-3">
            {indicators.samples}/{indicators.required_samples} velas · datos insuficientes o no válidos
          </span>
        )}
      </div>
      {indicators.ready && (
        <div className="mt-1 text-[9px] text-mv-neutral">
          SMA 20 <span className="text-yellow-400">—</span>
          {" · "}SMA 50 <span className="text-sky-400">—</span>
          {" · "}VWAP sesión: {indicators.values.vwap_session?.toFixed(5) ?? "N/D"}
        </div>
      )}
      {indicators.ready && indicators.values.price_action && (
        <div className="mt-1 text-[9px] text-mv-neutral">
          Acción: {indicators.values.price_action.structure.replace(/_/g, " ")}
          {" · "}Vela: {indicators.values.candle_pattern}
          {" · "}Ichimoku T/K: {indicators.values.ichimoku?.tenkan_sen.toFixed(5)}
          /{indicators.values.ichimoku?.kijun_sen.toFixed(5)}
        </div>
      )}
    </section>
  );
}
