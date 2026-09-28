const API_PORT = import.meta.env.DEV ? 8000 : 18731;
const API_BASE = `http://127.0.0.1:${API_PORT}/api`;

export type SignalResponse = {
  signal: "CALL" | "PUT" | "NO_TRADE";
  confidence: number;
  market_regime: string | null;
  risk_level: string | null;
  should_trade: boolean;
  reasons: string[];
  as_of: string | null;
  explainability: {
    method: string;
    baseline_score: number;
    score: number;
    contributions: {
      feature: string;
      direction: string;
      weight: number;
      contribution: number;
      description: string;
      value: unknown;
    }[];
    shap_available: boolean;
    shap_note: string;
    summary: string;
    ml_model: {
      version: string;
      prediction: "CALL" | "PUT";
      class_score_call: number;
      class_score_put: number;
      minimum_confidence: number;
      shap: {
        method: string;
        output_space: string;
        expected_value: number;
        contributions: {
          feature: string;
          value: number;
          baseline: number;
          contribution: number;
        }[];
        note: string;
      };
    } | null;
  };
};

export type StatusResponse = {
  mode: string;
  data_feed: {
    connected: boolean;
    fresh: boolean;
    quality_accepted: boolean;
    note: string;
  };
  active_model_version: string | null;
  ml_model: {
    configured_model_version: string | null;
    active_model_version: string | null;
    model_loaded: boolean;
    inference_enabled: boolean;
    minimum_confidence: number;
    metrics: Record<string, number> | null;
    drift: {
      available: boolean;
      drifted: boolean;
      samples: number;
      minimum_samples: number;
      window_size: number;
      threshold_standard_deviations: number;
      drifted_features: string[];
      last_sample_at: string | null;
      reason: string;
    } | null;
    error: string | null;
    reason: string;
  };
  risk: {
    trading_day: string;
    trades_today: number;
    consecutive_losses: number;
    daily_pnl_pct: number;
    trading_locked: boolean;
    lock_reason: string | null;
  };
};

export type MarketDataCandle = {
  symbol: string;
  timestamp: string;
  interval_seconds: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type TechnicalIndicators = {
  status: "OK" | "NO_TRADE";
  ready: boolean;
  as_of: string | null;
  samples: number;
  required_samples: number;
  values: {
    sma_20: number | null;
    sma_50: number | null;
    ema_20: number | null;
    ema_50: number | null;
    rsi_14: number | null;
    macd: { line: number; signal: number; histogram: number } | null;
    atr_14: number | null;
    vwap_session: number | null;
    price_action: {
      structure: string;
      candle_bias: string;
      close_position: number;
      body_pct: number;
    } | null;
    candle_pattern: string | null;
    ichimoku: {
      tenkan_sen: number;
      kijun_sen: number;
      senkou_span_a: number;
      senkou_span_b: number;
    } | null;
  };
  reasons: string[];
};

export type EconomicEventSummary = {
  name: string;
  currency: string;
  impact: "low" | "medium" | "high";
  timestamp: string;
  minutes_until: number;
};

export type EconomicCalendarStatus = {
  available: boolean;
  fresh: boolean;
  fetched_at: string | null;
  age_seconds: number | null;
  event_window_minutes: number;
  currencies: string[];
  active_events: EconomicEventSummary[];
  upcoming_events: EconomicEventSummary[];
  events_considered: number;
  error: string | null;
  reasons: string[];
};

export type MarketRegimeStatus = {
  status: "OK" | "NO_TRADE";
  ready: boolean;
  regime: string;
  as_of: string | null;
  metrics: {
    atr_pct: number | null;
    efficiency_ratio: number | null;
    sma_separation_atr: number | null;
    spread_pct: number | null;
  };
  flags: string[];
  trade_blocked: boolean;
  reasons: string[];
};

export type MarketDataSnapshot = {
  type: "market_data";
  status: {
    connected: boolean;
    quality_status: "OK" | "NO_TRADE";
    quality_accepted: boolean;
    quality_reasons: string[];
  };
  indicators: TechnicalIndicators;
  economic_calendar: EconomicCalendarStatus;
  market_regime: MarketRegimeStatus;
  candles: MarketDataCandle[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function finiteNumber(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseIndicators(value: unknown): TechnicalIndicators | null {
  if (!isRecord(value) || !isRecord(value.values)) return null;
  const { values } = value;
  const nullableNumber = (key: string) => {
    const number = finiteNumber(values, key);
    return number ?? (values[key] === null ? null : undefined);
  };
  const sma20 = nullableNumber("sma_20");
  const sma50 = nullableNumber("sma_50");
  const ema20 = nullableNumber("ema_20");
  const ema50 = nullableNumber("ema_50");
  const rsi = nullableNumber("rsi_14");
  const atr = nullableNumber("atr_14");
  const vwapSession = nullableNumber("vwap_session");
  let priceAction: TechnicalIndicators["values"]["price_action"] = null;
  if (values.price_action !== null) {
    if (!isRecord(values.price_action)) return null;
    const closePosition = finiteNumber(values.price_action, "close_position");
    const bodyPct = finiteNumber(values.price_action, "body_pct");
    if (
      typeof values.price_action.structure !== "string" ||
      typeof values.price_action.candle_bias !== "string" ||
      closePosition === null ||
      bodyPct === null
    ) return null;
    priceAction = {
      structure: values.price_action.structure,
      candle_bias: values.price_action.candle_bias,
      close_position: closePosition,
      body_pct: bodyPct,
    };
  }
  let ichimoku: TechnicalIndicators["values"]["ichimoku"] = null;
  if (values.ichimoku !== null) {
    if (!isRecord(values.ichimoku)) return null;
    const tenkanSen = finiteNumber(values.ichimoku, "tenkan_sen");
    const kijunSen = finiteNumber(values.ichimoku, "kijun_sen");
    const senkouSpanA = finiteNumber(values.ichimoku, "senkou_span_a");
    const senkouSpanB = finiteNumber(values.ichimoku, "senkou_span_b");
    if (
      tenkanSen === null ||
      kijunSen === null ||
      senkouSpanA === null ||
      senkouSpanB === null
    ) return null;
    ichimoku = {
      tenkan_sen: tenkanSen,
      kijun_sen: kijunSen,
      senkou_span_a: senkouSpanA,
      senkou_span_b: senkouSpanB,
    };
  }
  const candlePattern = values.candle_pattern;
  let macd: TechnicalIndicators["values"]["macd"] = null;
  if (values.macd !== null) {
    if (!isRecord(values.macd)) return null;
    const line = finiteNumber(values.macd, "line");
    const signal = finiteNumber(values.macd, "signal");
    const histogram = finiteNumber(values.macd, "histogram");
    if (line === null || signal === null || histogram === null) return null;
    macd = { line, signal, histogram };
  }
  if (
    (value.status !== "OK" && value.status !== "NO_TRADE") ||
    typeof value.ready !== "boolean" ||
    (value.as_of !== null && typeof value.as_of !== "string") ||
    typeof value.samples !== "number" ||
    typeof value.required_samples !== "number" ||
    !Array.isArray(value.reasons) ||
    !value.reasons.every((reason) => typeof reason === "string") ||
    [sma20, sma50, ema20, ema50, rsi, atr, vwapSession].some(
      (number) => number === undefined,
    ) ||
    (candlePattern !== null && typeof candlePattern !== "string") ||
    (values.price_action !== null && priceAction === null) ||
    (values.ichimoku !== null && ichimoku === null)
  ) {
    return null;
  }
  return {
    status: value.status,
    ready: value.ready,
    as_of: value.as_of,
    samples: value.samples,
    required_samples: value.required_samples,
    values: {
      sma_20: sma20 ?? null,
      sma_50: sma50 ?? null,
      ema_20: ema20 ?? null,
      ema_50: ema50 ?? null,
      rsi_14: rsi ?? null,
      macd,
      atr_14: atr ?? null,
      vwap_session: vwapSession ?? null,
      price_action: priceAction,
      candle_pattern: candlePattern,
      ichimoku,
    },
    reasons: value.reasons,
  };
}

function parseEconomicCalendar(value: unknown): EconomicCalendarStatus | null {
  if (!isRecord(value)) return null;
  const parseEvents = (events: unknown): EconomicEventSummary[] | null => {
    if (!Array.isArray(events)) return null;
    const parsed: EconomicEventSummary[] = [];
    for (const event of events) {
      if (
        !isRecord(event) ||
        typeof event.name !== "string" ||
        typeof event.currency !== "string" ||
        (event.impact !== "low" && event.impact !== "medium" && event.impact !== "high") ||
        typeof event.timestamp !== "string" ||
        typeof event.minutes_until !== "number" ||
        !Number.isFinite(event.minutes_until)
      ) return null;
      parsed.push({
        name: event.name,
        currency: event.currency,
        impact: event.impact,
        timestamp: event.timestamp,
        minutes_until: event.minutes_until,
      });
    }
    return parsed;
  };
  const activeEvents = parseEvents(value.active_events);
  const upcomingEvents = parseEvents(value.upcoming_events);
  const ageSeconds = value.age_seconds === null ? null : finiteNumber(value, "age_seconds");
  const fetchedAt = value.fetched_at === null ? null : value.fetched_at;
  const windowMinutes = finiteNumber(value, "event_window_minutes");
  if (
    typeof value.available !== "boolean" ||
    typeof value.fresh !== "boolean" ||
    (fetchedAt !== null && typeof fetchedAt !== "string") ||
    (ageSeconds === null && value.age_seconds !== null) ||
    windowMinutes === null ||
    !Array.isArray(value.currencies) ||
    !value.currencies.every((currency) => typeof currency === "string") ||
    activeEvents === null ||
    upcomingEvents === null ||
    typeof value.events_considered !== "number" ||
    (value.error !== null && typeof value.error !== "string") ||
    !Array.isArray(value.reasons) ||
    !value.reasons.every((reason) => typeof reason === "string")
  ) return null;
  return {
    available: value.available,
    fresh: value.fresh,
    fetched_at: fetchedAt,
    age_seconds: ageSeconds,
    event_window_minutes: windowMinutes,
    currencies: value.currencies,
    active_events: activeEvents,
    upcoming_events: upcomingEvents,
    events_considered: value.events_considered,
    error: value.error,
    reasons: value.reasons,
  };
}

function parseMarketRegime(value: unknown): MarketRegimeStatus | null {
  if (!isRecord(value) || !isRecord(value.metrics)) return null;
  const metrics = value.metrics;
  const nullableNumber = (key: string) => {
    const number = finiteNumber(metrics, key);
    return number ?? (metrics[key] === null ? null : undefined);
  };
  const atrPct = nullableNumber("atr_pct");
  const efficiencyRatio = nullableNumber("efficiency_ratio");
  const smaSeparation = nullableNumber("sma_separation_atr");
  const spreadPct = nullableNumber("spread_pct");
  if (
    (value.status !== "OK" && value.status !== "NO_TRADE") ||
    typeof value.ready !== "boolean" ||
    typeof value.regime !== "string" ||
    (value.as_of !== null && typeof value.as_of !== "string") ||
    [atrPct, efficiencyRatio, smaSeparation, spreadPct].some(
      (number) => number === undefined,
    ) ||
    !Array.isArray(value.flags) ||
    !value.flags.every((flag) => typeof flag === "string") ||
    typeof value.trade_blocked !== "boolean" ||
    !Array.isArray(value.reasons) ||
    !value.reasons.every((reason) => typeof reason === "string")
  ) return null;
  return {
    status: value.status,
    ready: value.ready,
    regime: value.regime,
    as_of: value.as_of,
    metrics: {
      atr_pct: atrPct ?? null,
      efficiency_ratio: efficiencyRatio ?? null,
      sma_separation_atr: smaSeparation ?? null,
      spread_pct: spreadPct ?? null,
    },
    flags: value.flags,
    trade_blocked: value.trade_blocked,
    reasons: value.reasons,
  };
}

function parseSnapshot(value: unknown): MarketDataSnapshot | null {
  if (!isRecord(value) || value.type !== "market_data" || !isRecord(value.status)) {
    return null;
  }
  const status = value.status;
  const indicators = parseIndicators(value.indicators);
  const economicCalendar = parseEconomicCalendar(value.economic_calendar);
  const marketRegime = parseMarketRegime(value.market_regime);
  if (
    typeof status.connected !== "boolean" ||
    (status.quality_status !== "OK" && status.quality_status !== "NO_TRADE") ||
    typeof status.quality_accepted !== "boolean" ||
    !Array.isArray(status.quality_reasons) ||
    !status.quality_reasons.every((reason) => typeof reason === "string") ||
    !Array.isArray(value.candles) ||
    indicators === null ||
    economicCalendar === null ||
    marketRegime === null
  ) {
    return null;
  }

  const candles: MarketDataCandle[] = [];
  for (const item of value.candles) {
    if (!isRecord(item) || typeof item.symbol !== "string" || typeof item.timestamp !== "string") {
      return null;
    }
    const intervalSeconds = finiteNumber(item, "interval_seconds");
    const open = finiteNumber(item, "open");
    const high = finiteNumber(item, "high");
    const low = finiteNumber(item, "low");
    const close = finiteNumber(item, "close");
    const volume = finiteNumber(item, "volume");
    if (
      intervalSeconds === null ||
      open === null ||
      high === null ||
      low === null ||
      close === null ||
      volume === null
    ) return null;
    candles.push({
      symbol: item.symbol,
      timestamp: item.timestamp,
      interval_seconds: intervalSeconds,
      open,
      high,
      low,
      close,
      volume,
    });
  }

  return {
    type: "market_data",
    status: {
      connected: status.connected,
      quality_status: status.quality_status,
      quality_accepted: status.quality_accepted,
      quality_reasons: status.quality_reasons,
    },
    indicators,
    economic_calendar: economicCalendar,
    market_regime: marketRegime,
    candles,
  };
}

export function subscribeMarketData(
  onSnapshot: (snapshot: MarketDataSnapshot) => void,
  onConnectionChange: (connected: boolean) => void,
): () => void {
  let stopped = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: number | undefined;
  const url = new URL("/api/market-data/ws", API_BASE);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";

  const connect = () => {
    if (stopped) return;
    socket = new WebSocket(url);
    socket.onopen = () => onConnectionChange(true);
    socket.onmessage = (event: MessageEvent<string>) => {
      try {
        const snapshot = parseSnapshot(JSON.parse(event.data) as unknown);
        if (snapshot) onSnapshot(snapshot);
      } catch {
        onConnectionChange(false);
      }
    };
    socket.onerror = () => onConnectionChange(false);
    socket.onclose = () => {
      onConnectionChange(false);
      if (!stopped) reconnectTimer = window.setTimeout(connect, 2000);
    };
  };
  connect();

  return () => {
    stopped = true;
    if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
    socket?.close();
  };
}

export async function getStatus(): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/status/`);
  if (!res.ok) throw new Error(`Status request failed: ${res.status}`);
  return res.json();
}

export async function getLatestSignal(): Promise<SignalResponse> {
  const res = await fetch(`${API_BASE}/signals/latest`);
  if (!res.ok) throw new Error(`Signal request failed: ${res.status}`);
  return res.json();
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    const detail = payload?.detail;
    throw new Error(
      typeof detail === "string" ? detail : `Request failed: ${res.status}`,
    );
  }
  return res.json();
}

export function setPaperMode(enabled: boolean): Promise<{ mode: string }> {
  return postJson("/risk/paper-mode", {
    enabled,
    confirmation: "PAPER_TRADING_ONLY",
  });
}

export function activateKillSwitch(): Promise<{ mode: string }> {
  return postJson("/risk/kill-switch");
}

export function unlockPaperRisk(): Promise<StatusResponse["risk"]> {
  return postJson("/risk/unlock", { confirmation: "UNLOCK_PAPER_TRADING" });
}
