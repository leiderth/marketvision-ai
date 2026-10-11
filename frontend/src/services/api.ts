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
  risk_limits: {
    max_loss_per_trade_pct: number;
    max_daily_loss_pct: number;
    max_daily_profit_pct: number;
    max_consecutive_losses: number;
    max_trades_per_day: number;
    max_trades_per_hour: number;
    paper_account_balance: number;
  };
};

export type LearningDashboard = {
  closed_trades: number;
  dataset_samples: number;
  by_asset: Record<string, { trades: number; wins: number; win_rate: number; pnl_pct: number }>;
  by_strategy: Record<string, { trades: number; wins: number; win_rate: number; pnl_pct: number }>;
  by_hour_utc: Record<string, { trades: number; wins: number; win_rate: number; pnl_pct: number }>;
  online_learning: {
    weights: Record<string, number>;
    confidence_threshold: number;
    updates: number;
    updated_at: string | null;
    feature_importance: { feature: string; importance: number; weight: number }[];
  };
  model_evolution: {
    model_version: string;
    evaluated_at: string;
    metric_name: string;
    metric_value: number;
    decision: string;
  }[];
  batch_training: {
    enabled: boolean;
    minimum_samples: number;
    note: string;
  };
};

export type PaperTrade = {
  order_id: string;
  asset: string;
  direction: "CALL" | "PUT";
  amount: number;
  expiration_seconds: number;
  signal_confidence: number;
  opened_at?: string;
  settled_at?: string;
  duration_seconds?: number;
  outcome?: "win" | "loss";
  pnl_pct?: number;
  status?: string;
};

export type PaperOrders = {
  open: PaperTrade[];
  closed: PaperTrade[];
};

export type VisionDecisionItem = {
  decision_id: string;
  created_at: string;
  asset: string;
  direction: "CALL" | "PUT";
  call_score: number;
  put_score: number;
  timeframe_seconds: number | null;
  signal_window: string | null;
  confidence_pct: number | null;
  market_regime: string;
  outcome: "win" | "loss" | null;
  outcome_source: "manual" | "screen_ocr" | "manual_correction" | null;
  payout_pct: number | null;
  duration_seconds: number | null;
  source: string;
};

export type ScreenRect = { left: number; top: number; width: number; height: number };
export type VisionChartCandidate = ScreenRect & {
  candle_count: number;
  visual_quality_score: number;
};
export type VisionPixelCandle = {
  asset: string;
  timeframe_seconds: number;
  candle_open_at: string;
  observed_at: string;
  candle_index: number;
  open: number;
  high: number;
  low: number;
  close: number;
  price_basis: "normalized_chart_pixels";
  chart_width: number;
  chart_height: number;
  read_quality_score: number;
  source: "local_screen_capture";
};
export type VisionSeriesDatasetStatus = {
  source: "pixel_series";
  price_basis: "normalized_chart_pixels";
  total_labeled: number;
  pending: number;
  expired: number;
  by_direction: { CALL: number; PUT: number; FLAT: number };
  model_training_enabled: boolean;
  note: string;
};
export type VisionSeriesModelStatus = {
  source: "pixel_series";
  price_basis: "normalized_chart_pixels";
  model_training_enabled: boolean;
  training_source: string;
  total_samples: number;
  minimum_samples: number;
  active_version: string | null;
  active_metrics: Record<string, number> | null;
  last_result: {
    trained: boolean;
    version?: string;
    decision?: "promoted" | "rejected";
    brier_score?: number;
    baseline_brier_score?: number;
    log_loss?: number;
    calibration_error?: number;
  } | null;
  last_attempt_reason: string | null;
  prediction: {
    probability_correct: number;
    interval_95: { lower: number; upper: number; samples: number; source: string } | null;
    calibrated: boolean;
    price_basis: "normalized_chart_pixels";
    interpretation: string;
  } | null;
  recent_evaluations: {
    model_version: string;
    evaluated_at: string;
    sample_count: number;
    validation_brier: number | null;
    brier_score: number | null;
    baseline_brier: number | null;
    log_loss: number | null;
    calibration_error: number | null;
    decision: "promoted" | "rejected" | "rollback";
  }[];
  feature_names: string[];
  note: string;
};
export type VisionOCRField = "asset" | "timer" | "payout" | "amount" | "outcome";
export type VisionButtonSide = "CALL" | "PUT";
export type VisionButtonCalibration = {
  calibrated: boolean;
  buttons?: Partial<Record<VisionButtonSide, { x: number; y: number }>>;
  screen_width?: number;
  screen_height?: number;
  dpi?: number;
  monitors?: { left: number; top: number; width: number; height: number; dpi: number }[];
  error?: string;
};
export type VisionReplayCapture = {
  capture_id: string;
  captured_at: string;
  saved_at: string;
  asset: string | null;
  timeframe_seconds: number | null;
  width: number;
  height: number;
  size_bytes: number;
};
export type VisionReplayResult = {
  source: "saved_chart_capture_replay";
  capture_id: string;
  captured_at: string;
  asset: string | null;
  timeframe_seconds: number | null;
  readable: boolean;
  read_quality_score: number;
  visual_quality_score: number;
  candle_count: number;
  trend: string;
  momentum: string;
  pattern: string;
  support_levels: number[];
  resistance_levels: number[];
  directional_scores: { CALL: number | null; PUT: number | null };
  decision: "NO_TRADE";
  reason: string;
};
export type VisionLearningDashboard = {
  labeled_decisions: number;
  dataset_samples: number;
  by_asset: Record<string, { trades: number; wins: number; win_rate: number }>;
  by_direction: Record<string, { trades: number; wins: number; win_rate: number }>;
  by_regime: Record<string, { trades: number; wins: number; win_rate: number }>;
  by_hour_utc: Record<string, { trades: number; wins: number; win_rate: number }>;
  by_context: Record<string, { trades: number; wins: number; win_rate: number }>;
  online_learning: {
    weights: Record<string, number>;
    confidence_threshold: number;
    updates: number;
    updated_at: string | null;
    feature_importance: { feature: string; importance: number; weight: number }[];
  };
  active_model_version: string | null;
  model_feature_importance: { feature: string; importance: number; weight: number }[];
  model_evolution: {
    model_version: string;
    evaluated_at: string;
    samples: number;
    brier_score: number | null;
    baseline_brier_score: number | null;
    decision: "candidate" | "promoted" | "rejected" | "rollback";
    reason: string | null;
  }[];
  calibration: {
    probability_bucket: string;
    samples: number;
    predicted_win_rate: number;
    observed_win_rate: number;
  }[];
  batch_training: {
    enabled: boolean;
    minimum_samples: number;
    retrain_every_samples: number;
    last_result: Record<string, unknown> | null;
    note: string;
  };
};
export type VisionOutcomeWatch = {
  decision_id: string;
  armed_at: string;
  expected_at: string;
};

export type VisionState = {
  source: "local_screen_capture";
  capture_fps_target?: number;
  readable: boolean;
  status: string;
  reason: string | null;
  asset: string | null;
  timeframe_seconds: number | null;
  candle_count: number;
  trend: string;
  momentum: string;
  market_regime?: string;
  support_levels: number[];
  resistance_levels: number[];
  pattern: string;
  read_quality_score: number;
  ohlc_price_basis?: "normalized_chart_pixels";
  ohlc_stored_count?: number;
  ohlc_candles?: VisionPixelCandle[];
  ohlc_storage_error?: string | null;
  visual_quality_score?: number;
  needs_recalibration?: boolean;
  dpi_aware?: boolean;
  monitor_layout?: { left: number; top: number; width: number; height: number; dpi: number }[];
  screen_change?: { status: string; reason: string };
  ocr?: {
    available: boolean;
    valid: boolean;
    fields: {
      asset: { value: string | null; normalized: string | null; matches_calibration: boolean; similarity: number; confidence: number; stable?: boolean; stable_samples?: number; error: string | null };
      timer: { text: string; seconds_remaining: number | null; confidence: number; stable?: boolean; stable_samples?: number; error: string | null };
      payout: { text: string; percent: number | null; confidence: number; stable?: boolean; stable_samples?: number; error: string | null };
      amount: { text: string; value: number | null; confidence: number; stable?: boolean; stable_samples?: number; error: string | null };
      outcome: { configured: boolean; available: boolean; text: string; result: "win" | "loss" | null; confidence: number; error: string | null; captured_at: string | null };
    };
  };
  directional_scores: { CALL: number | null; PUT: number | null };
  decision: "CALL" | "PUT" | "NO_TRADE";
  calibration_reason?: string;
  calibration_error?: string | null;
  calibration_status?: {
    calibrated: boolean;
    minimum_samples_per_direction: number;
    minimum_context_samples_per_direction: number;
    minimum_read_quality: number;
    read_quality_score: number;
    adaptive_confidence_threshold_pct: number;
    edge_margin_pct: number;
    configured_edge_margin_pct: number;
    adaptive_edge_margin_pct: number;
    break_even_pct: number | null;
    required_probability_pct: number | null;
    minimum_margin_pct: number;
    directional_margin_pct: number | null;
    visual_direction: "CALL" | "PUT" | null;
    visual_margin_pct: number | null;
    conservative_margin_pct: number | null;
    context: {
      asset: string | null;
      timeframe_seconds: number | null;
      market_regime: string | null;
      hour_utc: number | null;
      payout_pct: number | null;
    };
    contextual_estimates: Record<"CALL" | "PUT", {
      samples: number;
      wins: number;
      probability_pct: number | null;
      lower_95_pct: number | null;
      upper_95_pct: number | null;
      minimum_samples: number;
      sufficient: boolean;
    }>;
    directional_win_rates: { CALL: number | null; PUT: number | null };
    estimated_accuracy_pct: { CALL: number | null; PUT: number | null };
    validation: Record<string, {
      samples: number;
      wins: number;
      win_rate: number | null;
      validated: boolean;
      validation_samples: number;
      brier_score: number | null;
      baseline_brier_score: number | null;
      reason: string | null;
    }>;
    decision: "CALL" | "PUT" | "NO_TRADE";
    reason: string;
  };
  live_execution: { enabled: boolean; available: boolean; reason: string };
  auto_outcome_watch?: VisionOutcomeWatch | null;
  captured_at?: string;
};

export type PaperAutopilotStatus = {
  enabled: boolean;
  mode: "PAPER_ONLY";
  amount: number | null;
  payout_pct: number | null;
  max_open_positions: 1;
  last_candle: string | null;
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

export async function getLearningDashboard(): Promise<LearningDashboard> {
  const res = await fetch(`${API_BASE}/learning/dashboard`);
  if (!res.ok) throw new Error(`Learning dashboard request failed: ${res.status}`);
  return res.json();
}

export async function getPaperOrders(): Promise<PaperOrders> {
  const res = await fetch(`${API_BASE}/paper/orders`);
  if (!res.ok) throw new Error(`Paper orders request failed: ${res.status}`);
  return res.json();
}

export async function getPaperAutopilot(): Promise<PaperAutopilotStatus> {
  const res = await fetch(`${API_BASE}/paper/autopilot`);
  if (!res.ok) throw new Error(`Paper autopilot request failed: ${res.status}`);
  return res.json();
}

export async function getVisionState(): Promise<VisionState> {
  const res = await fetch(`${API_BASE}/vision/state`);
  if (!res.ok) throw new Error(`Vision state request failed: ${res.status}`);
  return res.json();
}

export async function setVisionThresholds(
  edge_margin_pct: number,
  minimum_margin_pct: number,
): Promise<{ edge_margin_pct: number; minimum_margin_pct: number }> {
  const res = await fetch(`${API_BASE}/vision/thresholds`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edge_margin_pct, minimum_margin_pct }),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error(typeof payload?.detail === "string" ? payload.detail : `Threshold update failed: ${res.status}`);
  }
  return res.json();
}

export async function setVisionCapturePaused(paused: boolean): Promise<{ paused: boolean }> {
  const res = await fetch(`${API_BASE}/vision/capture-state`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paused }),
  });
  if (!res.ok) throw new Error(`Capture control failed: ${res.status}`);
  return res.json();
}

export async function getDesktopPreview(): Promise<{
  image: string; left: number; top: number; width: number; height: number; dpi: number;
  monitors: { left: number; top: number; width: number; height: number; dpi: number }[];
  dpi_aware: boolean;
  dpi_error: string | null;
}> {
  const res = await fetch(`${API_BASE}/vision/desktop-preview`);
  if (!res.ok) throw new Error(`Desktop capture failed: ${res.status}`);
  return res.json();
}

export async function detectVisionChartCandidates(): Promise<{
  image: string; left: number; top: number; width: number; height: number; dpi: number;
  monitors: { left: number; top: number; width: number; height: number; dpi: number }[];
  dpi_aware: boolean;
  dpi_error: string | null;
  chart_candidates: VisionChartCandidate[];
}> {
  const res = await fetch(`${API_BASE}/vision/detect-chart`);
  if (!res.ok) throw new Error(`Chart detection failed: ${res.status}`);
  return res.json();
}

export async function getVisionPixelCandles(options: {
  asset?: string;
  timeframe_seconds?: number;
  limit?: number;
} = {}): Promise<{
  source: "local_screen_capture";
  price_basis: "normalized_chart_pixels";
  candles: VisionPixelCandle[];
}> {
  const params = new URLSearchParams();
  if (options.asset) params.set("asset", options.asset);
  if (options.timeframe_seconds !== undefined) {
    params.set("timeframe_seconds", String(options.timeframe_seconds));
  }
  if (options.limit !== undefined) params.set("limit", String(options.limit));
  const query = params.size ? `?${params.toString()}` : "";
  const res = await fetch(`${API_BASE}/vision/candles${query}`);
  if (!res.ok) throw new Error(`Vision candle history request failed: ${res.status}`);
  return res.json();
}

export async function getVisionSeriesDatasetStatus(): Promise<VisionSeriesDatasetStatus> {
  const res = await fetch(`${API_BASE}/vision/series/dataset-status`);
  if (!res.ok) throw new Error(`Vision series dataset request failed: ${res.status}`);
  return res.json();
}

export async function getVisionSeriesModelStatus(): Promise<VisionSeriesModelStatus> {
  const res = await fetch(`${API_BASE}/vision/series/model`);
  if (!res.ok) throw new Error(`Vision series model request failed: ${res.status}`);
  return res.json();
}

export async function trainVisionSeriesModel(): Promise<Record<string, unknown>> {
  return postJson("/vision/series/model/train", {
    confirmation: "EVALUATE_VISUAL_SERIES_MODEL",
  });
}

export async function getChartPreview(): Promise<{ image: string }> {
  const res = await fetch(`${API_BASE}/vision/chart-preview`);
  if (!res.ok) throw new Error(`Chart preview request failed: ${res.status}`);
  return res.json();
}

export async function saveVisionReplayCapture(): Promise<VisionReplayCapture> {
  return postJson("/vision/replay/captures", {});
}

export async function getVisionReplayCaptures(): Promise<VisionReplayCapture[]> {
  const res = await fetch(`${API_BASE}/vision/replay/captures`);
  if (!res.ok) throw new Error(`Replay capture list failed: ${res.status}`);
  const body: { captures: VisionReplayCapture[] } = await res.json();
  return body.captures;
}

export async function runVisionReplayCapture(captureId: string): Promise<VisionReplayResult> {
  return postJson(`/vision/replay/captures/${encodeURIComponent(captureId)}/run`, {});
}

export async function calibrateVisionRegion(body: {
  left: number;
  top: number;
  width: number;
  height: number;
  asset: string;
  timeframe_seconds: number;
  ocr_regions: Record<Exclude<VisionOCRField, "outcome">, ScreenRect> &
    Partial<Record<"outcome", ScreenRect>>;
  click_call?: [number, number] | null;
  click_put?: [number, number] | null;
}): Promise<unknown> {
  return postJson("/vision/calibration", body);
}

export async function getVisionButtonCalibration(): Promise<VisionButtonCalibration> {
  const res = await fetch(`${API_BASE}/vision/calibration`);
  if (!res.ok) throw new Error(`Button calibration request failed: ${res.status}`);
  return res.json();
}

export async function calibrateVisionButtons(body: {
  click_call: [number, number];
  click_put: [number, number];
}): Promise<VisionButtonCalibration> {
  return postJson("/vision/calibration/buttons", body);
}

export async function probeVisionButton(side: VisionButtonSide): Promise<{
  side: VisionButtonSide;
  positioned: boolean;
  clicked: false;
  x: number;
  y: number;
  message: string;
}> {
  return postJson("/vision/calibration/buttons/probe", { side });
}

export async function recordVisionDecision(): Promise<{
  decision_id: string;
  candidate_direction: "CALL" | "PUT";
  duplicate?: boolean;
  series_labeling?: {
    watching: boolean;
    target_candle_open_at?: string;
    reason?: string;
    label_source: "pixel_series";
  };
}> {
  return postJson("/vision/decisions");
}

export async function getVisionDecisions(): Promise<{
  items: VisionDecisionItem[];
  calibration: VisionState["calibration_status"];
  auto_outcome_watch: VisionOutcomeWatch | null;
}> {
  const res = await fetch(`${API_BASE}/vision/decisions`);
  if (!res.ok) throw new Error(`Vision decision history failed: ${res.status}`);
  return res.json();
}

export async function recordVisionOutcome(
  decisionId: string,
  outcome: "win" | "loss",
  metadata: { payout_pct?: number; duration_seconds?: number } = {},
): Promise<unknown> {
  return postJson(`/vision/decisions/${encodeURIComponent(decisionId)}/outcome`, {
    outcome,
    ...metadata,
  });
}

export async function watchVisionOutcome(
  decisionId: string,
  duration_seconds: number,
): Promise<VisionOutcomeWatch> {
  return postJson(`/vision/decisions/${encodeURIComponent(decisionId)}/watch-outcome`, {
    duration_seconds,
  });
}

export async function cancelVisionOutcomeWatch(decisionId: string): Promise<{ cancelled: boolean }> {
  const res = await fetch(`${API_BASE}/vision/decisions/${encodeURIComponent(decisionId)}/watch-outcome`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Outcome watch cancellation failed: ${res.status}`);
  return res.json();
}

export async function correctVisionOutcome(
  decisionId: string,
  outcome: "win" | "loss",
): Promise<unknown> {
  const res = await fetch(`${API_BASE}/vision/decisions/${encodeURIComponent(decisionId)}/outcome`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ outcome, note: "Corrección manual desde MarketVision AI." }),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error(typeof payload?.detail === "string" ? payload.detail : `Outcome correction failed: ${res.status}`);
  }
  return res.json();
}

export async function getVisionLearning(): Promise<VisionLearningDashboard> {
  const res = await fetch(`${API_BASE}/vision/learning`);
  if (!res.ok) throw new Error(`Vision learning request failed: ${res.status}`);
  return res.json();
}

export function setPaperAutopilot(
  enabled: boolean,
  amount: number,
  payoutPct: number,
): Promise<PaperAutopilotStatus> {
  return postJson("/paper/autopilot", {
    enabled,
    amount,
    payout_pct: payoutPct,
    confirmation: "PAPER_AUTOPILOT_ONLY",
  });
}

export function submitPaperOrder(amount: number, expirationSeconds: number): Promise<unknown> {
  return postJson("/paper/orders", {
    amount,
    expiration_seconds: expirationSeconds,
    confirmation: "SIMULATE_ONLY",
  });
}

export function settlePaperOrder(
  orderId: string,
  pnlPct: number,
  payoutPct: number | null,
): Promise<PaperTrade> {
  return postJson(`/paper/orders/${encodeURIComponent(orderId)}/settle`, {
    pnl_pct: pnlPct,
    payout_pct: payoutPct,
    confirmation: "SIMULATE_ONLY",
  });
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
