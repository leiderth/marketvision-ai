import { useEffect, useRef, useState, type MouseEvent } from "react";
import {
  calibrateVisionRegion,
  calibrateVisionButtons,
  detectVisionChartCandidates,
  cancelVisionOutcomeWatch,
  correctVisionOutcome,
  getChartPreview,
  getDesktopPreview,
  getVisionDecisions,
  getVisionLearning,
  getVisionSeriesDatasetStatus,
  getVisionSeriesModelStatus,
  getVisionState,
  getVisionReplayCaptures,
  getStatus,
  trainVisionSeriesModel,
  setPaperMode,
  activateKillSwitch,
  unlockPaperRisk,
  getVisionButtonCalibration,
  recordVisionDecision,
  recordVisionOutcome,
  probeVisionButton,
  runVisionReplayCapture,
  saveVisionReplayCapture,
  setVisionCapturePaused,
  setVisionThresholds,
  watchVisionOutcome,
  type StatusResponse,
  type ScreenRect,
  type VisionChartCandidate,
  type VisionPixelCandle,
  type VisionDecisionItem,
  type VisionButtonCalibration,
  type VisionButtonSide,
  type VisionLearningDashboard,
  type VisionSeriesDatasetStatus,
  type VisionSeriesModelStatus,
  type VisionOutcomeWatch,
  type VisionReplayCapture,
  type VisionReplayResult,
  type VisionOCRField,
  type VisionState,
} from "../services/api";

type DesktopPreview = {
  image: string;
  left: number;
  top: number;
  width: number;
  height: number;
  dpi: number;
  monitors: { left: number; top: number; width: number; height: number; dpi: number }[];
  dpi_aware: boolean;
  dpi_error: string | null;
  chart_candidates?: VisionChartCandidate[];
};

type ScreenSelection = { left: number; top: number; width: number; height: number };
type SelectionMode = "chart" | VisionOCRField | "call" | "put";

const ocrFieldLabels: Record<VisionOCRField, string> = {
  asset: "Activo",
  timer: "Temporizador",
  payout: "Payout",
  amount: "Monto",
  outcome: "Resultado cerrado (opcional)",
};

const buttonClass =
  "rounded border border-mv-border px-2 py-1 text-[10px] text-mv-neutral hover:text-white disabled:opacity-40";

export default function VisionConsole() {
  const [vision, setVision] = useState<VisionState | null>(null);
  const [systemStatus, setSystemStatus] = useState<StatusResponse | null>(null);
  const [preview, setPreview] = useState<DesktopPreview | null>(null);
  const [chartCandidates, setChartCandidates] = useState<VisionChartCandidate[]>([]);
  const [chartImage, setChartImage] = useState<string | null>(null);
  const [history, setHistory] = useState<VisionDecisionItem[]>([]);
  const [learning, setLearning] = useState<VisionLearningDashboard | null>(null);
  const [seriesDataset, setSeriesDataset] = useState<VisionSeriesDatasetStatus | null>(null);
  const [seriesModel, setSeriesModel] = useState<VisionSeriesModelStatus | null>(null);
  const [buttonCalibration, setButtonCalibration] = useState<VisionButtonCalibration | null>(null);
  const [replayCaptures, setReplayCaptures] = useState<VisionReplayCapture[]>([]);
  const [selectedReplayId, setSelectedReplayId] = useState("");
  const [replayResult, setReplayResult] = useState<VisionReplayResult | null>(null);
  const [outcomeWatch, setOutcomeWatch] = useState<VisionOutcomeWatch | null>(null);
  const [selection, setSelection] = useState<ScreenSelection | null>(null);
  const [ocrRegions, setOcrRegions] = useState<Partial<Record<VisionOCRField, ScreenRect>>>({});
  const [callPoint, setCallPoint] = useState<[number, number] | null>(null);
  const [putPoint, setPutPoint] = useState<[number, number] | null>(null);
  const [selectionMode, setSelectionMode] = useState<SelectionMode>("chart");
  const [asset, setAsset] = useState("");
  const [timeframe, setTimeframe] = useState("60");
  const [edgeMarginPct, setEdgeMarginPct] = useState(5);
  const [minimumMarginPct, setMinimumMarginPct] = useState(15);
  const [outcomePayout, setOutcomePayout] = useState("");
  const [outcomeDuration, setOutcomeDuration] = useState("");
  const [opacity, setOpacity] = useState(92);
  const [clickThrough, setClickThrough] = useState(false);
  const [panelRegionReady, setPanelRegionReady] = useState(false);
  const [currentTime, setCurrentTime] = useState(Date.now());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelNotice, setModelNotice] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const dragStart = useRef<[number, number] | null>(null);

  const refresh = async () => {
    try {
      const [nextVision, decisions, nextSystemStatus] = await Promise.all([
        getVisionState(),
        getVisionDecisions(),
        getStatus(),
      ]);
      setVision(nextVision);
      setHistory(decisions.items);
      setOutcomeWatch(decisions.auto_outcome_watch);
      setSystemStatus(nextSystemStatus);
      setApiError(null);
    } catch (requestError) {
      setApiError(requestError instanceof Error ? requestError.message : "No se pudo leer la API local.");
    }
  };

  const refreshLearning = async () => {
    try {
      setLearning(await getVisionLearning());
    } catch (requestError) {
      setApiError(requestError instanceof Error ? requestError.message : "No se pudo leer el panel de aprendizaje.");
    }
  };

  const refreshSeriesDataset = async () => {
    try {
      const [dataset, model] = await Promise.all([
        getVisionSeriesDatasetStatus(),
        getVisionSeriesModelStatus(),
      ]);
      setSeriesDataset(dataset);
      setSeriesModel(model);
    } catch (requestError) {
      setApiError(requestError instanceof Error ? requestError.message : "No se pudo leer el dataset visual.");
    }
  };

  const evaluateSeriesModel = async () => {
    setBusy(true);
    try {
      const result = await trainVisionSeriesModel();
      setModelNotice(typeof result.reason === "string" ? result.reason : "Evaluación de modelo finalizada.");
      await refreshSeriesDataset();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo evaluar el modelo visual.");
    } finally {
      setBusy(false);
    }
  };

  const refreshButtonCalibration = async () => {
    try {
      setButtonCalibration(await getVisionButtonCalibration());
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo leer la calibración CALL/PUT.");
    }
  };

  const refreshReplayCaptures = async () => {
    try {
      const captures = await getVisionReplayCaptures();
      setReplayCaptures(captures);
      setSelectedReplayId((current) =>
        captures.some((capture) => capture.capture_id === current)
          ? current
          : captures[0]?.capture_id ?? "",
      );
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo leer el índice de replay.");
    }
  };

  useEffect(() => {
    void refresh();
    void refreshLearning();
    void refreshSeriesDataset();
    void refreshButtonCalibration();
    void refreshReplayCaptures();
    const timer = window.setInterval(() => {
      setCurrentTime(Date.now());
      void refresh();
    }, 1000);
    const learningTimer = window.setInterval(() => {
      void refreshLearning();
      void refreshSeriesDataset();
    }, 15000);
    return () => {
      window.clearInterval(timer);
      window.clearInterval(learningTimer);
    };
  }, []);

  useEffect(() => {
    let disposed = false;
    let unlistenClickThrough: (() => void) | undefined;
    let unlistenKillSwitch: (() => void) | undefined;

    const subscribe = async () => {
      try {
        const { isTauri } = await import("@tauri-apps/api/core");
        if (!isTauri() || disposed) return;
        const { listen } = await import("@tauri-apps/api/event");
        const stopListeningClickThrough = await listen<boolean>("marketvision-click-through", (event) => {
          setClickThrough(event.payload);
        });
        if (disposed) {
          stopListeningClickThrough();
          return;
        }
        unlistenClickThrough = stopListeningClickThrough;
        const stopListeningKillSwitch = await listen("marketvision-kill-switch", () => {
          void activateKillSwitch()
            .then(() => setVisionCapturePaused(true))
            .then(refresh)
            .catch((requestError: unknown) => {
              setError(requestError instanceof Error ? requestError.message : "Falló la parada de emergencia.");
            });
        });
        if (disposed) {
          stopListeningKillSwitch();
          return;
        }
        unlistenKillSwitch = stopListeningKillSwitch;
      } catch (requestError) {
        if (!disposed) {
          setError(requestError instanceof Error ? requestError.message : "No se pudieron activar los atajos globales.");
        }
      }
    };

    void subscribe();
    return () => {
      disposed = true;
      unlistenClickThrough?.();
      unlistenKillSwitch?.();
    };
  }, []);

  useEffect(() => {
    let disposed = false;
    let observer: ResizeObserver | undefined;
    let previousRect = "";

    const observePanel = async () => {
      try {
        const { invoke, isTauri } = await import("@tauri-apps/api/core");
        if (!isTauri() || disposed) return;
        const panel = panelRef.current;
        if (!panel) return;

        const reportRegion = () => {
          const rect = panel.getBoundingClientRect();
          const key = [rect.left, rect.top, rect.width, rect.height]
            .map((value) => value.toFixed(2))
            .join(",");
          if (key === previousRect || rect.width <= 0 || rect.height <= 0) return;
          previousRect = key;
          void invoke("set_overlay_interaction_region", {
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height,
          })
            .then(() => {
              if (!disposed) setPanelRegionReady(true);
            })
            .catch((requestError: unknown) => {
              if (!disposed) {
                setPanelRegionReady(false);
                setError(requestError instanceof Error ? requestError.message : "No se pudo actualizar la región interactiva.");
              }
            });
        };

        observer = new ResizeObserver(reportRegion);
        observer.observe(panel);
        reportRegion();
      } catch (requestError) {
        if (!disposed) {
          setError(requestError instanceof Error ? requestError.message : "No se pudo medir el panel del overlay.");
        }
      }
    };

    void observePanel();
    return () => {
      disposed = true;
      observer?.disconnect();
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "F8") {
        event.preventDefault();
        void setVisionCapturePaused(vision?.status !== "capture_paused")
          .then(refresh)
          .catch((requestError: unknown) => {
            setError(requestError instanceof Error ? requestError.message : "No se pudo cambiar el estado.");
          });
      } else if (event.key === "Escape") {
        void setVisionCapturePaused(true)
          .then(refresh)
          .catch((requestError: unknown) => {
            setError(requestError instanceof Error ? requestError.message : "No se pudo pausar la captura.");
          });
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [vision?.status]);

  const takeDesktopPreview = async () => {
    setBusy(true);
    try {
      const capture = await getDesktopPreview();
      setPreview(capture);
      setChartCandidates([]);
      setSelection(null);
      setOcrRegions({});
      setCallPoint(null);
      setPutPoint(null);
      setChartImage(null);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falló la captura de pantalla.");
    } finally {
      setBusy(false);
    }
  };

  const detectDesktopCharts = async () => {
    setBusy(true);
    try {
      const capture = await detectVisionChartCandidates();
      setPreview(capture);
      const candidates = capture.chart_candidates ?? [];
      setChartCandidates(candidates);
      setOcrRegions({});
      setCallPoint(null);
      setPutPoint(null);
      setChartImage(null);
      setSelectionMode("chart");
      setSelection(candidates[0]
        ? {
            left: candidates[0].left,
            top: candidates[0].top,
            width: candidates[0].width,
            height: candidates[0].height,
          }
        : null);
      setError(candidates.length
        ? `Sugerencia automática: ${candidates[0].candle_count} velas candidatas. Verifica el recuadro y corrígelo arrastrando antes de guardar.`
        : "No se detectó un gráfico con suficiente certeza. Selecciónalo manualmente en la captura.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falló la detección local del gráfico.");
    } finally {
      setBusy(false);
    }
  };

  const startButtonCalibration = async () => {
    setBusy(true);
    try {
      const capture = await getDesktopPreview();
      setPreview(capture);
      setCallPoint(null);
      setPutPoint(null);
      setSelectionMode("call");
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falló la captura para calibrar botones.");
    } finally {
      setBusy(false);
    }
  };

  const pointFromMouse = (event: MouseEvent<HTMLImageElement>): [number, number] | null => {
    if (!preview || !imageRef.current) return null;
    const rect = imageRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(preview.width, (event.clientX - rect.left) * preview.width / rect.width));
    const y = Math.max(0, Math.min(preview.height, (event.clientY - rect.top) * preview.height / rect.height));
    return [
      Math.round(preview.left + x),
      Math.round(preview.top + y),
    ];
  };

  const onImageMouseDown = (event: MouseEvent<HTMLImageElement>) => {
    const point = pointFromMouse(event);
    if (!point) return;
    dragStart.current = point;
    event.preventDefault();
  };

  const onImageMouseUp = (event: MouseEvent<HTMLImageElement>) => {
    if (!dragStart.current) return;
    const end = pointFromMouse(event);
    const start = dragStart.current;
    dragStart.current = null;
    if (!end) return;
    const left = Math.min(start[0], end[0]);
    const top = Math.min(start[1], end[1]);
    const width = Math.abs(end[0] - start[0]);
    const height = Math.abs(end[1] - start[1]);
    const rectangle = { left, top, width, height };
    if (selectionMode === "chart") {
      if (width < 320 || height < 160) {
        setError("El área del gráfico debe medir al menos 320×160 píxeles.");
        return;
      }
      setSelection(rectangle);
      setError(null);
      return;
    }
    if (["asset", "timer", "payout", "amount", "outcome"].includes(selectionMode)) {
      if (width < 12 || height < 10 || width > 1600 || height > 500) {
        setError("La región OCR debe medir entre 12×10 y 1600×500 píxeles.");
        return;
      }
      setOcrRegions((current) => ({
        ...current,
        [selectionMode as VisionOCRField]: rectangle,
      }));
      setError(null);
      return;
    }
    if (width > 8 || height > 8) {
      setError("Para CALL/PUT, haz clic una vez en el centro del botón.");
      return;
    }
    if (selectionMode === "call") setCallPoint(start);
    if (selectionMode === "put") setPutPoint(start);
    setError(null);
  };

  const saveCalibration = async () => {
    if (!selection || !asset.trim() || !["asset", "timer", "payout", "amount"].every((field) => ocrRegions[field as VisionOCRField])) {
      setError("Selecciona el gráfico, completa las cuatro regiones OCR obligatorias e indica el activo esperado.");
      return;
    }
    setBusy(true);
    try {
      await calibrateVisionRegion({
        ...selection,
        asset: asset.trim(),
        timeframe_seconds: Number(timeframe),
        ocr_regions: ocrRegions as Record<Exclude<VisionOCRField, "outcome">, ScreenRect> &
          Partial<Record<"outcome", ScreenRect>>,
        click_call: callPoint,
        click_put: putPoint,
      });
      setPreview(null);
      await refreshButtonCalibration();
      await refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo guardar la calibración.");
    } finally {
      setBusy(false);
    }
  };

  const saveButtonCalibration = async () => {
    if (!callPoint || !putPoint) {
      setError("Marca primero el centro de los botones CALL y PUT.");
      return;
    }
    setBusy(true);
    try {
      setButtonCalibration(await calibrateVisionButtons({
        click_call: callPoint,
        click_put: putPoint,
      }));
      setPreview(null);
      setError("Coordenadas CALL/PUT guardadas con el monitor y DPI actuales. El clic real sigue bloqueado.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo guardar CALL/PUT.");
    } finally {
      setBusy(false);
    }
  };

  const probeButton = async (side: VisionButtonSide) => {
    try {
      const result = await probeVisionButton(side);
      setError(`${side}: cursor movido a ${result.x}, ${result.y}; no se hizo clic.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : `No se pudo probar la posición ${side}.`);
    }
  };

  const showChart = async () => {
    try {
      const value = await getChartPreview();
      setChartImage(value.image);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo cargar el recorte.");
    }
  };

  const saveReplay = async () => {
    try {
      const capture = await saveVisionReplayCapture();
      await refreshReplayCaptures();
      setSelectedReplayId(capture.capture_id);
      setReplayResult(null);
      setError(`Captura ${capture.capture_id} guardada localmente para replay.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo guardar la captura para replay.");
    }
  };

  const replaySelectedCapture = async () => {
    if (!selectedReplayId) return;
    try {
      setReplayResult(await runVisionReplayCapture(selectedReplayId));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo reproducir la captura.");
    }
  };

  const saveDecision = async () => {
    try {
      const result = await recordVisionDecision();
      setError(
        result.duplicate
          ? `Esta señal de ${result.candidate_direction} ya está registrada para esta vela.`
          : result.series_labeling?.watching
            ? `Candidato ${result.candidate_direction} registrado. Se medirá visualmente la vela siguiente desde ${result.series_labeling.target_candle_open_at}; no es el resultado de una operación.`
            : `Candidato ${result.candidate_direction} registrado. ${result.series_labeling?.reason ?? "Sin etiqueta automática para esta vela."}`,
      );
      void refreshSeriesDataset();
      await refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo guardar la decisión.");
    }
  };

  const saveThresholds = async () => {
    try {
      const result = await setVisionThresholds(edgeMarginPct, minimumMarginPct);
      setEdgeMarginPct(result.edge_margin_pct);
      setMinimumMarginPct(result.minimum_margin_pct);
      await refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudieron guardar los umbrales.");
    }
  };

  const saveOutcome = async (decisionId: string, outcome: "win" | "loss") => {
    try {
      await recordVisionOutcome(decisionId, outcome, {
        ...(outcomePayout ? { payout_pct: Number(outcomePayout) } : {}),
        ...(outcomeDuration ? { duration_seconds: Number(outcomeDuration) } : {}),
      });
      await refresh();
      await refreshLearning();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo registrar el resultado.");
    }
  };

  const followDemoOutcome = async (decisionId: string) => {
    try {
      const seconds = Number(outcomeDuration);
      if (!Number.isInteger(seconds) || seconds < 1 || seconds > 86400) {
        throw new Error("Indica la duración esperada de la operación DEMO en segundos.");
      }
      await watchVisionOutcome(decisionId, seconds);
      await refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo iniciar el seguimiento OCR.");
    }
  };

  const correctOutcome = async (decisionId: string, outcome: "win" | "loss") => {
    try {
      await correctVisionOutcome(decisionId, outcome);
      await refresh();
      await refreshLearning();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo corregir el resultado.");
    }
  };

  const cancelOutcomeWatch = async (decisionId: string) => {
    try {
      await cancelVisionOutcomeWatch(decisionId);
      await refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo cancelar el seguimiento.");
    }
  };

  const toggleClickThrough = async () => {
    const next = !clickThrough;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("set_click_through", { enabled: next });
      setClickThrough(next);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Click-through solo está disponible en la app de escritorio.");
    }
  };

  const triggerKillSwitch = async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("set_click_through", { enabled: false });
      setClickThrough(false);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo desactivar click-through.");
    }
    try {
      await setVisionCapturePaused(true);
      await refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "No se pudo pausar la captura.");
    }
  };

  const decision = vision?.calibration_status?.decision ?? "NO_TRADE";
  const calibrated = vision?.calibration_status?.calibrated ?? false;
  const buttonsCalibrated = Boolean(buttonCalibration?.buttons?.CALL && buttonCalibration?.buttons?.PUT);
  const watchSecondsRemaining = outcomeWatch
    ? Math.max(0, Math.ceil((new Date(outcomeWatch.expected_at).getTime() - currentTime) / 1000))
    : null;
  const latestPixelCandle: VisionPixelCandle | undefined =
    vision?.ohlc_candles?.[vision.ohlc_candles.length - 1];

  return (
    <main
      className="h-screen w-screen overflow-hidden bg-transparent text-white font-mono p-3"
      style={{ opacity: opacity / 100 }}
    >
      <section
        ref={panelRef}
        className="max-h-[calc(100vh-24px)] w-full max-w-xl overflow-y-auto rounded-md border border-mv-border bg-mv-panel/95 shadow-xl"
      >
        <header data-tauri-drag-region className="flex cursor-move items-center justify-between border-b border-mv-border px-3 py-2">
          <strong className="text-xs tracking-widest text-mv-neutral">MARKETVISION · VISIÓN LOCAL</strong>
          <span className="text-right text-[10px] text-mv-neutral">
            {clickThrough ? "CLICK-THROUGH ACTIVO" : "PANEL INTERACTIVO"}
            {" · "}captura {vision?.capture_fps_target ?? 5} FPS
            {" · "}{vision?.readable ? "GRÁFICO LEGIBLE" : "SIN LECTURA DEL GRÁFICO"}
          </span>
        </header>

        <div className="flex flex-wrap items-center gap-2 border-b border-mv-border px-3 py-2 text-[10px]">
          <button className={buttonClass} onClick={() => void takeDesktopPreview()} disabled={busy}>
            {busy ? "CAPTURANDO..." : "CALIBRAR ÁREA"}
          </button>
          <button className={buttonClass} onClick={() => void detectDesktopCharts()} disabled={busy}>
            DETECTAR GRÁFICO
          </button>
          <button className={buttonClass} onClick={() => void startButtonCalibration()} disabled={busy}>
            CALIBRAR CALL/PUT
          </button>
          <button className={buttonClass} onClick={() => void probeButton("CALL")} disabled={!buttonsCalibrated}>
            PROBAR CURSOR CALL
          </button>
          <button className={buttonClass} onClick={() => void probeButton("PUT")} disabled={!buttonsCalibrated}>
            PROBAR CURSOR PUT
          </button>
          <button className={buttonClass} onClick={toggleClickThrough}>
            {clickThrough ? "DESACTIVAR CLICK-THROUGH" : "ACTIVAR CLICK-THROUGH"}
          </button>
          <button
            className={buttonClass}
            onClick={() => {
              void setVisionCapturePaused(vision?.status !== "capture_paused")
                .then(refresh)
                .catch((requestError: unknown) => {
                  setError(requestError instanceof Error ? requestError.message : "No se pudo cambiar el estado.");
                });
            }}
          >
            {vision?.status === "capture_paused" ? "REANUDAR (F8)" : "PAUSAR (F8)"}
          </button>
          <button className={buttonClass} onClick={() => void triggerKillSwitch()}>
            PARADA DE EMERGENCIA
          </button>
          <span className="text-[10px] text-mv-neutral">
            F9 mostrar/ocultar · F10 click-through · F12 parada · F8 pausa · panel{" "}
            {panelRegionReady ? "listo" : "midiendo"}
          </span>
          <label className="ml-auto flex items-center gap-2 text-mv-neutral">
            OPACIDAD
            <input
              aria-label="Opacidad del overlay"
              type="range"
              min="35"
              max="100"
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
            />
            {opacity}%
          </label>
        </div>

        <section className="space-y-2 border-b border-mv-border px-3 py-2 text-[10px]">
          <div className="tracking-widest text-mv-neutral">REPLAY VISUAL · SIN MERCADO NI CLICS</div>
          <div className="flex flex-wrap items-center gap-2">
            <button className={buttonClass} onClick={() => void saveReplay()} disabled={!vision?.captured_at}>
              GUARDAR CAPTURA LOCAL
            </button>
            <select
              aria-label="Captura para replay"
              value={selectedReplayId}
              onChange={(event) => setSelectedReplayId(event.target.value)}
              className="max-w-64 rounded border border-mv-border bg-mv-bg px-2 py-1 text-[10px]"
            >
              {replayCaptures.length === 0 && <option value="">Sin capturas guardadas</option>}
              {replayCaptures.map((capture) => (
                <option key={capture.capture_id} value={capture.capture_id}>
                  {new Date(capture.captured_at).toLocaleString()} · {capture.asset ?? "activo desconocido"}
                </option>
              ))}
            </select>
            <button
              className={buttonClass}
              onClick={() => void replaySelectedCapture()}
              disabled={!selectedReplayId}
            >
              REPRODUCIR · PAPER VISUAL
            </button>
          </div>
          {replayResult && (
            <div className="rounded border border-mv-border p-2 text-mv-neutral">
              {replayResult.asset ?? "Activo no guardado"} · {replayResult.candle_count} velas ·
              {" "}calidad visual {replayResult.visual_quality_score}/100 · CALL{" "}
              {replayResult.directional_scores.CALL === null
                ? "—"
                : `${Math.round(replayResult.directional_scores.CALL * 100)}%`}
              {" · "}PUT{" "}
              {replayResult.directional_scores.PUT === null
                ? "—"
                : `${Math.round(replayResult.directional_scores.PUT * 100)}%`}
              {" · "}decisión {replayResult.decision}
              <div>{replayResult.reason}</div>
            </div>
          )}
        </section>

        {preview && (
          <div className="space-y-2 border-b border-mv-border p-3">
            <div className="text-[10px] text-mv-neutral">
              El escritorio virtual incluye todos los monitores. Arrastra para cada región OCR y el gráfico.
              {" "}Para CALL/PUT, selecciona el botón en el menú y marca su centro con un clic.
              {" "}La detección del gráfico es una sugerencia visual; comprueba el recuadro antes de guardarlo.
              Escritorio {preview.width}×{preview.height} · DPI {preview.dpi}
              {" · "}DPI por monitor {preview.dpi_aware ? "activo" : `no disponible: ${preview.dpi_error ?? "revisa Windows"}`}
            </div>
            <div className="relative">
              <img
                ref={imageRef}
                src={`data:image/jpeg;base64,${preview.image}`}
                alt="Captura local del escritorio para calibración"
                draggable={false}
                onMouseDown={onImageMouseDown}
                onMouseUp={onImageMouseUp}
                className="block h-auto w-full cursor-crosshair border border-mv-border"
              />
              {selection && (
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute border-2 border-yellow-300 bg-yellow-300/10"
                  style={{
                    left: `${((selection.left - preview.left) / preview.width) * 100}%`,
                    top: `${((selection.top - preview.top) / preview.height) * 100}%`,
                    width: `${(selection.width / preview.width) * 100}%`,
                    height: `${(selection.height / preview.height) * 100}%`,
                  }}
                />
              )}
            </div>
            {chartCandidates.length > 0 && (
              <div className="flex flex-wrap gap-1 text-[10px] text-mv-neutral">
                Sugerencias:
                {chartCandidates.map((candidate, index) => (
                  <button
                    key={`${candidate.left}-${candidate.top}-${index}`}
                    className={buttonClass}
                    onClick={() => setSelection({
                      left: candidate.left,
                      top: candidate.top,
                      width: candidate.width,
                      height: candidate.height,
                    })}
                  >
                    #{index + 1} · {candidate.candle_count} velas · calidad {candidate.visual_quality_score}
                  </button>
                ))}
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              <select
                aria-label="Modo de selección"
                value={selectionMode}
                onChange={(event) => setSelectionMode(event.target.value as SelectionMode)}
                className="rounded border border-mv-border bg-mv-bg px-2 py-1 text-[10px]"
              >
                <option value="chart">Seleccionar gráfico</option>
                <option value="asset">Región OCR: activo</option>
                <option value="timer">Región OCR: temporizador</option>
                <option value="payout">Región OCR: payout</option>
                <option value="amount">Región OCR: monto</option>
                <option value="outcome">Región OCR: resultado final (opcional)</option>
                <option value="call">Marcar botón CALL</option>
                <option value="put">Marcar botón PUT</option>
              </select>
              <input
                aria-label="Activo esperado para validar OCR"
                placeholder="Activo esperado"
                value={asset}
                onChange={(event) => setAsset(event.target.value)}
                className="w-32 rounded border border-mv-border bg-mv-bg px-2 py-1 text-[10px]"
              />
              <input
                aria-label="Duración de vela en segundos"
                type="number"
                min="1"
                max="86400"
                value={timeframe}
                onChange={(event) => setTimeframe(event.target.value)}
                className="w-20 rounded border border-mv-border bg-mv-bg px-2 py-1 text-[10px]"
              />
              <button className={buttonClass} onClick={() => void saveCalibration()} disabled={busy || !selection}>
                GUARDAR ÁREA
              </button>
              <button
                className={buttonClass}
                onClick={() => void saveButtonCalibration()}
                disabled={busy || !callPoint || !putPoint}
              >
                GUARDAR BOTONES CALL/PUT
              </button>
            </div>
            <div className="text-[10px] text-mv-neutral">
              Área: {selection ? `${selection.left},${selection.top} · ${selection.width}×${selection.height}` : "sin seleccionar"}
              {(["asset", "timer", "payout", "amount", "outcome"] as const).map((field) => (
                <span key={field}>{" · "}{ocrFieldLabels[field]}: {ocrRegions[field] ? `${ocrRegions[field]?.left},${ocrRegions[field]?.top} · ${ocrRegions[field]?.width}×${ocrRegions[field]?.height}` : "pendiente"}</span>
              ))}
              {" · "}CALL: {callPoint?.join(",") ?? "sin marcar"} · PUT: {putPoint?.join(",") ?? "sin marcar"}
              {" · "}CALL {callPoint ? `${callPoint[0]},${callPoint[1]}` : "sin marcar"}
              {" · "}PUT {putPoint ? `${putPoint[0]},${putPoint[1]}` : "sin marcar"}
            </div>
          </div>
        )}

        <div className="border-b border-mv-border px-3 py-2 text-[10px] text-mv-neutral">
          Botones guardados: CALL {buttonCalibration?.buttons?.CALL
            ? `${buttonCalibration.buttons.CALL.x},${buttonCalibration.buttons.CALL.y}`
            : "sin calibrar"}
          {" · "}PUT {buttonCalibration?.buttons?.PUT
            ? `${buttonCalibration.buttons.PUT.x},${buttonCalibration.buttons.PUT.y}`
            : "sin calibrar"}
          {buttonCalibration?.screen_width && buttonCalibration?.screen_height &&
            ` · ${buttonCalibration.screen_width}×${buttonCalibration.screen_height}`}
          {buttonCalibration?.dpi ? ` · DPI ${buttonCalibration.dpi}` : ""}
          {" · "}prueba mueve el cursor, nunca hace clic.
        </div>

        <section className="grid grid-cols-2 gap-3 p-3">
          <div>
            <div className="text-[10px] tracking-widest text-mv-neutral">LECTURA VISUAL</div>
            <div className={`mt-2 text-xl font-bold ${decision === "NO_TRADE" ? "text-mv-neutral" : decision === "CALL" ? "text-mv-up" : "text-mv-down"}`}>
              {decision === "CALL" ? "CALL" : decision === "PUT" ? "PUT" : "NO OPERAR"}
            </div>
            <div className="mt-2 text-[10px] text-mv-neutral">
              {vision?.reason ?? "Iniciando captura local."}
            </div>
            <div className="mt-2 text-[10px] text-mv-neutral">
              Activo: {vision?.asset ?? "no calibrado"} · velas: {vision?.candle_count ?? 0}
            </div>
            <div className="mt-1 text-[10px] text-mv-neutral">
              Tendencia: {vision?.trend ?? "—"} · momentum: {vision?.momentum ?? "—"}
            </div>
            <div className="mt-1 text-[10px] text-mv-neutral">
              Calidad lectura: {vision?.read_quality_score ?? 0}/100
              {" · "}payout {vision?.ocr?.fields.payout.percent ?? "—"}%
              {" · "}monto {vision?.ocr?.fields.amount.value ?? "—"}
              {" · "}timer {vision?.ocr?.fields.timer.seconds_remaining ?? "—"} s
            </div>
            <div className="mt-1 text-[10px] text-mv-neutral">
              Activo OCR: {vision?.ocr?.fields.asset.value ?? "—"}
              {" · "}monitores {vision?.monitor_layout?.length ?? 0}
              {" · "}resultado DEMO OCR{" "}
              {vision?.ocr?.fields.outcome.result?.toUpperCase() ?? "sin resultado final"}
              {" · "}cambio pantalla {vision?.screen_change?.status ?? "—"}
            </div>
            <div className="mt-1 text-[10px] text-mv-neutral">
              OCR temporal: activo {vision?.ocr?.fields.asset.stable ? "estable" : "confirmando"}
              {" · "}timer {vision?.ocr?.fields.timer.stable ? "estable" : "confirmando"}
              {" · "}payout {vision?.ocr?.fields.payout.stable ? "estable" : "confirmando"}
              {" · "}monto {vision?.ocr?.fields.amount.stable ? "estable" : "confirmando"}
            </div>
            <div className="mt-1 text-[10px] text-mv-neutral">
              Serie OHLC local: {vision?.ohlc_stored_count ?? 0} velas · escala normalizada de píxeles, no precios de mercado
              {latestPixelCandle && (
                <> · última O {latestPixelCandle.open.toFixed(3)} H {latestPixelCandle.high.toFixed(3)}
                  {" "}L {latestPixelCandle.low.toFixed(3)} C {latestPixelCandle.close.toFixed(3)}</>
              )}
            </div>
            <div className="mt-1 text-[10px] text-mv-neutral">
              Puntuación visual (no probabilidad): CALL {vision?.directional_scores.CALL === null || vision?.directional_scores.CALL === undefined ? "—" : `${Math.round(vision.directional_scores.CALL * 100)}%`}
              {" · "}PUT {vision?.directional_scores.PUT === null || vision?.directional_scores.PUT === undefined ? "—" : `${Math.round(vision.directional_scores.PUT * 100)}%`}
            </div>
            <div className="mt-2 text-[10px] text-mv-down">
              {vision?.calibration_error ?? vision?.calibration_status?.reason ?? "Esperando historial etiquetado."}
            </div>
          </div>
          <div className="space-y-2 text-[10px] text-mv-neutral">
            <div className="flex flex-wrap items-center gap-1">
              <span>VALIDACIÓN TEMPORAL: {calibrated ? "SUPERADA" : "BLOQUEADA"}</span>
              <label>
                Margen sobre breakeven (pp)
                <input aria-label="Margen porcentual sobre el breakeven" type="number" min="0" max="25"
                  value={edgeMarginPct} onChange={(event) => setEdgeMarginPct(Number(event.target.value))}
                  className="ml-1 w-12 rounded border border-mv-border bg-mv-bg px-1 py-1" />
              </label>
              <label>
                Separación mínima (pp)
                <input aria-label="Margen mínimo direccional" type="number" min="0" max="49"
                  value={minimumMarginPct} onChange={(event) => setMinimumMarginPct(Number(event.target.value))}
                  className="ml-1 w-12 rounded border border-mv-border bg-mv-bg px-1 py-1" />
              </label>
              <button className={buttonClass} onClick={() => void saveThresholds()}>GUARDAR</button>
              <span>
                Margen adaptativo actual: {vision?.calibration_status?.adaptive_edge_margin_pct ?? "—"} pp
              </span>
            </div>
            {(["CALL", "PUT"] as const).map((side) => (
              <div key={side}>
                {side}: probabilidad histórica del contexto{" "}
                {vision?.calibration_status?.contextual_estimates[side]?.probability_pct === null ||
                vision?.calibration_status?.contextual_estimates[side]?.probability_pct === undefined
                  ? "—"
                  : `${vision.calibration_status.contextual_estimates[side].probability_pct}%`}
                {" · "}IC95%{" "}
                {vision?.calibration_status?.contextual_estimates[side]?.lower_95_pct ?? "—"}
                {"–"}
                {vision?.calibration_status?.contextual_estimates[side]?.upper_95_pct ?? "—"}
                {" · "}n={vision?.calibration_status?.contextual_estimates[side]?.samples ?? 0}/
                {vision?.calibration_status?.minimum_context_samples_per_direction ?? 30}
                {" · "}winrate global{" "}
                {vision?.calibration_status?.directional_win_rates[side] ?? "—"}%
              </div>
            ))}
            <div>
              Breakeven según payout: {vision?.calibration_status?.break_even_pct ?? "—"}%
              {" · "}umbral con margen: {vision?.calibration_status?.required_probability_pct ?? "—"}%
              {" · "}separación observada: {vision?.calibration_status?.directional_margin_pct ?? "—"} pp
              {" · "}separación conservadora IC95%: {vision?.calibration_status?.conservative_margin_pct ?? "—"} pp
            </div>
            <div>
              Dirección visual: {vision?.calibration_status?.visual_direction ?? "indecisa"}
              {" · "}separación visual: {vision?.calibration_status?.visual_margin_pct ?? "—"} pp
            </div>
            <div>
              Contexto: {vision?.asset ?? "—"} / {vision?.timeframe_seconds ?? "—"} s /
              {" "}{vision?.market_regime ?? "unknown"} / hora UTC{" "}
              {vision?.calibration_status?.context.hour_utc ?? "—"}.
              {" "}La decisión requiere concordancia entre la dirección visual y la probabilidad histórica contextual.
            </div>
            <div>
              Holdout global CALL: {vision?.calibration_status?.validation.CALL?.validation_samples ?? 0} muestras;
              {" "}PUT: {vision?.calibration_status?.validation.PUT?.validation_samples ?? 0}.
            </div>
            <div className="border-t border-mv-border pt-2">
              Captura y tendencia basadas en colores/forma de píxeles; no son precios ni probabilidades.
            </div>
            <button
              className={buttonClass}
              onClick={() => void showChart()}
              disabled={!vision?.readable}
            >
              VISTA DEL GRÁFICO
            </button>
          </div>
        </section>

        {chartImage && (
          <img
            src={`data:image/jpeg;base64,${chartImage}`}
            alt="Recorte capturado del gráfico"
            className="max-h-56 w-full border-y border-mv-border object-contain"
          />
        )}

        <section className="border-t border-mv-border p-3">
          <div className="mb-2 text-[10px] text-mv-neutral">
            RIESGO PAPER GENERAL (no autoriza clic visual): {systemStatus?.risk.trading_locked ? `BLOQUEADO · ${systemStatus.risk.lock_reason ?? "límite activo"}` : "sin bloqueo"}
            {" · "}operaciones hoy {systemStatus?.risk.trades_today ?? 0}
            / {systemStatus?.risk_limits.max_trades_per_day ?? "—"}
            {" · "}última hora {systemStatus?.risk ? "controlada por RiskManager" : "—"}
            {" · "}P&L {systemStatus?.risk.daily_pnl_pct ?? 0}%
            {" · "}objetivo +{systemStatus?.risk_limits.max_daily_profit_pct ?? "—"}%
            {" · "}límite -{systemStatus?.risk_limits.max_daily_loss_pct ?? "—"}%
            {" · "}racha máx. {systemStatus?.risk_limits.max_consecutive_losses ?? "—"}
            {" · "}operaciones/h {systemStatus?.risk_limits.max_trades_per_hour ?? "—"}
          </div>
          <div className="mb-2 flex flex-wrap gap-2">
            <button
              className={buttonClass}
              disabled={busy || systemStatus?.mode === "PAPER_TRADING_MODE" || systemStatus?.risk.trading_locked}
              onClick={async () => {
                if (!window.confirm("¿Iniciar AUTO PAPER? Solo crea operaciones simuladas locales; no hace clic ni envía órdenes a Quotex.")) return;
                setBusy(true);
                try {
                  await setPaperMode(true);
                  await refresh();
                } catch (requestError) {
                  setError(requestError instanceof Error ? requestError.message : "No se pudo activar paper trading.");
                } finally {
                  setBusy(false);
                }
              }}
            >
              INICIAR AUTO PAPER
            </button>
            <button
              className={buttonClass}
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await activateKillSwitch();
                  await refresh();
                } catch (requestError) {
                  setError(requestError instanceof Error ? requestError.message : "No se pudo activar el kill switch.");
                } finally {
                  setBusy(false);
                }
              }}
            >
              KILL SWITCH
            </button>
            {systemStatus?.risk.trading_locked && (
              <button
                className={buttonClass}
                disabled={busy}
                onClick={async () => {
                  if (!window.confirm("¿Desbloquear RiskManager? Confirma que has revisado los límites y el motivo del bloqueo.")) return;
                  setBusy(true);
                  try {
                    await unlockPaperRisk();
                    await refresh();
                  } catch (requestError) {
                    setError(requestError instanceof Error ? requestError.message : "No se pudo desbloquear el riesgo.");
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                DESBLOQUEAR (CONFIRMACIÓN)
              </button>
            )}
            {" · "}sin posiciones visuales abiertas
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] tracking-widest text-mv-neutral">PAPER / DEMO · RESULTADOS REALES ETIQUETADOS</span>
            <button className={buttonClass} onClick={() => void saveDecision()} disabled={!vision?.readable}>
              REGISTRAR CANDIDATO DEMO
            </button>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-mv-neutral">
            <label>
              Payout observado %
              <input
                aria-label="Payout observado en porcentaje"
                type="number"
                min="0"
                max="100"
                value={outcomePayout}
                onChange={(event) => setOutcomePayout(event.target.value)}
                className="ml-1 w-16 rounded border border-mv-border bg-mv-bg px-1 py-1"
              />
            </label>
            <label>
              Duración operación DEMO s
              <input
                aria-label="Duración observada en segundos"
                type="number"
                min="1"
                max="86400"
                value={outcomeDuration}
                onChange={(event) => setOutcomeDuration(event.target.value)}
                className="ml-1 w-16 rounded border border-mv-border bg-mv-bg px-1 py-1"
              />
            </label>
            <span>Para vigilar OCR, calibra también la región opcional del resultado.</span>
          </div>
          {outcomeWatch && (
            <div className="mt-2 flex items-center justify-between gap-2 text-[10px] text-mv-neutral">
              Resultado DEMO bajo seguimiento · cierra después de{" "}
              {new Date(outcomeWatch.expected_at).toLocaleTimeString()} · quedan {watchSecondsRemaining} s · solo OCR final estable etiqueta.
              <button className={buttonClass} onClick={() => void cancelOutcomeWatch(outcomeWatch.decision_id)}>
                CANCELAR
              </button>
            </div>
          )}
          <div className="mt-2 max-h-36 space-y-1 overflow-auto">
            {history.length === 0 ? (
              <div className="text-[10px] text-mv-neutral">Sin decisiones registradas.</div>
            ) : history.slice(0, 12).map((item) => (
              <div key={item.decision_id} className="flex items-center justify-between gap-2 text-[10px]">
                <span>
                  {new Date(item.created_at).toLocaleTimeString()} · {item.asset} · {item.direction}
                  {item.timeframe_seconds ? ` · ${item.timeframe_seconds}s` : ""}
                  {item.confidence_pct !== null ? ` · ${item.confidence_pct}% empírico` : ""}
                </span>
                {item.outcome ? (
                  <span className="flex items-center gap-1">
                    {item.outcome.toUpperCase()}
                    {" · "}{item.outcome_source === "screen_ocr" ? "OCR" : item.outcome_source === "manual_correction" ? "corregido" : "manual"}
                    {item.payout_pct !== null ? ` · ${item.payout_pct}% payout` : ""}
                    {item.duration_seconds !== null ? ` · ${item.duration_seconds}s` : ""}
                    <button
                      className={buttonClass}
                      onClick={() => void correctOutcome(item.decision_id, item.outcome === "win" ? "loss" : "win")}
                    >
                      CORREGIR
                    </button>
                  </span>
                ) : (
                  <span className="flex gap-1">
                    <button className={buttonClass} onClick={() => void saveOutcome(item.decision_id, "win")}>WIN</button>
                    <button className={buttonClass} onClick={() => void saveOutcome(item.decision_id, "loss")}>LOSS</button>
                    <button
                      className={buttonClass}
                      disabled={!vision?.ocr?.fields.outcome.configured || !vision?.ocr?.fields.outcome.available || Boolean(outcomeWatch)}
                      onClick={() => void followDemoOutcome(item.decision_id)}
                    >
                      VIGILAR OCR DEMO
                    </button>
                  </span>
                )}
              </div>
                ))}
              </div>
              <div className="mt-3 border-t border-mv-border pt-2 text-[10px] text-mv-neutral">
                <div className="mb-1 tracking-widest">APRENDIZAJE VISUAL</div>
                <div>
                  Etiquetas: {learning?.labeled_decisions ?? 0} · muestras dataset: {learning?.dataset_samples ?? 0}
                  {" · "}umbral de confianza incremental: {vision?.calibration_status?.adaptive_confidence_threshold_pct ?? "—"}%
                  {" · "}actualizaciones: {learning?.online_learning.updates ?? 0}
                  {" · "}modelo activo: {learning?.active_model_version ?? "baseline"}
                </div>
                <div className="mt-1">
                  Próxima vela: {seriesDataset?.total_labeled ?? 0} etiquetas ·
                  {" "}pendientes {seriesDataset?.pending ?? 0} · vencidas {seriesDataset?.expired ?? 0} ·
                  {" "}CALL {seriesDataset?.by_direction.CALL ?? 0} ·
                  {" "}PUT {seriesDataset?.by_direction.PUT ?? 0} ·
                  {" "}FLAT {seriesDataset?.by_direction.FLAT ?? 0}
                  {" · "}dataset visual separado del resultado de operaciones
                </div>
                <div className="mt-1">
                  Modelo visual OOS: {seriesModel?.active_version ?? "sin campeón"}
                  {" · "}muestras {seriesModel?.total_samples ?? 0}/{seriesModel?.minimum_samples ?? 200}
                  {seriesModel?.last_result?.brier_score !== undefined && (
                    <> · Brier test {seriesModel.last_result.brier_score.toFixed(3)}
                      {" / baseline "}{seriesModel.last_result.baseline_brier_score?.toFixed(3) ?? "—"}
                      {" · log-loss "}{seriesModel.last_result.log_loss?.toFixed(3) ?? "—"}
                      {" · calibración "}{seriesModel.last_result.calibration_error?.toFixed(3) ?? "—"}</>
                  )}
                </div>
                {seriesModel?.prediction && (
                  <div className="mt-1">
                    Próxima vela visual: {Math.round(seriesModel.prediction.probability_correct * 100)}%
                    {" · IC95% "}
                    {seriesModel.prediction.interval_95
                      ? `${Math.round(seriesModel.prediction.interval_95.lower * 100)}–${Math.round(seriesModel.prediction.interval_95.upper * 100)}% (n=${seriesModel.prediction.interval_95.samples})`
                      : "sin intervalo con soporte suficiente"}
                    {" · no es probabilidad de ganancia"}
                  </div>
                )}
                <button
                  className={buttonClass}
                  onClick={() => void evaluateSeriesModel()}
                  disabled={busy || (seriesModel?.total_samples ?? 0) < (seriesModel?.minimum_samples ?? 200)}
                >
                  EVALUAR MODELO VISUAL OOS
                </button>
                <div className="mt-1">{seriesModel?.note}</div>
                {(modelNotice || seriesModel?.last_attempt_reason) && (
                  <div className="mt-1 text-mv-neutral">
                    {modelNotice ?? seriesModel?.last_attempt_reason}
                  </div>
                )}
                <div className="mt-1">
                  Winrate por activo: {Object.entries(learning?.by_asset ?? {}).slice(0, 4)
                    .map(([name, stats]) => `${name} ${Math.round(stats.win_rate * 100)}% (n=${stats.trades})`).join(" · ") || "sin muestras"}
                </div>
                <div className="mt-1">
                  Winrate por contexto: {Object.entries(learning?.by_context ?? {}).slice(0, 3)
                    .map(([name, stats]) => `${name} ${Math.round(stats.win_rate * 100)}% (n=${stats.trades})`).join(" · ") || "sin muestras"}
                </div>
                <div className="mt-1">
                  Features aprendidas: {learning?.online_learning.feature_importance.slice(0, 4)
                    .map((item) => `${item.feature} ${item.weight.toFixed(3)}`).join(" · ") || "pendiente de etiquetas"}
                </div>
                <div className="mt-1">
                  Coeficientes modelo: {learning?.model_feature_importance.slice(0, 4)
                    .map((item) => `${item.feature} ${item.weight.toFixed(3)}`).join(" · ") || "modelo baseline"}
                </div>
                <div className="mt-1">
                  Entrenamiento batch: {learning?.batch_training.note ?? "inicializando"}
                </div>
                {learning?.model_evolution.slice(0, 3).map((model) => (
                  <div key={`${model.model_version}-${model.evaluated_at}`}>
                    {model.decision.toUpperCase()} · {model.model_version} · Brier {model.brier_score?.toFixed(3) ?? "—"}
                    {" / baseline "}{model.baseline_brier_score?.toFixed(3) ?? "—"} · n={model.samples}
                  </div>
                ))}
                <div className="mt-1">
                  Calibración OOS: {learning?.calibration.slice(0, 4)
                    .map((item) => `${item.probability_bucket} → real ${Math.round(item.observed_win_rate * 100)}% (n=${item.samples})`).join(" · ") || "sin holdout promovido"}
                </div>
          </div>
          <div className="mt-2 border-t border-mv-border pt-2 text-[10px] text-mv-down">
            CLIC REAL BLOQUEADO: requiere historial validado y detector fiable de operación abierta.
            {vision?.live_execution.reason && ` ${vision.live_execution.reason}`}
          </div>
        </section>

        {(apiError ?? error) && (
          <div role="alert" className="border-t border-mv-border px-3 py-2 text-[10px] text-mv-down">
            {apiError ?? error}
          </div>
        )}
      </section>
    </main>
  );
}
