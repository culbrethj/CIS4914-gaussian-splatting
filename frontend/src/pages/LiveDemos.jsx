import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import GaussViewer from "../components/GaussViewer";
import InfoTip from "../components/InfoTip";
import StageSummary from "../components/StageSummary";
import { parsePipelineLog, formatEta } from "../utils/pipelineLog";
import { fetchJson, parseApiError } from "../utils/apiClient";
import "./LiveDemos.css";

// Plain-English explanations for every Advanced Settings knob. Kept in
// one dictionary so the tooltips + Settings docs never drift.
const SETTING_INFO = {
  sfmMethod: "COLMAP is the traditional method (~8-12 min of CPU SfM). VGGT uses a neural network for near-instant camera estimation (<1 min on a GPU). VGGT is a feed-forward transformer from CVPR 2025 that replaces the full feature-extraction + matching + mapping pipeline with one pass over the images.",
  iters: "How many training passes. More = better quality but slower. 10,000 is good for testing; 30,000 for final quality. Doesn't invalidate cached SfM.",
  blur: "Drops frames whose Laplacian variance is below this number (i.e. blurry frames). Higher = stricter. Affects the SfM step, so changing this triggers reprocessing.",
  dup: "Drops frames that look nearly identical to the last kept frame. Higher = more aggressive dedup. Affects the SfM step.",
  fps: "How many frames per second to pull from the source video. Lower = fewer total frames = faster SfM, but less scene coverage. 0 uses the source video's native FPS.",
  downscale: "Resize each extracted frame by this factor before preprocessing. 1.0 = original resolution (default); 0.5 = half the source resolution. Lower speeds up SfM but loses detail.",
  maxWidth: "Hard cap on output frame width in pixels. Any frame wider than this is downscaled further. Useful for very high-res source video.",
  seed: "Random seed for training. Use the same seed to reproduce a previous run. Doesn't affect SfM.",
  scaleReset: "Shrinks every gaussian's size by a factor every K iterations. From the Shorter-Splatting paper — aims to reduce per-pixel gaussian overlap and speed up training.",
  scaleResetEvery: "How often (in iterations) the scale-reset step fires. 1000 is a reasonable starting point.",
  scaleResetFactor: "Multiplier applied to each gaussian's scale at every reset (<1 shrinks). 0.9 is mild; 0.5 is aggressive.",
  entropy: "Adds an entropy penalty on gaussian opacities during training so each gaussian is pushed toward fully-on or fully-off. Polarized opacities mean fewer gaussians contribute to each pixel.",
  entropyWeight: "Strength of the entropy term in the loss (the λ coefficient). 0.01 is a reasonable starting point.",
  progressive: "Trains at low resolution early and steps up to full resolution later. Cheaper early iterations, same final quality in theory.",
  progressiveSchedule: "Schedule as 'iter:scale,iter:scale,...'. Example: '0:0.25,5000:0.5,10000:1.0' starts at 25%, jumps to 50% at iter 5000, full resolution at 10000.",
};

// Must match the DATASET_RE in backend/main.py so names we accept here are
// also accepted on the server side. Keep these in sync.
const DATASET_NAME_RE = /^[A-Za-z0-9_.-]+$/;

// Default prep knobs. These mirror the defaults in pipeline.py; we set them
// here so the "advanced" panel starts populated with the same numbers the
// backend would use if the user never touches the panel.
const SIMPLE_PREP_DEFAULTS = {
  iters: 1000,
  duplicateThreshold: 1.5,
  blurThreshold: 20,
  fps: 12,
  downscale: 1.0,
  maxWidth: 1280,
};

// Shorter-Splatting experiment defaults. Every toggle defaults OFF so the
// Advanced panel runs like a plain baseline unless the user opts in.
// See backend/experiments/faster-gs/shortgs/README.md for what each does.
const SHORTGS_DEFAULTS = {
  seed: 1,
  scaleResetEnabled: false,
  scaleResetEvery: 1000,
  scaleResetFactor: 0.9,
  entropyEnabled: false,
  entropyWeight: 0.01,
  progressiveEnabled: false,
  progressiveSchedule: "0:0.25,5000:0.5,10000:1.0",
};
const BACKEND_OPTIONS = [
  {
    value: "fastergs",
    label: "Faster-GS",
    help: "Recommended. Currently the most reliable path on this HPG setup.",
  },
  {
    value: "opensplat",
    label: "OpenSplat",
    help: "Available, but depends on your HPG workspace setup.",
  },
];
const ACTIVE_JOB_STORAGE_KEY = "live_demos_active_job";

// Color-code each log line in the feed. The "<<error:" / "<<done:N" markers
// are control lines the server injects to signal status; plain "error" or
// "warning" substrings come from whichever subprocess (colmap, training,
// rsync) printed the line. Order of checks matters - the explicit markers
// win over the substring heuristics.
function logKind(line) {
  const normalized = String(line || "").toLowerCase();
  if (normalized.startsWith("<<error:")) return "error";
  if (normalized.startsWith("<<done:0")) return "success";
  if (normalized.startsWith("<<done:")) return "error";
  if (normalized.includes("error")) return "error";
  if (normalized.includes("warning")) return "warn";
  if (normalized.includes("finished") || normalized.includes("successful")) return "success";
  return "info";
}

/**
 * LiveDemos: single-page orchestration for pipeline runs.
 *
 * Organized into four sections:
 *   1. State + effects (lines 106-940): ~30 state pieces and the effects
 *      that poll jobs, reconnect the WebSocket, and persist active-job
 *      state in localStorage.
 *   2. Upload + Run panel (lines 944-1341): the left panel with New
 *      video / Existing dataset tabs, dataset form, backend picker,
 *      and Shorter-Splatting Advanced Settings.
 *   3. Progress + Logs panel (lines 1351-1422): stage chips, ETA, and
 *      the live log feed.
 *   4. Viewer panel (lines 1424-1496): renders the published splat via
 *      GaussViewer once the run finishes.
 */
export default function LiveDemos() {
  // `datasets` = every dataset on disk, regardless of whether a splat
  // exists yet. Used by the "Existing dataset" picker so we can rerun
  // SfM / training against a preprocessed dataset that was never fully
  // trained. `datasetsWithSplats` is the subset the viewer cares about.
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [uploadedDataset, setUploadedDataset] = useState("");
  // Input mode for "1) Upload + Run". `new` keeps the upload form as
  // before; `existing` hides the file picker + upload button, shows a
  // dataset dropdown, and tells the backend to skip frame extraction.
  const [inputMode, setInputMode] = useState("new");
  const [existingDataset, setExistingDataset] = useState("");
  // Tracks whether the current dataset already has a cached preprocess +
  // SfM fingerprint on the server. When present, the Advanced panel shows
  // a "SfM cached" note so users know reruns with matching settings skip
  // the 10-minute SfM step.
  const [prepStatus, setPrepStatus] = useState(null);
  const [backendChoice, setBackendChoice] = useState("fastergs");
  // SfM method is a separate axis from the training backend. Defaults to
  // colmap so existing workflows are unchanged; vggt is an opt-in neural
  // alternative that runs on a GPU in ~1 minute instead of ~10.
  const [sfmMethod, setSfmMethod] = useState("colmap");

  const [logs, setLogs] = useState([]);
  // Default view is the stage-card summary; toggle reveals the raw log
  // stream for people who want the detailed pipeline output.
  const [showRawLogs, setShowRawLogs] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("idle");
  const [uploadMessage, setUploadMessage] = useState("");
  // Read any persisted active job synchronously so the first paint
  // already looks like "reconnecting" instead of flashing idle before
  // the async restore effect completes. We don't trust localStorage
  // past that initial hint - the server's /api/jobs/{id} response is
  // what ultimately sets the real state.
  const persistedJobHint = (() => {
    try {
      const raw = typeof localStorage !== "undefined"
        ? localStorage.getItem(ACTIVE_JOB_STORAGE_KEY)
        : null;
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  })();
  const [runStatus, setRunStatus] = useState(persistedJobHint?.jobId ? "reconnecting" : "idle");
  const [runMessage, setRunMessage] = useState(
    persistedJobHint?.jobId ? "Reconnecting to running job..." : "",
  );
  const [jobId, setJobId] = useState(persistedJobHint?.jobId || "");
  const [jobStage, setJobStage] = useState(persistedJobHint?.jobId ? "reconnecting" : "");

  const [advancedActive, setAdvancedActive] = useState(false);
  const [blurThreshold, setBlurThreshold] = useState(SIMPLE_PREP_DEFAULTS.blurThreshold);
  const [duplicateThreshold, setDuplicateThreshold] = useState(SIMPLE_PREP_DEFAULTS.duplicateThreshold);
  const [fps, setFps] = useState(SIMPLE_PREP_DEFAULTS.fps);
  const [downscale, setDownscale] = useState(SIMPLE_PREP_DEFAULTS.downscale);
  const [maxWidth, setMaxWidth] = useState(SIMPLE_PREP_DEFAULTS.maxWidth);
  const [numIters, setNumIters] = useState(SIMPLE_PREP_DEFAULTS.iters);

  // Experiment / Shorter-Splatting settings. Only applied when the user
  // opens Advanced and toggles a technique on. Sent to the backend as
  // shortgs_* fields; backend forwards into the SLURM job as env vars.
  const [seed, setSeed] = useState(SHORTGS_DEFAULTS.seed);
  const [scaleResetEnabled, setScaleResetEnabled] = useState(SHORTGS_DEFAULTS.scaleResetEnabled);
  const [scaleResetEvery, setScaleResetEvery] = useState(SHORTGS_DEFAULTS.scaleResetEvery);
  const [scaleResetFactor, setScaleResetFactor] = useState(SHORTGS_DEFAULTS.scaleResetFactor);
  const [entropyEnabled, setEntropyEnabled] = useState(SHORTGS_DEFAULTS.entropyEnabled);
  const [entropyWeight, setEntropyWeight] = useState(SHORTGS_DEFAULTS.entropyWeight);
  const [progressiveEnabled, setProgressiveEnabled] = useState(SHORTGS_DEFAULTS.progressiveEnabled);
  const [progressiveSchedule, setProgressiveSchedule] = useState(SHORTGS_DEFAULTS.progressiveSchedule);

  const [datasetsLoading, setDatasetsLoading] = useState(false);
  // Runs for the viewer's selected dataset. Keyed by dataset name so we
  // don't re-fetch when the user flips between scenes of the same
  // dataset. Populated by an effect that watches selectedDataset.
  const [viewerRunsByDataset, setViewerRunsByDataset] = useState({});
  const [viewerRunTag, setViewerRunTag] = useState("");
  const wsRef = useRef(null);
  const logRef = useRef(null);

  const isUploading = uploadStatus === "uploading";
  const isRunning = runStatus === "running";
  // Treat "reconnecting" as busy so the start-new-run buttons stay
  // disabled until we know whether the saved job is still alive.
  const isBusy = isUploading || isRunning || runStatus === "reconnecting";

  // Viewer-specific filters. Any dataset with a root splat (showcase)
  // OR at least one training run (has run_count > 0) is worth listing
  // in the viewer so the user can pick between individual runs.
  const datasetsWithSplats = useMemo(
    () => datasets.filter((d) => d.has_splat || (d.run_count || 0) > 0),
    [datasets],
  );
  const datasetsWithImages = useMemo(
    () => datasets.filter((d) => d.has_images),
    [datasets],
  );
  const selectedEntry = useMemo(
    () => datasetsWithSplats.find((d) => d.name === selectedDataset) || null,
    [datasetsWithSplats, selectedDataset],
  );

  // Viewer runs for the currently-selected dataset (only ones with an
  // actual splat on disk). The effect below fetches on-demand.
  const viewerRuns = useMemo(() => {
    const all = viewerRunsByDataset[selectedDataset] || [];
    return all.filter((r) => r.splat_path);
  }, [viewerRunsByDataset, selectedDataset]);

  const showViewerRunPicker = selectedDataset && viewerRuns.length > 1;

  const viewerRun = useMemo(
    () => viewerRuns.find((r) => (r.run_tag || "__showcase__") === viewerRunTag) || null,
    [viewerRuns, viewerRunTag],
  );
  // Path passed to the <GaussViewer>. When only one run exists we skip
  // the picker and point at that run directly.
  const viewerSplatPath = viewerRun?.splat_path || null;

  // What dataset the pipeline should run against, regardless of mode.
  const activeDatasetName = inputMode === "existing" ? existingDataset : (uploadedDataset || datasetName.trim());
  const pipelineReady = inputMode === "existing"
    ? !!existingDataset && !isBusy
    : uploadStatus === "uploaded" && !isBusy;

  const appendLog = useCallback((line) => {
    const entry = {
      id: crypto.randomUUID(),
      text: line,
      kind: logKind(line),
      ts: new Date().toLocaleTimeString(),
    };
    setLogs((prev) => [...prev, entry]);
  }, []);

  const persistActiveJob = useCallback((payload) => {
    try {
      localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // ignore localStorage failures
    }
  }, []);

  const clearPersistedActiveJob = useCallback(() => {
    try {
      localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
    } catch {
      // ignore localStorage failures
    }
  }, []);

  const closeCurrentWs = useCallback(() => {
    if (!wsRef.current) return;
    try {
      wsRef.current.close();
    } catch {
      // ignore
    }
    wsRef.current = null;
  }, []);

  const refreshDatasets = useCallback(async () => {
    setDatasetsLoading(true);
    try {
      const res = await fetch("/api/datasets");
      if (!res.ok) throw new Error(await parseApiError(res));
      const list = await res.json();
      setDatasets(Array.isArray(list) ? list : []);
    } catch (err) {
      appendLog(`Failed to load datasets: ${String(err)}`);
      setDatasets([]);
    } finally {
      setDatasetsLoading(false);
    }
  }, [appendLog]);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  useEffect(() => {
    if (selectedDataset && !datasetsWithSplats.some((d) => d.name === selectedDataset)) {
      setSelectedDataset("");
    }
  }, [datasetsWithSplats, selectedDataset]);

  // Fetch runs for the selected viewer dataset the first time the user
  // picks it. Subsequent dataset swaps hit the cache.
  useEffect(() => {
    if (!selectedDataset) return;
    if (viewerRunsByDataset[selectedDataset]) return;
    let cancelled = false;
    (async () => {
      const runs = await fetchJson(
        `/api/datasets/${encodeURIComponent(selectedDataset)}/runs`,
        [],
      );
      if (!cancelled) {
        setViewerRunsByDataset((prev) => ({ ...prev, [selectedDataset]: runs || [] }));
      }
    })();
    return () => { cancelled = true; };
  }, [selectedDataset, viewerRunsByDataset]);

  // Auto-pick the single run or clear stale selection when the dataset
  // swaps. Runs are ordered oldest-first on the server, so viewerRuns[0]
  // is "Run 1" and viewerRuns[N-1] is the most recent.
  useEffect(() => {
    if (!selectedDataset) {
      if (viewerRunTag) setViewerRunTag("");
      return;
    }
    if (viewerRuns.length === 0) {
      if (viewerRunTag) setViewerRunTag("");
      return;
    }
    if (viewerRuns.length === 1) {
      const only = viewerRuns[0];
      const key = only.run_tag || "__showcase__";
      if (viewerRunTag !== key) setViewerRunTag(key);
      return;
    }
    const still = viewerRuns.some((r) => (r.run_tag || "__showcase__") === viewerRunTag);
    if (!still) {
      // Default to the newest run so "just finished a training" flows
      // show the latest output immediately.
      const latest = viewerRuns[viewerRuns.length - 1];
      setViewerRunTag(latest.run_tag || "__showcase__");
    }
  }, [selectedDataset, viewerRuns, viewerRunTag]);

  // After a successful run the pipeline writes a new run under the
  // active dataset, so invalidate the viewer cache for that dataset so
  // the new run appears in the picker without a page refresh.
  useEffect(() => {
    if (runStatus !== "success" || !selectedDataset) return;
    setViewerRunsByDataset((prev) => {
      if (!(selectedDataset in prev)) return prev;
      const { [selectedDataset]: _drop, ...rest } = prev;
      return rest;
    });
  }, [runStatus, selectedDataset]);

  // Keep the existing-dataset picker sane: if the selected dataset
  // disappears from the list (another tab deleted it, or the server
  // re-scanned and it no longer has images), clear the selection.
  useEffect(() => {
    if (existingDataset && !datasetsWithImages.some((d) => d.name === existingDataset)) {
      setExistingDataset("");
    }
  }, [datasetsWithImages, existingDataset]);

  // Fetch preprocess fingerprint status for the currently-staged dataset
  // (either freshly uploaded, typed in, or picked from the existing-
  // dataset dropdown) so we can show a "SfM cached" note in Advanced
  // Settings.
  useEffect(() => {
    const target = inputMode === "existing" ? existingDataset : (uploadedDataset || datasetName);
    if (!target) {
      setPrepStatus(null);
      return;
    }
    let cancelled = false;
    (async () => {
      const data = await fetchJson(
        `/api/datasets/${encodeURIComponent(target)}/prep-status`,
        null,
      );
      if (!cancelled) setPrepStatus(data);
    })();
    return () => { cancelled = true; };
  }, [uploadedDataset, datasetName, existingDataset, inputMode, runStatus]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    if (!jobId || runStatus !== "running") {
      return;
    }

    let stopped = false;
    let timerId = null;

    const pollJob = async () => {
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
        // 404 means the server forgot this job - happens after an
        // uvicorn restart wiped JOB_META, or after a job aged out of
        // memory. Treat it as terminal so the UI doesn't sit polling
        // forever pretending the pipeline is still alive.
        if (res.status === 404) {
          if (!stopped && runStatus === "running") {
            setRunStatus("error");
            setRunMessage("Server lost track of this job (likely restarted). Start a new run to continue.");
            clearPersistedActiveJob();
            closeCurrentWs();
          }
          return;
        }
        if (!res.ok) {
          throw new Error(`job poll failed (${res.status})`);
        }
        const meta = await res.json();
        if (stopped) return;

        if (meta.stage) {
          setJobStage(meta.stage);
        }

        if (meta.status === "completed" && runStatus === "running") {
          setRunStatus("success");
          setRunMessage("Pipeline finished successfully.");
          clearPersistedActiveJob();
          closeCurrentWs();
          await refreshDatasets();
          if (meta.dataset) {
            setSelectedDataset(meta.dataset);
          }
          return;
        }

        if (meta.status === "failed" && runStatus === "running") {
          setRunStatus("error");
          setRunMessage(meta.error || "Pipeline failed.");
          clearPersistedActiveJob();
          closeCurrentWs();
          return;
        }
      } catch {
        // polling failures are non-fatal; websocket logs still carry main status
      } finally {
        if (!stopped && runStatus === "running") {
          timerId = window.setTimeout(pollJob, 1500);
        }
      }
    };

    pollJob();

    return () => {
      stopped = true;
      if (timerId !== null) {
        window.clearTimeout(timerId);
      }
    };
  }, [jobId, runStatus, closeCurrentWs, refreshDatasets, clearPersistedActiveJob]);

  // Job recovery after a page refresh. Training can take 10+ minutes, so if
  // the user reloads the page mid-run we want to reattach to the WebSocket
  // and keep streaming logs instead of losing progress. We stash the active
  // job id in localStorage and check it here on mount; if the server still
  // considers that job alive, we hook back up to its log stream.
  //
  // NOTE: no ref guard here. Under React StrictMode the effect is invoked
  // twice on mount (mount -> cleanup -> mount again), and a naive "only
  // run once" ref guard left the UI stuck in "Reconnecting..." forever:
  // the first invocation gets cancelled by the cleanup before the fetch
  // resolves, then the second invocation sees the ref set and bails out.
  // The restore logic is idempotent (same localStorage, same /api/jobs
  // response, setState is a no-op when values match) so letting both
  // invocations run is fine.
  useEffect(() => {
    let cancelled = false;

    const restoreActiveJob = async () => {
      let saved = null;
      try {
        const raw = localStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
        if (raw) {
          saved = JSON.parse(raw);
        }
      } catch {
        saved = null;
      }

      // Pull the rolling log buffer from the server so the stage cards
      // and progress parser have history to work with. Without this, a
      // user navigating back to LiveDemos mid-run sees an empty feed
      // until the next WS line arrives (could be several seconds).
      const hydrateLogsFromServer = async (id) => {
        try {
          const res = await fetch(`/api/jobs/${encodeURIComponent(id)}/logs`);
          if (!res.ok) return;
          const body = await res.json();
          const lines = Array.isArray(body?.lines) ? body.lines : [];
          if (!lines.length) return;
          const hydrated = lines.map((text) => ({
            id: crypto.randomUUID(),
            text,
            kind: logKind(text),
            ts: "replay",
          }));
          setLogs(hydrated);
        } catch {
          // replay is best-effort; WS reconnection still carries live state
        }
      };

      const attachFromMeta = async (meta, sourceLabel, fallbackDataset = "") => {
        if (!meta) return false;
        const resolvedJobId = meta.job_id || meta.jobId;
        if (!resolvedJobId) return false;

        if (meta.status === "running" || meta.status === "queued") {
          setJobId(resolvedJobId);
          setJobStage(meta.stage || "queued");
          setRunStatus("running");
          setRunMessage("Reconnected to running job after refresh.");
          if (meta.dataset) {
            setDatasetName(meta.dataset);
          }
          persistActiveJob({
            jobId: resolvedJobId,
            dataset: meta.dataset || fallbackDataset || "",
            backend: meta.backend || "",
            startedAt: meta.created_at || new Date().toISOString(),
          });
          await hydrateLogsFromServer(resolvedJobId);
          appendLog(`Reconnected to active job ${resolvedJobId} (${sourceLabel}).`);
          openWS(resolvedJobId, meta.dataset || fallbackDataset || "");
          return true;
        }

        if (meta.status === "completed") {
          setJobId(resolvedJobId);
          setJobStage("completed");
          setRunStatus("success");
          setRunMessage("Previous job completed.");
          if (meta.dataset) {
            setSelectedDataset(meta.dataset);
          }
          await hydrateLogsFromServer(resolvedJobId);
          await refreshDatasets();
          clearPersistedActiveJob();
          return true;
        }

        if (meta.status === "failed") {
          setJobId(resolvedJobId);
          setJobStage("failed");
          setRunStatus("error");
          setRunMessage(meta.error || "Previous job failed.");
          await hydrateLogsFromServer(resolvedJobId);
          clearPersistedActiveJob();
          return true;
        }
        return false;
      };

      const fetchJobWithRetry = async (id, retries = 10, delayMs = 700) => {
        for (let attempt = 1; attempt <= retries; attempt += 1) {
          if (cancelled) return null;
          try {
            const res = await fetch(`/api/jobs/${encodeURIComponent(id)}`);
            if (res.ok) {
              return await res.json();
            }
          } catch {
            // transient network/proxy issue; retry
          }
          await new Promise((resolve) => window.setTimeout(resolve, delayMs));
        }
        return null;
      };

      if (saved?.jobId) {
        const savedMeta = await fetchJobWithRetry(saved.jobId, 10, 700);
        if (cancelled) return;
        if (await attachFromMeta(savedMeta, "saved job", saved.dataset || "")) {
          return;
        }
      }

      try {
        const activeRes = await fetch("/api/jobs-active");
        if (activeRes.ok) {
          const activeMeta = await activeRes.json();
          if (cancelled) return;
          if (await attachFromMeta(activeMeta, "backend active job")) {
            return;
          }
        }
      } catch {
        // ignore; fall through to cleanup
      }

      if (saved?.jobId) {
        appendLog(`Could not reconnect to saved job ${saved.jobId}.`);
      }
      clearPersistedActiveJob();
      // Reset the optimistic "reconnecting" state set from the lazy
      // useState initializer - the saved job is gone.
      setRunStatus((prev) => (prev === "reconnecting" ? "idle" : prev));
      setRunMessage((prev) => (prev === "Reconnecting to running job..." ? "" : prev));
      setJobStage((prev) => (prev === "reconnecting" ? "" : prev));
      setJobId((prev) => (saved?.jobId && prev === saved.jobId ? "" : prev));
    };

    restoreActiveJob();

    return () => {
      cancelled = true;
    };
  }, [appendLog, clearPersistedActiveJob, refreshDatasets, persistActiveJob]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {
          // ignore
        }
      }
    };
  }, []);

  // Open a WebSocket to the per-job log stream on the backend. Each line we
  // receive is either a raw log line or a control marker "<<DONE:N>>" /
  // "<<ERROR:...>>" that signals the pipeline finished or blew up. We use
  // ws:// against the current host and let Vite's proxy forward it to the
  // FastAPI server in dev.
  function openWS(id, datasetForResult) {
    closeCurrentWs();

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/api/ws/${id}`);

    ws.onopen = () => appendLog(`Connected to job ${id}`);

    ws.onmessage = async (event) => {
      const line = String(event.data || "");
      appendLog(line);

      if (line.startsWith("<<DONE:")) {
        const codeText = line.replace("<<DONE:", "").replace(">>", "").trim();
        const exitCode = Number(codeText);
        const ok = Number.isInteger(exitCode) && exitCode === 0;

        setRunStatus(ok ? "success" : "error");
        setRunMessage(ok ? "Pipeline finished successfully." : `Pipeline failed with exit code ${codeText}.`);
        clearPersistedActiveJob();

        closeCurrentWs();
        await refreshDatasets();
        if (ok) {
          setSelectedDataset(datasetForResult);
        }
        return;
      }

      if (line.startsWith("<<ERROR:")) {
        const message = line.replace("<<ERROR:", "").replace(">>", "").trim();
        setRunStatus("error");
        setRunMessage(message || "Pipeline error");
        clearPersistedActiveJob();
        closeCurrentWs();
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      appendLog("Disconnected from log stream.");
    };

    ws.onerror = () => {
      appendLog("WebSocket error while streaming logs.");
      setRunMessage((prev) => (prev ? prev : "Log stream interrupted. Continuing with status polling."));
    };

    wsRef.current = ws;
  }

  async function handleCancelJob() {
    if (!jobId) return;
    try {
      const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST",
      });
      if (!res.ok) {
        const errText = await parseApiError(res, "Cancel request failed");
        setRunMessage(`Cancel failed: ${errText}`);
        return;
      }
      setRunStatus("cancelled");
      setRunMessage("Job cancelled.");
    } catch (err) {
      setRunMessage(`Cancel failed: ${err?.message || err}`);
    }
  }

  async function handleUpload(event) {
    event.preventDefault();

    const cleanDataset = datasetName.trim();
    if (!videoFile) {
      setUploadStatus("error");
      setUploadMessage("Choose a video file before uploading.");
      return;
    }
    if (!DATASET_NAME_RE.test(cleanDataset)) {
      setUploadStatus("error");
      setUploadMessage("Dataset name must use letters, numbers, _, -, or .");
      return;
    }

    setUploadStatus("uploading");
    setUploadMessage("Uploading video...");
    setRunStatus("idle");
    setRunMessage("");

    const formData = new FormData();
    formData.append("video", videoFile, videoFile.name);
    formData.append("dataset", cleanDataset);

    try {
      const response = await fetch("/api/upload", { method: "POST", body: formData });
      if (!response.ok) {
        throw new Error(await parseApiError(response));
      }

      const data = await response.json();
      setUploadStatus("uploaded");
      setUploadMessage(`Uploaded ${data.filename} to dataset ${data.dataset}.`);
      setUploadedDataset(data.dataset);
      setDatasetName(data.dataset);
      setSelectedDataset(data.dataset);
      appendLog(`Uploaded file ${data.filename} for dataset ${data.dataset}`);
      await refreshDatasets();
    } catch (err) {
      setUploadStatus("error");
      setUploadMessage(`Upload failed: ${String(err)}`);
      appendLog(`Upload failed: ${String(err)}`);
    }
  }

  async function startPipeline({ advanced }) {
    const targetDataset = inputMode === "existing"
      ? existingDataset
      : (uploadedDataset || datasetName.trim());
    if (!DATASET_NAME_RE.test(targetDataset)) {
      setRunStatus("error");
      setRunMessage(inputMode === "existing"
        ? "Pick an existing dataset first."
        : "Upload a valid dataset first.");
      return;
    }

    setRunStatus("running");
    setRunMessage("Pipeline is running...");
    setLogs([]);
    setJobStage("queued");

    const useExistingFrames = inputMode === "existing";

    const payload = {
      dataset: targetDataset,
      backend: backendChoice,
      // SfM method is only user-tunable in Advanced; simple mode always
      // gets the default (colmap) so teammates triggering a quick run
      // don't accidentally pick up an experimental path.
      sfm_method: advanced ? sfmMethod : "colmap",
      only: "all",
      // Tells the server to skip frame extraction and reuse
      // datasets/<name>/images/ as-is.
      use_existing_frames: useExistingFrames,
      iters: advanced ? Number(numIters) : SIMPLE_PREP_DEFAULTS.iters,
      duplicate_threshold: advanced ? Number(duplicateThreshold) : SIMPLE_PREP_DEFAULTS.duplicateThreshold,
      blur_threshold: advanced ? Number(blurThreshold) : SIMPLE_PREP_DEFAULTS.blurThreshold,
      fps: advanced ? Number(fps) : SIMPLE_PREP_DEFAULTS.fps,
      downscale: advanced ? Number(downscale) : SIMPLE_PREP_DEFAULTS.downscale,
      max_width: advanced ? Number(maxWidth) : SIMPLE_PREP_DEFAULTS.maxWidth,
    };
    // Experiment settings only get sent when Advanced is active. Each
    // shortgs field is only included when its toggle is on so the backend
    // can distinguish "user explicitly set to 0" from "technique disabled".
    if (advanced) {
      payload.seed = Number(seed);
      if (scaleResetEnabled) {
        payload.shortgs_scale_reset_every = Number(scaleResetEvery);
        payload.shortgs_scale_reset_factor = Number(scaleResetFactor);
      }
      if (entropyEnabled) {
        payload.shortgs_entropy_weight = Number(entropyWeight);
      }
      if (progressiveEnabled) {
        payload.shortgs_progressive_resolution = String(progressiveSchedule).trim();
      }
    }

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await parseApiError(response));
      }

      const data = await response.json();
      setJobId(data.job_id);
      appendLog(
        useExistingFrames
          ? `Started ${backendChoice} job ${data.job_id} against existing dataset '${targetDataset}'`
          : `Started ${backendChoice} job ${data.job_id}`,
      );
      persistActiveJob({
        jobId: data.job_id,
        dataset: targetDataset,
        backend: backendChoice,
        startedAt: new Date().toISOString(),
      });
      openWS(data.job_id, targetDataset);
    } catch (err) {
      setRunStatus("error");
      setRunMessage(`Pipeline start failed: ${String(err)}`);
      clearPersistedActiveJob();
      appendLog(`Pipeline start failed: ${String(err)}`);
    }
  }

  const statusText = (() => {
    if (runStatus === "reconnecting") return "Reconnecting...";
    if (isUploading) return "Uploading video";
    if (isRunning) return "Processing pipeline";
    if (runStatus === "success") return "Result ready";
    if (runStatus === "error" || uploadStatus === "error") return "Action required";
    if (uploadStatus === "uploaded") return "Ready to run";
    return "Idle";
  })();

  const statusTone =
    runStatus === "success"
      ? "good"
      : runStatus === "error" || uploadStatus === "error"
        ? "bad"
        : isRunning || isUploading || runStatus === "reconnecting"
          ? "busy"
          : "neutral";

  // Derived pipeline state from the raw log stream. Parsing runs every
  // time logs change (typically once per WebSocket message) and produces a
  // stage-by-stage snapshot plus live training metrics. We then reconcile
  // against the backend's authoritative jobStage so the stage cards stay
  // honest even if the log buffer is missing lines (e.g. reconnect race
  // or buffer overflow) - if the server says we're training, SfM and
  // preprocess are by definition already done.
  const pipelineState = useMemo(() => {
    const parsed = parsePipelineLog(logs);
    // parsed.stages is an array (in STAGE_ORDER). Reconcile by id so
    // the jobStage override flips the right card.
    const stages = parsed.stages.map((stage) => ({ ...stage }));
    const forceDone = (id, detail) => {
      const s = stages.find((x) => x.id === id);
      if (s && s.status !== "done" && s.status !== "skipped" && s.status !== "failed") {
        s.status = "done";
        s.detail = s.detail || detail;
      }
    };
    // Promote a still-pending card to "active" when the backend tells
    // us we're in that stage. Only runs when the log parser hasn't
    // already said something about that card - we never downgrade a
    // done/failed/skipped status here, just give a pending card a
    // reasonable detail line instead of letting it sit on WAITING.
    const forceActive = (id, detail) => {
      const s = stages.find((x) => x.id === id);
      if (s && s.status === "pending") {
        s.status = "active";
        s.detail = detail;
      }
    };
    let sfmEtaSeconds = parsed.sfmEtaSeconds;
    if (jobStage === "fastergs" || jobStage === "opensplat" || jobStage === "completed") {
      forceDone("preprocess", "");
      forceDone("sfm", "Sparse model ready");
      sfmEtaSeconds = null;
    } else if (jobStage === "sfm") {
      forceDone("preprocess", "");
    }
    // Training card should flip the moment the backend dispatches the
    // training job. Without this, the card stays on WAITING for ~90s
    // of SLURM queue + setup before the first iteration log line
    // arrives, which makes the UI look stuck.
    if (jobStage === "fastergs") {
      forceActive("training", "Training Gaussian splats (Faster-GS)");
    } else if (jobStage === "opensplat") {
      forceActive("training", "Training Gaussian splats");
    }
    if (jobStage === "completed") {
      forceDone("training", "Training finished");
      forceDone("publish", "Splat published");
    }
    return { ...parsed, stages, sfmEtaSeconds };
  }, [logs, jobStage]);
  const trainingEtaLabel = pipelineState.trainingLive?.etaSeconds
    ? formatEta(pipelineState.trainingLive.etaSeconds)
    : null;
  const sfmEtaLabel = pipelineState.sfmEtaSeconds
    ? formatEta(pipelineState.sfmEtaSeconds)
    : null;
  // Pick the best single ETA for the bottom-of-panel label: training wins
  // when it's live (most precise), SfM as fallback, nothing otherwise.
  const activeStageEta = trainingEtaLabel || sfmEtaLabel;

  const stageLabel =
    jobStage === "prepare"
      ? "Preparing frames"
      : jobStage === "sfm"
        ? "Running SfM reconstruction"
        : jobStage === "opensplat"
          ? "Training Gaussian splats"
          : jobStage === "fastergs"
            ? "Training Gaussian splats (Faster-GS)"
          : jobStage === "reconnecting"
            ? "Reconnecting to running job"
          : jobStage === "completed"
            ? "Completed"
            : jobStage === "failed"
              ? "Failed"
              : "Queued";

  const progress = (() => {
    if (runStatus === "reconnecting") {
      return { value: 10, label: "Reconnecting to running job", tone: "busy" };
    }
    if (uploadStatus === "uploading") {
      return { value: 8, label: "Uploading source video", tone: "busy" };
    }
    if (runStatus === "running") {
      if (jobStage === "prepare") return { value: 30, label: "Preparing frames", tone: "busy" };
      if (jobStage === "sfm") return { value: 60, label: "Reconstructing with SfM + undistort", tone: "busy" };
      if (jobStage === "fastergs" || jobStage === "opensplat") {
        return { value: 85, label: "Training Gaussian splats", tone: "busy" };
      }
      return { value: 15, label: "Queued / dispatching job", tone: "busy" };
    }
    if (runStatus === "success") return { value: 100, label: "Completed", tone: "good" };
    if (runStatus === "error") return { value: 100, label: "Failed", tone: "bad" };
    if (uploadStatus === "uploaded") return { value: 12, label: "Ready to run", tone: "neutral" };
    return { value: 0, label: "Idle", tone: "neutral" };
  })();

  return (
    <main className="live-demos-page">
      <header className="page-header">
        <h2>Live Pipeline Demo</h2>
        <p>Upload video, run reconstruction, and inspect generated splats in one flow.</p>
      </header>

      <section className="demo-grid">
        <article className="panel">
          <div className="panel-head">
            <h3>1) Upload + Run</h3>
            <span className={`status-pill ${statusTone}`}>{statusText}</span>
          </div>

          <div className="mode-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              className={`mode-tab ${inputMode === "new" ? "active" : ""}`}
              aria-selected={inputMode === "new"}
              onClick={() => setInputMode("new")}
              disabled={isBusy}
            >
              New video
            </button>
            <button
              type="button"
              role="tab"
              className={`mode-tab ${inputMode === "existing" ? "active" : ""}`}
              aria-selected={inputMode === "existing"}
              onClick={() => setInputMode("existing")}
              disabled={isBusy}
            >
              Existing dataset
            </button>
          </div>

          <form onSubmit={handleUpload} className="control-form">
            {inputMode === "new" ? (
              <>
                <label>
                  Dataset name
                  <input
                    type="text"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    disabled={isBusy}
                    placeholder="my_dataset"
                    required
                  />
                </label>

                <label>
                  Video file
                  <input
                    type="file"
                    accept="video/*"
                    disabled={isBusy}
                    onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                    required={inputMode === "new"}
                  />
                </label>
              </>
            ) : (
              <label>
                Existing dataset
                <select
                  value={existingDataset}
                  onChange={(e) => setExistingDataset(e.target.value)}
                  disabled={isBusy || datasetsLoading}
                >
                  <option value="">-- select a dataset --</option>
                  {datasetsWithImages.map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.name}
                      {d.run_count ? ` · ${d.run_count} run${d.run_count === 1 ? "" : "s"}` : ""}
                      {d.has_sfm ? " · SfM cached" : ""}
                    </option>
                  ))}
                </select>
                <span className="field-help">
                  Only datasets with preprocessed images show up here. The
                  pipeline will skip frame extraction and reuse what's on disk.
                </span>
              </label>
            )}

            <label>
              Backend
              <select
                value={backendChoice}
                onChange={(e) => setBackendChoice(e.target.value)}
                disabled={isBusy}
              >
                {BACKEND_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="field-help">
              {BACKEND_OPTIONS.find((option) => option.value === backendChoice)?.help}
            </div>

            {inputMode === "new" ? (
              <div className="file-preview">
                {videoFile ? `Selected: ${videoFile.name}` : "No file selected"}
              </div>
            ) : null}

            <div className="button-row">
              {inputMode === "new" ? (
                <button type="submit" disabled={isBusy || !videoFile}>
                  {isUploading ? "Uploading..." : "Upload Video"}
                </button>
              ) : null}
              <button
                type="button"
                className="secondary"
                disabled={!pipelineReady || advancedActive}
                onClick={() => startPipeline({ advanced: false })}
              >
                {isRunning ? "Running..." : "Run Simple"}
              </button>
              <button
                type="button"
                className="secondary"
                disabled={isRunning}
                onClick={() => setAdvancedActive((v) => !v)}
              >
                {advancedActive ? "Hide Advanced" : "Show Advanced"}
              </button>
              {isRunning && jobId ? (
                <button
                  type="button"
                  className="secondary danger"
                  onClick={handleCancelJob}
                  title="Terminate the local pipeline and request scancel on any running SLURM job"
                >
                  Cancel Job
                </button>
              ) : null}
            </div>
          </form>

          {advancedActive ? (
            <div className="advanced-panel">
              {/* Show the "SfM cached" note only when SfM actually
                  finished and its outputs are still present. Server
                  computes this via _is_sfm_cached; don't OR in the
                  prep fingerprint, since that gets written mid-pipeline
                  before SfM has even started. */}
              {prepStatus?.has_sfm_cached ? (
                <div className="sfm-cache-note">
                  <span className="sfm-cache-badge">SfM cached</span>
                  <span className="sfm-cache-text">
                    This dataset already has SfM output. A rerun with matching
                    preprocessing settings will skip straight to training
                    (~4 min instead of ~15 min). Change any preprocessing
                    setting below to trigger a fresh SfM pass.
                  </span>
                </div>
              ) : null}

              <div className="advanced-section-title">
                Preprocessing <span className="advanced-section-sub">
                  {inputMode === "existing"
                    ? "re-runs filters on the existing raw frames"
                    : "changes here invalidate the cached SfM"}
                </span>
              </div>
              <div className="advanced-grid">
                <label>
                  <span className="advanced-label">
                    SfM method<InfoTip text={SETTING_INFO.sfmMethod} />
                  </span>
                  <select
                    value={sfmMethod}
                    onChange={(e) => setSfmMethod(e.target.value)}
                    disabled={isBusy}
                  >
                    <option value="colmap">COLMAP (CPU, ~8-12 min)</option>
                    <option value="vggt">VGGT (GPU, &lt;1 min)</option>
                  </select>
                </label>
                {/* FPS is the only truly-locked knob in existing-
                    dataset mode: raw frames were already sampled from
                    the video at a fixed step, so changing this after
                    the fact wouldn't actually resample anything. Every
                    other filter (blur / dup / downscale / max_width)
                    re-runs on the raw/ dir. */}
                {inputMode === "new" ? (
                  <label>
                    <span className="advanced-label">
                      Extraction FPS (0 = source)<InfoTip text={SETTING_INFO.fps} />
                    </span>
                    <input
                      type="number"
                      min="0"
                      max="120"
                      step="1"
                      value={fps}
                      onChange={(e) => setFps(e.target.value)}
                      disabled={isBusy}
                    />
                  </label>
                ) : null}
                <label>
                  <span className="advanced-label">
                    Downscale factor<InfoTip text={SETTING_INFO.downscale} />
                  </span>
                  <input
                    type="number"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={downscale}
                    onChange={(e) => setDownscale(e.target.value)}
                    disabled={isBusy}
                  />
                </label>
                <label>
                  <span className="advanced-label">
                    Max output width<InfoTip text={SETTING_INFO.maxWidth} />
                  </span>
                  <input
                    type="number"
                    min="320"
                    max="4096"
                    step="10"
                    value={maxWidth}
                    onChange={(e) => setMaxWidth(e.target.value)}
                    disabled={isBusy}
                  />
                </label>
                <label>
                  <span className="advanced-label">
                    Blur threshold<InfoTip text={SETTING_INFO.blur} />
                  </span>
                  <input
                    type="number"
                    min="0"
                    max="5000"
                    step="1"
                    value={blurThreshold}
                    onChange={(e) => setBlurThreshold(e.target.value)}
                    disabled={isBusy}
                  />
                </label>
                <label>
                  <span className="advanced-label">
                    Duplicate threshold<InfoTip text={SETTING_INFO.dup} />
                  </span>
                  <input
                    type="number"
                    min="0"
                    max="255"
                    step="0.5"
                    value={duplicateThreshold}
                    onChange={(e) => setDuplicateThreshold(e.target.value)}
                    disabled={isBusy}
                  />
                </label>
              </div>

              <div className="advanced-section-title advanced-section-title--spaced">
                Training <span className="advanced-section-sub">fast rerun, cached SfM is reused</span>
              </div>
              <div className="advanced-grid">
                <label>
                  <span className="advanced-label">
                    Iterations<InfoTip text={SETTING_INFO.iters} />
                  </span>
                  <input
                    type="number"
                    min="50"
                    max="100000"
                    step="50"
                    value={numIters}
                    onChange={(e) => setNumIters(e.target.value)}
                    disabled={isBusy}
                  />
                </label>
                <label>
                  <span className="advanced-label">
                    Seed<InfoTip text={SETTING_INFO.seed} />
                  </span>
                  <input
                    type="number"
                    min="0"
                    step="1"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    disabled={isBusy}
                  />
                </label>
              </div>

              {/* Shorter-Splatting experiment controls. Gated behind per-technique
                  toggles so a run without any boxes ticked is a plain baseline.
                  All three are training-only - no SfM impact, cached SfM still reused. */}
              <div className="experiment-panel">
                <div className="experiment-head">
                  <h4>Shorter-Splatting experiments <span className="advanced-section-sub">training only, cached SfM reused</span></h4>
                  <span className="experiment-sub">
                    Baseline unless a technique is toggled on. See <code>/reports</code> to compare runs.
                  </span>
                </div>

                <div className="experiment-grid">
                  <div className="experiment-row">
                    <label className="experiment-toggle">
                      <input
                        type="checkbox"
                        checked={scaleResetEnabled}
                        onChange={(e) => setScaleResetEnabled(e.target.checked)}
                        disabled={isBusy}
                      />
                      <span>Scale reset<InfoTip text={SETTING_INFO.scaleReset} /></span>
                    </label>
                    <div className="experiment-params">
                      <label>
                        <span className="advanced-label">
                          Every (iters)<InfoTip text={SETTING_INFO.scaleResetEvery} />
                        </span>
                        <input
                          type="number"
                          min="1"
                          step="100"
                          value={scaleResetEvery}
                          onChange={(e) => setScaleResetEvery(e.target.value)}
                          disabled={isBusy || !scaleResetEnabled}
                        />
                      </label>
                      <label>
                        <span className="advanced-label">
                          Factor<InfoTip text={SETTING_INFO.scaleResetFactor} />
                        </span>
                        <input
                          type="number"
                          min="0.01"
                          max="1"
                          step="0.01"
                          value={scaleResetFactor}
                          onChange={(e) => setScaleResetFactor(e.target.value)}
                          disabled={isBusy || !scaleResetEnabled}
                        />
                      </label>
                    </div>
                  </div>

                  <div className="experiment-row">
                    <label className="experiment-toggle">
                      <input
                        type="checkbox"
                        checked={entropyEnabled}
                        onChange={(e) => setEntropyEnabled(e.target.checked)}
                        disabled={isBusy}
                      />
                      <span>Entropy constraint<InfoTip text={SETTING_INFO.entropy} /></span>
                    </label>
                    <div className="experiment-params">
                      <label>
                        <span className="advanced-label">
                          Weight (λ)<InfoTip text={SETTING_INFO.entropyWeight} />
                        </span>
                        <input
                          type="number"
                          min="0"
                          step="0.001"
                          value={entropyWeight}
                          onChange={(e) => setEntropyWeight(e.target.value)}
                          disabled={isBusy || !entropyEnabled}
                        />
                      </label>
                    </div>
                  </div>

                  <div className="experiment-row">
                    <label className="experiment-toggle">
                      <input
                        type="checkbox"
                        checked={progressiveEnabled}
                        onChange={(e) => setProgressiveEnabled(e.target.checked)}
                        disabled={isBusy}
                      />
                      <span>Progressive resolution<InfoTip text={SETTING_INFO.progressive} /></span>
                    </label>
                    <div className="experiment-params">
                      <label className="wide">
                        <span className="advanced-label">
                          Schedule<InfoTip text={SETTING_INFO.progressiveSchedule} />
                        </span>
                        <input
                          type="text"
                          value={progressiveSchedule}
                          onChange={(e) => setProgressiveSchedule(e.target.value)}
                          disabled={isBusy || !progressiveEnabled}
                          placeholder="iter:scale,iter:scale,..."
                        />
                      </label>
                    </div>
                  </div>
                </div>
              </div>

              <div className="button-row">
                <button
                  type="button"
                  className="primary"
                  disabled={!pipelineReady}
                  onClick={() => startPipeline({ advanced: true })}
                >
                  {isRunning ? "Running..." : "Run Advanced"}
                </button>
              </div>
            </div>
          ) : null}

          <div className="messages">
            {uploadMessage ? <p>{uploadMessage}</p> : null}
            {runMessage ? <p>{runMessage}</p> : null}
            {jobId ? <p>Job ID: {jobId}</p> : null}
          </div>
        </article>

        <article className="panel logs-panel">
          <div className="panel-head">
            <h3>2) Pipeline Progress</h3>
            <div className="panel-head-actions">
              <button
                type="button"
                className="tiny"
                onClick={() => setShowRawLogs((v) => !v)}
                disabled={logs.length === 0}
              >
                {showRawLogs ? "Hide full logs" : "Show full logs"}
              </button>
              <button
                type="button"
                className="tiny"
                onClick={() => setLogs([])}
                disabled={logs.length === 0}
              >
                Clear
              </button>
            </div>
          </div>

          {/* Stage summary: one card per pipeline stage with status +
              live progress. Default view so users aren't staring at a
              firehose of raw pipeline output. */}
          <StageSummary
            stages={pipelineState.stages}
            trainingLive={pipelineState.trainingLive}
            sfmEtaSeconds={pipelineState.sfmEtaSeconds}
          />

          {showRawLogs && (
            <div ref={logRef} className="log-box" aria-live="polite">
              {logs.length === 0 ? (
                <div className="log-placeholder">Logs will appear here after the pipeline starts.</div>
              ) : (
                logs.map((entry) => (
                  <div key={entry.id} className={`log-line ${entry.kind}`}>
                    <span className="log-time">[{entry.ts}]</span>
                    <span>{entry.text}</span>
                  </div>
                ))
              )}
            </div>
          )}

          {isUploading || isRunning || runStatus === "reconnecting" || runStatus === "success" || runStatus === "error" ? (
            <div className="processing-preview logs-progress">
              <div className="processing-stage">
                Current project progress: {progress.label}
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progress.value}
              >
                <div className={`progress-fill ${progress.tone}`} style={{ width: `${progress.value}%` }} />
              </div>
              <div className="progress-meta">
                {progress.value}% · Stage: {stageLabel}
                {activeStageEta ? <> · ~{activeStageEta} remaining</> : null}
              </div>
            </div>
          ) : null}
        </article>
      </section>

      {/* Viewer is the main focus of the page. The dataset picker sits
          inline in the panel header so it doesn't eat vertical space and
          the viewer gets as much room as possible. */}
      <section className="viewer-section">
        <article className="panel viewer-panel viewer-panel-wide">
          <div className="viewer-head">
            <h3>3) Viewer</h3>
            <div className="viewer-head-controls">
              {datasetsWithSplats.length === 0 ? (
                <span className="empty-inline">No results yet.</span>
              ) : (
                <>
                  <label className="dataset-picker inline">
                    <span className="dataset-picker-label">Dataset</span>
                    <select
                      value={selectedDataset}
                      onChange={(e) => {
                        setSelectedDataset(e.target.value);
                        setViewerRunTag("");
                      }}
                      disabled={datasetsLoading}
                    >
                      <option value="">-- select --</option>
                      {datasetsWithSplats.map((d) => (
                        <option key={d.name} value={d.name}>
                          {d.name}
                          {d.run_count ? ` · ${d.run_count} run${d.run_count === 1 ? "" : "s"}` : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  {showViewerRunPicker ? (
                    <label className="dataset-picker inline">
                      <span className="dataset-picker-label">Run</span>
                      <select
                        value={viewerRunTag}
                        onChange={(e) => setViewerRunTag(e.target.value)}
                      >
                        <option value="">-- select a run --</option>
                        {viewerRuns.map((r) => (
                          <option
                            key={r.run_tag || "__showcase__"}
                            value={r.run_tag || "__showcase__"}
                            title={r.run_tag || r.splat_filename || ""}
                          >
                            {r.is_showcase ? "Showcase" : (r.display_label || `Run ${r.run_number}`)}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                </>
              )}
              <button
                type="button"
                className="tiny"
                onClick={refreshDatasets}
                disabled={datasetsLoading}
              >
                {datasetsLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {viewerRun ? (
            <p className="dataset-path">
              Source: {viewerRun.splat_filename || viewerRun.splat_path}
            </p>
          ) : selectedEntry ? (
            <p className="dataset-path">Source: {selectedEntry.splat_path}</p>
          ) : null}

          <GaussViewer splatApiPath={viewerSplatPath} />
        </article>
      </section>
    </main>
  );
}
