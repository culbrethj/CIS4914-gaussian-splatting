import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import GaussViewer from "../components/GaussViewer";
import "./LiveDemos.css";

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
  downscale: 0.75,
  maxWidth: 1280,
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
    help: "Available, but depends on teammate-specific HPG environment setup.",
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

async function parseApiError(response) {
  const fallback = `Request failed with status ${response.status}`;
  try {
    const body = await response.json();
    if (body?.detail) {
      return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    // intentionally ignore JSON parse errors
  }

  try {
    const text = await response.text();
    return text || fallback;
  } catch {
    return fallback;
  }
}

export default function LiveDemos() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [uploadedDataset, setUploadedDataset] = useState("");
  const [backendChoice, setBackendChoice] = useState("fastergs");

  const [logs, setLogs] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("idle");
  const [uploadMessage, setUploadMessage] = useState("");
  const [runStatus, setRunStatus] = useState("idle");
  const [runMessage, setRunMessage] = useState("");
  const [jobId, setJobId] = useState("");
  const [jobStage, setJobStage] = useState("");

  const [advancedActive, setAdvancedActive] = useState(false);
  const [blurThreshold, setBlurThreshold] = useState(SIMPLE_PREP_DEFAULTS.blurThreshold);
  const [duplicateThreshold, setDuplicateThreshold] = useState(SIMPLE_PREP_DEFAULTS.duplicateThreshold);
  const [fps, setFps] = useState(SIMPLE_PREP_DEFAULTS.fps);
  const [downscale, setDownscale] = useState(SIMPLE_PREP_DEFAULTS.downscale);
  const [maxWidth, setMaxWidth] = useState(SIMPLE_PREP_DEFAULTS.maxWidth);
  const [numIters, setNumIters] = useState(SIMPLE_PREP_DEFAULTS.iters);

  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const wsRef = useRef(null);
  const logRef = useRef(null);
  const restoreAttemptedRef = useRef(false);

  const isUploading = uploadStatus === "uploading";
  const isRunning = runStatus === "running";
  const isBusy = isUploading || isRunning;

  const selectedEntry = useMemo(
    () => datasets.find((d) => d.name === selectedDataset) || null,
    [datasets, selectedDataset],
  );
  const viewerDatasetName = selectedEntry?.name || null;

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
      const withSplats = (list || []).filter((d) => d.has_splat);
      setDatasets(withSplats);
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
    if (selectedDataset && !datasets.some((d) => d.name === selectedDataset)) {
      setSelectedDataset("");
    }
  }, [datasets, selectedDataset]);

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
        if (!stopped) {
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
  // The ref guard stops this from running twice under React StrictMode.
  useEffect(() => {
    if (restoreAttemptedRef.current) return;
    restoreAttemptedRef.current = true;

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
          await refreshDatasets();
          clearPersistedActiveJob();
          return true;
        }

        if (meta.status === "failed") {
          setJobId(resolvedJobId);
          setJobStage("failed");
          setRunStatus("error");
          setRunMessage(meta.error || "Previous job failed.");
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
    const targetDataset = uploadedDataset || datasetName.trim();
    if (!DATASET_NAME_RE.test(targetDataset)) {
      setRunStatus("error");
      setRunMessage("Upload a valid dataset first.");
      return;
    }

    setRunStatus("running");
    setRunMessage("Pipeline is running...");
    setLogs([]);
    setJobStage("queued");

    const payload = {
      dataset: targetDataset,
      backend: backendChoice,
      only: "all",
      iters: advanced ? Number(numIters) : SIMPLE_PREP_DEFAULTS.iters,
      duplicate_threshold: advanced ? Number(duplicateThreshold) : SIMPLE_PREP_DEFAULTS.duplicateThreshold,
      blur_threshold: advanced ? Number(blurThreshold) : SIMPLE_PREP_DEFAULTS.blurThreshold,
      fps: advanced ? Number(fps) : SIMPLE_PREP_DEFAULTS.fps,
      downscale: advanced ? Number(downscale) : SIMPLE_PREP_DEFAULTS.downscale,
      max_width: advanced ? Number(maxWidth) : SIMPLE_PREP_DEFAULTS.maxWidth,
    };

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
      appendLog(`Started ${backendChoice} job ${data.job_id}`);
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
        : isRunning || isUploading
          ? "busy"
          : "neutral";

  const stageLabel =
    jobStage === "prepare"
      ? "Preparing frames"
      : jobStage === "sfm"
        ? "Running SfM reconstruction"
        : jobStage === "opensplat"
          ? "Training Gaussian splats"
          : jobStage === "fastergs"
            ? "Training Gaussian splats (Faster-GS)"
          : jobStage === "completed"
            ? "Completed"
            : jobStage === "failed"
              ? "Failed"
              : "Queued";

  const progress = (() => {
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

          <form onSubmit={handleUpload} className="control-form">
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
                required
              />
            </label>

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

            <div className="file-preview">
              {videoFile ? `Selected: ${videoFile.name}` : "No file selected"}
            </div>

            <div className="button-row">
              <button type="submit" disabled={isBusy || !videoFile}>
                {isUploading ? "Uploading..." : "Upload Video"}
              </button>
              <button
                type="button"
                className="secondary"
                disabled={isBusy || uploadStatus !== "uploaded" || advancedActive}
                onClick={() => startPipeline({ advanced: false })}
              >
                {isRunning ? "Running..." : "Run Simple"}
              </button>
              <button
                type="button"
                className={advancedActive ? "accent" : "secondary"}
                disabled={isRunning}
                onClick={() => setAdvancedActive((v) => !v)}
              >
                {advancedActive ? "Hide Advanced" : "Show Advanced"}
              </button>
            </div>
          </form>

          {advancedActive ? (
            <div className="advanced-panel">
              <div className="advanced-grid">
                <label>
                  Iterations
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
                  Blur threshold
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
                  Duplicate threshold
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
                <label>
                  Extraction FPS (0 = source)
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
                <label>
                  Downscale factor
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
                  Max output width
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
              </div>

              <div className="button-row">
                <button
                  type="button"
                  className="accent"
                  disabled={isBusy || uploadStatus !== "uploaded"}
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
            <h3>2) Pipeline Logs</h3>
            <button type="button" className="tiny" onClick={() => setLogs([])} disabled={logs.length === 0}>
              Clear
            </button>
          </div>

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

          {isUploading || isRunning || runStatus === "success" || runStatus === "error" ? (
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
                {progress.value}% · Stage: {stageLabel} · Stage-based estimate
              </div>
            </div>
          ) : null}
        </article>
      </section>

      <section className="demo-grid lower">
        <article className="panel">
          <div className="panel-head">
            <h3>3) Available Results</h3>
            <button type="button" className="tiny" onClick={refreshDatasets} disabled={datasetsLoading}>
              {datasetsLoading ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          {datasets.length === 0 ? (
            <div className="empty">No generated datasets with `.splat` found yet.</div>
          ) : (
            <label className="dataset-picker">
              Choose dataset
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={datasetsLoading}
              >
                <option value="">-- select dataset --</option>
                {datasets.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
          )}

          {selectedEntry ? <p className="dataset-path">Source: {selectedEntry.splat_path}</p> : null}
        </article>

        <article className="panel viewer-panel">
          <h3>4) Viewer</h3>
          <GaussViewer datasetName={viewerDatasetName} />
        </article>
      </section>

      <div className="back-row">
        <Link to="/">Back</Link>
      </div>
    </main>
  );
}
