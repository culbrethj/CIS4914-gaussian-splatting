import React, { useEffect, useState, useRef } from "react";
import { Link } from "react-router-dom";
import GaussViewer from "../components/GaussViewer";
import VideoUpload from "../components/VideoUpload";

/**
 * LiveDemos page:
 * - Upload a video => POST /api/upload => receives { job_id }
 * - Open WebSocket to ws://<host>/api/ws/{job_id} to stream logs
 * - List existing datasets via GET /api/datasets
 * - Click dataset to reveal available files (links served from /datasets/...)
 */

export default function LiveDemos() {
  const [datasets, setDatasets] = useState([]);
  const [selected, setSelected] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [logs, setLogs] = useState([]);
  const [uploadingFile, setUploadingFile] = useState(false); // uploading the video file
  const [uploadComplete, setUploadComplete] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState(null);
  const [running, setRunning] = useState(false); // pipeline run state
  const [datasetName, setDatasetName] = useState(""); // <-- new
  const [advancedActive, setAdvancedActive] = useState(false); // new: whether advanced UI is shown
  // advanced fields:
  const [blurThreshold, setBlurThreshold] = useState(0);
  const [duplicateThreshold, setDuplicateThreshold] = useState(0);
  const [fps, setFps] = useState(30);
  const [downscale, setDownscale] = useState(1);
  const [numIters, setNumIters] = useState(1000);
  const wsRef = useRef(null);
  const logRef = useRef(null);

  useEffect(() => {
    fetch("/api/datasets")
      .then((r) => r.json())
      .then((list) => {
        // only keep datasets that have a .splat
        setDatasets(list.filter((d) => d.has_splat));
      })
      .catch((e) => {
        console.error("datasets fetch failed", e);
        setDatasets([]);
      });
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  function openWS(id) {
    if (wsRef.current) {
      try { wsRef.current.close(); } catch {}
      wsRef.current = null;
    }
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/api/ws/${id}`);
    ws.onopen = () => setLogs((l) => [...l, `CONNECTED to job ${id}`]);
    ws.onmessage = (ev) => {
      setLogs((l) => [...l, ev.data]);
      // close when job done
      if (ev.data.startsWith("<<DONE:") || ev.data.startsWith("<<ERROR:")) {
        setUploadingFile(false);
        setRunning(false); // job finished
        ws.close();
        wsRef.current = null;
        // refresh datasets list after job finishes (only keep those with splat)
        fetch("/api/datasets")
          .then((r) => r.json())
          .then((list) => setDatasets(list.filter((d) => d.has_splat)))
          .catch(() => {});
      }
    };
    ws.onclose = () => setLogs((l) => [...l, `DISCONNECTED job ${id}`]);
    ws.onerror = (e) => setLogs((l) => [...l, `WS error: ${String(e)}`]);
    wsRef.current = ws;
  }

  async function handleUpload(evt) {
    evt.preventDefault();
    const fileInput = evt.target.querySelector('input[type="file"]');
    const f = fileInput.files[0];
    if (!f) return alert("Choose a file first");

    // require dataset name
    if (!datasetName || !/^[A-Za-z0-9_.-]+$/.test(datasetName)) {
      return alert("Enter a valid dataset name (letters, numbers, underscore, dash, dot).");
    }

    setUploadingFile(true);
    setUploadComplete(false);
    setUploadedDataset(null);

    const fd = new FormData();
    fd.append("video", f, f.name);
    fd.append("dataset", datasetName);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "upload failed");
      }
      const data = await res.json();
      // returned: { dataset, filename }
      setUploadedDataset(data.dataset);
      setUploadComplete(true);
      // keep datasetName in sync with returned (sanitized) name
      setDatasetName(data.dataset);
      setLogs((l) => [...l, `Uploaded ${data.filename} as dataset ${data.dataset}`]);
    } catch (e) {
      setUploadComplete(false);
      setLogs((l) => [...l, `Upload error: ${String(e)}`]);
    } finally {
      setUploadingFile(false);
    }
  }

  async function runSimple() {
    if (!uploadedDataset) return;
    setRunning(true);
    setLogs([]);
    setJobId(null);

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: uploadedDataset, iters: 1000, only: "all" }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "run failed");
      }
      const data = await res.json();
      setJobId(data.job_id);
      openWS(data.job_id);
    } catch (e) {
      setRunning(false);
      setLogs((l) => [...l, `Run error: ${String(e)}`]);
    }
  }

  // new: run with advanced parameters
  async function runAdvanced() {
    if (!uploadedDataset) return;
    setRunning(true);
    setLogs([]);
    setJobId(null);

    const payload = {
      dataset: uploadedDataset,
      iters: Number(numIters),
      only: "all",
      // extra advanced params sent to backend (backend may ignore unknown keys)
      blur_threshold: Number(blurThreshold),
      duplicate_threshold: Number(duplicateThreshold),
      fps: Number(fps),
      downscale: Number(downscale),
    };

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "run failed");
      }
      const data = await res.json();
      setJobId(data.job_id);
      openWS(data.job_id);
    } catch (e) {
      setRunning(false);
      setLogs((l) => [...l, `Run error: ${String(e)}`]);
    }
  }

  const selectedEntry = datasets.find((d) => d.name === selected) || null;

  return (
    <main style={{ padding: 24 }}>
      <h2>Live Demos</h2>

      <section style={{ marginTop: 12 }}>
        <h3>Upload video</h3>
        <form onSubmit={handleUpload} style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input
            type="text"
            placeholder="dataset name (e.g. my_dataset)"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            disabled={uploadingFile || running}
            style={{ padding: "8px 10px", borderRadius: 8 }}
          />
          <input type="file" accept="video/*" disabled={uploadingFile || running} />
          <button type="submit" style={buttonStyle} disabled={uploadingFile || running}>
            {uploadingFile ? "Uploading..." : "Upload"}
          </button>
        </form>

        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 13, color: "#666" }}>
            Upload status: {uploadingFile ? "Uploading…" : uploadComplete ? `Uploaded (${uploadedDataset})` : "—"}
          </div>
          <div
            ref={logRef}
            style={{
              marginTop: 8,
              background: "#0b1220",
              color: "#dfefff",
              padding: 12,
              borderRadius: 8,
              height: 220,
              overflow: "auto",
              fontFamily: "monospace",
              fontSize: 12,
            }}
            aria-live="polite"
          >
            {logs.map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
          <button
            style={{ ...buttonStyle, background: uploadComplete && !running && !advancedActive ? "#1f6feb" : "#999" }}
            onClick={runSimple}
            disabled={!uploadComplete || running || advancedActive}
          >
            {running ? "Running..." : "Run (Simple)"}
          </button>

          <button
            onClick={() => setAdvancedActive((v) => !v)}
            style={{
              ...buttonStyle,
              background: advancedActive ? "#1f6feb" : "#ccc",
              color: advancedActive ? "#fff" : "#333",
            }}
            disabled={running}
          >
            {advancedActive ? "Advanced (active)" : "Advanced"}
          </button>
        </div>

        {/* Advanced options panel */}
        {advancedActive && (
          <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: "#f5f7fb", maxWidth: 720 }}>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
                Blur threshold
                <input
                  type="number"
                  value={blurThreshold}
                  onChange={(e) => setBlurThreshold(e.target.value)}
                  style={{ padding: "6px 8px", borderRadius: 6 }}
                />
              </label>

              <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
                Duplicate threshold
                <input
                  type="number"
                  value={duplicateThreshold}
                  onChange={(e) => setDuplicateThreshold(e.target.value)}
                  style={{ padding: "6px 8px", borderRadius: 6 }}
                />
              </label>

              <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
                FPS
                <input
                  type="number"
                  value={fps}
                  onChange={(e) => setFps(e.target.value)}
                  style={{ padding: "6px 8px", borderRadius: 6 }}
                />
              </label>

              <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
                Downscale factor
                <input
                  type="number"
                  value={downscale}
                  onChange={(e) => setDownscale(e.target.value)}
                  style={{ padding: "6px 8px", borderRadius: 6 }}
                />
              </label>

              <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
                Num iterations
                <input
                  type="number"
                  value={numIters}
                  onChange={(e) => setNumIters(e.target.value)}
                  style={{ padding: "6px 8px", borderRadius: 6 }}
                />
              </label>
            </div>

            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button
                onClick={runAdvanced}
                style={{ ...buttonStyle, background: uploadComplete && !running ? "#1f6feb" : "#999" }}
                disabled={!uploadComplete || running}
              >
                {running ? "Running..." : "Run (Advanced)"}
              </button>

              <button
                onClick={() => setAdvancedActive(false)}
                style={{ ...buttonStyle, background: "#eee", color: "#333" }}
                disabled={running}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </section>

      <section style={{ marginTop: 24 }}>
        <h3>Available datasets</h3>

        <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 12 }}>
          {datasets.length === 0 ? (
            <div style={{ color: "#666" }}>No datasets with .splat found</div>
          ) : (
            <select
              value={selected || ""}
              onChange={(e) => setSelected(e.target.value || null)}
              style={{ padding: "8px 10px", borderRadius: 8 }}
            >
              <option value="">-- select dataset --</option>
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name}
                </option>
              ))}
            </select>
          )}

          {selectedEntry ? (
            <div style={{ color: "#666", fontSize: 13 }}>{selectedEntry.splat_path}</div>
          ) : null}
        </div>
      </section>

      <section style={{ marginTop: 24 }}>
        <h3>Viewer</h3>
        <div style={{ padding: 12, borderRadius: 12, background: "#fff", boxShadow: "0 4px 10px rgba(0,0,0,0.06)" }}>
          <GaussViewer datasetName={selected} />
        </div>
      </section>

      <div style={{marginTop:24}}>
        <Link to="/">Back</Link>
      </div>
    </main>
  );
}

const buttonStyle = {
  padding: "10px 14px",
  borderRadius: 10,
  background: "#1f6feb",
  color: "#fff",
  border: "none",
  fontWeight: 700,
  cursor: "pointer",
};