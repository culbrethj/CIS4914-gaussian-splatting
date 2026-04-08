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
  const [uploading, setUploading] = useState(false);
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
        setUploading(false);
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
    const f = evt.target.querySelector('input[type="file"]').files[0];
    if (!f) return alert("Choose a file first");
    setUploading(true);
    setLogs([]);
    setJobId(null);

    const fd = new FormData();
    fd.append("video", f, f.name);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "upload failed");
      }
      const data = await res.json();
      setJobId(data.job_id);
      openWS(data.job_id);
    } catch (e) {
      setUploading(false);
      setLogs((l) => [...l, `Upload error: ${String(e)}`]);
    }
  }

  const selectedEntry = datasets.find((d) => d.name === selected) || null;

  return (
    <main style={{ padding: 24 }}>
      <h2>Live Demos</h2>

      <section style={{ marginTop: 12 }}>
        <h3>Upload video & run pipeline</h3>
        <form onSubmit={handleUpload} style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input type="file" accept="video/*" />
          <button type="submit" style={buttonStyle} disabled={uploading}>
            {uploading ? "Uploading..." : "Upload & Run"}
          </button>
        </form>

        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 13, color: "#666" }}>Job id: {jobId || "—"}</div>
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