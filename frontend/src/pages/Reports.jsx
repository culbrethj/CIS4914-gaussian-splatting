import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./Reports.css";

// The Reports page lets you pick a dataset, pick one or more training runs,
// and look at how metrics changed over iterations. Two view modes:
//   - "stacked"  = one row of small charts per selected run
//   - "overlay"  = one chart per metric, with every selected run drawn on top
// Everything here tolerates missing data. If the backend returns no runs for
// the dataset, we show an empty state. If a specific metric is null for
// every selected run, we hide that chart to avoid empty-looking panels.

const METRIC_DEFS = [
  { key: "psnr", label: "PSNR (dB)" },
  { key: "ssim", label: "SSIM" },
  { key: "lpips", label: "LPIPS" },
  { key: "loss", label: "Loss" },
  { key: "num_gaussians", label: "Gaussians" },
  { key: "splats_per_frame", label: "Splats / Frame" },
  { key: "wall_seconds", label: "Wall Time (s)" },
];

// Recharts color palette. Four colors is plenty since we cap overlay at 4 runs.
const RUN_COLORS = ["#0f6bd8", "#c53535", "#0f8f5f", "#a56800"];

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 1000) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return n.toFixed(digits);
}

export default function Reports() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState("");

  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState("");

  const [selectedRunTags, setSelectedRunTags] = useState([]);
  const [seriesByRun, setSeriesByRun] = useState({});
  const [mode, setMode] = useState("overlay"); // "overlay" or "stacked"

  // Load dataset list on mount.
  useEffect(() => {
    let cancelled = false;
    // Clearing is conditional and only happens when the user picks a new
    // dataset. Disabling the lint rule is intentional here.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDatasetsLoading(true);
    fetchJson("/api/datasets")
      .then((data) => {
        if (cancelled) return;
        const names = Array.isArray(data)
          ? data.map((d) => (typeof d === "string" ? d : d?.name)).filter(Boolean)
          : [];
        names.sort();
        setDatasets(names);
        setDatasetsError("");
      })
      .catch((err) => {
        if (cancelled) return;
        setDatasetsError(String(err));
      })
      .finally(() => {
        if (!cancelled) setDatasetsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load run list + summaries when the dataset changes. Resetting the run
  // cache here is required - stale series data from the previous dataset
  // shouldn't show up mixed in with the new one. Lint rule suppressed
  // because the reset is exactly what we want.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!selectedDataset) {
      setRuns([]);
      setSelectedRunTags([]);
      setSeriesByRun({});
      setRunsError("");
      return;
    }
    let cancelled = false;
    setRunsLoading(true);
    setRuns([]);
    setSelectedRunTags([]);
    setSeriesByRun({});
    fetchJson(`/api/datasets/${encodeURIComponent(selectedDataset)}/metrics`)
      .then((data) => {
        if (cancelled) return;
        const list = Array.isArray(data) ? data : [];
        setRuns(list);
        setRunsError("");
      })
      .catch((err) => {
        if (cancelled) return;
        setRunsError(String(err));
      })
      .finally(() => {
        if (!cancelled) setRunsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedDataset]);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Fetch the time-series jsonl for each selected run on demand. Cache in
  // state keyed by run_tag so toggling a run on-off-on doesn't refetch.
  useEffect(() => {
    if (!selectedDataset || selectedRunTags.length === 0) return;
    let cancelled = false;

    const missing = selectedRunTags.filter((tag) => !seriesByRun[tag]);
    if (missing.length === 0) return;

    Promise.all(
      missing.map((tag) =>
        fetchJson(`/api/datasets/${encodeURIComponent(selectedDataset)}/metrics/${encodeURIComponent(tag)}/series`)
          .then((records) => ({ tag, records }))
          .catch((err) => ({ tag, error: String(err) }))
      )
    ).then((results) => {
      if (cancelled) return;
      setSeriesByRun((prev) => {
        const next = { ...prev };
        for (const r of results) {
          next[r.tag] = r.error ? { error: r.error, records: [] } : { records: r.records, error: "" };
        }
        return next;
      });
    });

    return () => {
      cancelled = true;
    };
  }, [selectedDataset, selectedRunTags, seriesByRun]);

  const toggleRun = useCallback((tag) => {
    setSelectedRunTags((prev) => {
      if (prev.includes(tag)) return prev.filter((t) => t !== tag);
      // Cap at 4 runs when comparing - more than that makes every chart a
      // tangled mess and we only have 4 colors anyway.
      if (prev.length >= 4) return [...prev.slice(1), tag];
      return [...prev, tag];
    });
  }, []);

  // Assemble overlay-mode chart data: one array of points per metric, where
  // each point has { iteration, <tag>: value } for each selected run.
  const overlayDataByMetric = useMemo(() => {
    const out = {};
    for (const { key } of METRIC_DEFS) {
      const iterSet = new Set();
      const perRun = {};
      for (const tag of selectedRunTags) {
        const series = seriesByRun[tag]?.records || [];
        perRun[tag] = new Map();
        for (const rec of series) {
          if (rec.iteration == null) continue;
          iterSet.add(rec.iteration);
          const v = rec[key];
          if (v !== null && v !== undefined && !Number.isNaN(Number(v))) {
            perRun[tag].set(rec.iteration, Number(v));
          }
        }
      }
      const iters = [...iterSet].sort((a, b) => a - b);
      const points = iters.map((iter) => {
        const point = { iteration: iter };
        for (const tag of selectedRunTags) {
          const v = perRun[tag].get(iter);
          if (v !== undefined) point[tag] = v;
        }
        return point;
      });
      out[key] = points;
    }
    return out;
  }, [selectedRunTags, seriesByRun]);

  // Which metrics have at least one non-null sample among selected runs? We
  // hide the rest so the grid doesn't show empty SSIM/LPIPS tiles for
  // backends that didn't compute them.
  const visibleMetrics = useMemo(() => {
    return METRIC_DEFS.filter(({ key }) => {
      const points = overlayDataByMetric[key] || [];
      return points.some((p) => selectedRunTags.some((tag) => p[tag] !== undefined));
    });
  }, [overlayDataByMetric, selectedRunTags]);

  const hasRunsSelected = selectedRunTags.length > 0;

  return (
    <main className="reports-page">
      <div className="page-header">
        <h2>Reports / Metrics</h2>
        <p>
          Pick a dataset and one or more training runs to compare PSNR, SSIM, LPIPS,
          loss, gaussian count, splats-per-frame, and wall time.
        </p>
        <Link className="back-link" to="/">Back</Link>
      </div>

      <div className="reports-controls">
        <div>
          <label htmlFor="dataset-select">Dataset</label>
          <select
            id="dataset-select"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            disabled={datasetsLoading}
          >
            <option value="">
              {datasetsLoading ? "Loading datasets..." : "Select a dataset"}
            </option>
            {datasets.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          {datasetsError && (
            <div style={{ color: "var(--bad)", marginTop: 6, fontSize: 13 }}>
              Failed to load datasets: {datasetsError}
            </div>
          )}
        </div>

        {selectedDataset && (
          <>
            <div>
              <label>Runs</label>
              {runsLoading && <div>Loading runs...</div>}
              {!runsLoading && runsError && (
                <div style={{ color: "var(--bad)" }}>Failed to load runs: {runsError}</div>
              )}
              {!runsLoading && !runsError && runs.length === 0 && (
                <div style={{ color: "var(--muted)" }}>
                  No runs with metrics yet for this dataset. Kick off a training job to
                  populate this list.
                </div>
              )}
              {!runsLoading && !runsError && runs.length > 0 && (
                <div className="run-list">
                  {runs.map((run, idx) => {
                    const tag = run.run_tag;
                    const selected = selectedRunTags.includes(tag);
                    const s = run.summary || {};
                    const color = selected ? RUN_COLORS[selectedRunTags.indexOf(tag) % RUN_COLORS.length] : "transparent";
                    return (
                      <label key={tag || idx} className={`run-card${selected ? " selected" : ""}`}>
                        <div style={{ display: "flex", alignItems: "center" }}>
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => toggleRun(tag)}
                          />
                          <span
                            aria-hidden
                            style={{
                              display: "inline-block",
                              width: 10,
                              height: 10,
                              borderRadius: "50%",
                              background: color,
                              marginRight: 8,
                              border: selected ? "none" : "1px solid var(--line)",
                            }}
                          />
                          <span className="run-tag">{tag}</span>
                        </div>
                        <div className="run-meta">
                          {s.backend && <span>backend: {s.backend}</span>}
                          {s.iterations != null && <span>iters: {s.iterations}</span>}
                          {s.final_psnr != null && <span>PSNR: {formatNumber(s.final_psnr, 2)}</span>}
                          {s.final_num_gaussians != null && (
                            <span>gaussians: {formatNumber(s.final_num_gaussians, 0)}</span>
                          )}
                          {s.total_wall_seconds != null && (
                            <span>wall: {formatNumber(s.total_wall_seconds, 0)}s</span>
                          )}
                        </div>
                        <div className="run-actions">
                          <a
                            href={`/api/datasets/${encodeURIComponent(selectedDataset)}/metrics/${encodeURIComponent(tag)}/download`}
                            onClick={(e) => e.stopPropagation()}
                          >
                            Download zip
                          </a>
                        </div>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>

            {hasRunsSelected && (
              <div>
                <label>View</label>
                <div className="reports-mode">
                  <button
                    type="button"
                    className={mode === "overlay" ? "active" : ""}
                    onClick={() => setMode("overlay")}
                  >
                    Overlay runs
                  </button>
                  <button
                    type="button"
                    className={mode === "stacked" ? "active" : ""}
                    onClick={() => setMode("stacked")}
                  >
                    One row per run
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {hasRunsSelected && (
        <>
          <SummaryTable runs={runs.filter((r) => selectedRunTags.includes(r.run_tag))} />

          {mode === "overlay" ? (
            <OverlayCharts
              metrics={visibleMetrics}
              runTags={selectedRunTags}
              dataByMetric={overlayDataByMetric}
            />
          ) : (
            <StackedCharts
              metrics={visibleMetrics}
              runTags={selectedRunTags}
              seriesByRun={seriesByRun}
            />
          )}
        </>
      )}

      {!hasRunsSelected && selectedDataset && runs.length > 0 && (
        <div className="reports-empty">Pick one or more runs above to view metrics.</div>
      )}
    </main>
  );
}

function SummaryTable({ runs }) {
  if (!runs.length) return null;
  return (
    <table className="summary-table">
      <thead>
        <tr>
          <th>Run</th>
          <th>Backend</th>
          <th>Iterations</th>
          <th>Final PSNR</th>
          <th>Final SSIM</th>
          <th>Final LPIPS</th>
          <th>Gaussians</th>
          <th>Wall (s)</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => {
          const s = run.summary || {};
          return (
            <tr key={run.run_tag}>
              <td>{run.run_tag}</td>
              <td>{s.backend || "-"}</td>
              <td>{s.iterations ?? "-"}</td>
              <td>{formatNumber(s.final_psnr, 2)}</td>
              <td>{formatNumber(s.final_ssim, 3)}</td>
              <td>{formatNumber(s.final_lpips, 3)}</td>
              <td>{formatNumber(s.final_num_gaussians, 0)}</td>
              <td>{formatNumber(s.total_wall_seconds, 0)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OverlayCharts({ metrics, runTags, dataByMetric }) {
  if (!metrics.length) {
    return <div className="reports-empty">No metric data recorded for the selected runs.</div>;
  }
  return (
    <div className="charts-grid">
      <div className="charts-row">
        {metrics.map(({ key, label }) => (
          <div key={key} className="chart-card">
            <h4>{label}</h4>
            <div className="chart-sub">{runTags.length} run(s) overlaid</div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dataByMetric[key]} margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5ecf6" />
                  <XAxis dataKey="iteration" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} width={50} />
                  <Tooltip formatter={(v) => formatNumber(v, 3)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {runTags.map((tag, idx) => (
                    <Line
                      key={tag}
                      type="monotone"
                      dataKey={tag}
                      stroke={RUN_COLORS[idx % RUN_COLORS.length]}
                      dot={false}
                      strokeWidth={1.75}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StackedCharts({ metrics, runTags, seriesByRun }) {
  return (
    <div className="charts-grid">
      {runTags.map((tag, tagIdx) => {
        const entry = seriesByRun[tag];
        const records = entry?.records || [];
        const color = RUN_COLORS[tagIdx % RUN_COLORS.length];
        return (
          <div key={tag}>
            <h3 style={{ margin: "0 0 6px" }}>{tag}</h3>
            {entry?.error ? (
              <div className="reports-empty">Failed to load series: {entry.error}</div>
            ) : records.length === 0 ? (
              <div className="reports-empty">No series recorded.</div>
            ) : (
              <div className="charts-row">
                {metrics
                  .filter(({ key }) => records.some((r) => r[key] != null))
                  .map(({ key, label }) => (
                    <div key={key} className="chart-card">
                      <h4>{label}</h4>
                      <div className="chart-wrap">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={records} margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5ecf6" />
                            <XAxis dataKey="iteration" tick={{ fontSize: 11 }} />
                            <YAxis tick={{ fontSize: 11 }} width={50} />
                            <Tooltip formatter={(v) => formatNumber(v, 3)} />
                            <Line
                              type="monotone"
                              dataKey={key}
                              stroke={color}
                              dot={false}
                              strokeWidth={1.75}
                              connectNulls
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
