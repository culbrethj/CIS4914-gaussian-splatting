import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import RunDetailsCard from "../components/RunDetailsCard";
import { formatRunDisplay, parseRunTag, computeRunNumberMap } from "../utils/runDisplay";
import { formatMetric, formatNumber } from "../utils/reportsFormat";
import "./Reports.css";

// The Reports page lets you pick a dataset, pick one or more training runs,
// and look at how metrics changed over training. Two view modes:
//   - "stacked"  = one row of small charts per selected run
//   - "overlay"  = one chart per metric, with every selected run drawn on top
// Everything here tolerates missing data. If the backend returns no runs for
// the dataset, we show an empty state. If a specific metric is null for
// every selected run, we hide that chart to avoid empty-looking panels.
//
// The "iteration" charts use the dense loss/PSNR records from the jsonl.
// The "wall time" chart replots PSNR with wall_seconds on the x axis so you
// can eyeball actual speedup (the Shorter-Splatting paper's core claim).
// The histogram charts come from the summary JSON, not the jsonl - they're
// a single snapshot of the final model's scale + opacity distribution.

// Charts where x = iteration. We only expose metrics our training pipeline
// actually produces. SSIM and LPIPS aren't computed by our trainer config
// (see the written report's Testing Information section for details); the
// backend endpoints still accept the keys in case a future run emits them.
// splats_per_frame falls back to the current gaussian count when the
// rasterizer doesn't expose a per-frame list, so it duplicates Gaussians
// here - removed to avoid promising a metric we don't really have.
const METRIC_DEFS = [
  { key: "psnr", label: "PSNR (dB)" },
  { key: "loss", label: "Loss" },
  { key: "num_gaussians", label: "Gaussians" },
  { key: "wall_seconds", label: "Wall Time (s)" },
];

// Short always-visible explainers under each chart title. Readers who
// don't live in 3DGS land need a line of context per metric.
const METRIC_EXPLAINERS = {
  psnr: "Peak Signal-to-Noise Ratio - how close rendered frames match training frames. Higher is better. 20-25 dB weak, 28-32 good, 35+ excellent.",
  loss: "L1 + D-SSIM loss (what the optimizer minimizes). Lower is better. Good runs end below 0.05. Spikes are normal (opacity resets, densification).",
  num_gaussians: "Number of 3D gaussians in the model at this iteration. No strict better/worse - depends on scene complexity. Typical scenes end 100k-3M.",
  wall_seconds: "Total elapsed seconds from training start. Lower is better at matched quality. Small scenes at 10k iters: ~200-500s on B200.",
};

// Six distinct colors. Chosen to stay distinguishable under normal and
// red/green-colorblind viewing. Cap overlay at 6 runs.
const RUN_COLORS = [
  "#0f6bd8", // blue
  "#c53535", // red
  "#0f8f5f", // green
  "#a56800", // amber
  "#7e41b8", // purple
  "#19858a", // teal
];
const BASELINE_COLOR = "#0f6bd8"; // blue
const SHORTGS_COLOR = "#c53535"; // red
const MAX_OVERLAY_RUNS = RUN_COLORS.length;

// Color assignment: baseline=blue, shortgs runs=red, others picked from the
// palette. Keeps chart colors predictable across datasets.
function colorForRun(runTag, summary, occurrences) {
  const isShortgs = (summary?.shortgs?.scale_reset_every > 0)
    || (summary?.shortgs?.entropy_weight > 0)
    || !!summary?.shortgs?.progressive_resolution
    || runTag.includes("shortgs");
  const isBaseline = !isShortgs;
  // occurrences maps a canonical color to how many runs already used it.
  // First baseline gets blue, first shortgs gets red, overflow gets the
  // palette (skipping blue + red).
  if (isBaseline && !occurrences.baseline) {
    occurrences.baseline = 1;
    return BASELINE_COLOR;
  }
  if (isShortgs && !occurrences.shortgs) {
    occurrences.shortgs = 1;
    return SHORTGS_COLOR;
  }
  // Fallback: skip blue+red so the same run type doesn't get a confusing
  // near-match color for a sibling.
  const fallback = RUN_COLORS.filter((c) => c !== BASELINE_COLOR && c !== SHORTGS_COLOR);
  const idx = (occurrences.fallback = (occurrences.fallback || 0) + 1) - 1;
  return fallback[idx % fallback.length];
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

// Display name for a run (see utils/runDisplay.js for the single source of
// truth). Kept as a thin alias so the hundreds of call sites below don't
// have to be rewritten.
function shortLabelForTag(tag) {
  return formatRunDisplay(tag);
}

const formatAxisTick = (value) => formatMetric(value, undefined, { axis: true });
const formatMetricValue = (metricKey, value) => formatMetric(value, metricKey);

// Custom styled tooltip card for the line charts. Recharts' default
// tooltip is text-heavy and doesn't let us format per-metric values
// (commas for counts, "dB" for PSNR, "2m 3s" for wall time). We
// replicate the layout as a white card with a bold header (iteration
// or wall time) and one row per run with its color swatch + name +
// formatted value.
function ChartTooltip({ active, payload, label, metricKey, labelKey = "iteration", labelFormatter }) {
  if (!active || !payload || payload.length === 0) return null;
  const headerLabel = (() => {
    if (labelFormatter) return labelFormatter(label);
    if (labelKey === "iteration") return `Iteration ${formatMetricValue("num_gaussians", label)}`;
    if (labelKey === "wall_seconds") return `Wall ${formatMetricValue("wall_seconds", label)}`;
    return String(label ?? "");
  })();
  return (
    <div className="chart-tooltip-card">
      <div className="chart-tooltip-head">{headerLabel}</div>
      <div className="chart-tooltip-rows">
        {payload.map((p) => (
          <div key={p.dataKey || p.name} className="chart-tooltip-row">
            <span className="chart-tooltip-dot" style={{ background: p.color || p.stroke || p.fill }} />
            <span className="chart-tooltip-name" title={p.name}>{p.name}</span>
            <span className="chart-tooltip-value">{formatMetricValue(metricKey, p.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Histogram bars use a single-series tooltip with the bin range in the
// header and the count below. Split out from ChartTooltip so the bin
// range (from payload[0].payload.range) can drive the header without
// juggling labelFormatter/labelKey conventions.
function HistogramTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0];
  const range = p?.payload?.range || "";
  return (
    <div className="chart-tooltip-card">
      <div className="chart-tooltip-head">{range}</div>
      <div className="chart-tooltip-rows">
        <div className="chart-tooltip-row">
          <span className="chart-tooltip-dot" style={{ background: p.color || p.fill }} />
          <span className="chart-tooltip-name">Count</span>
          <span className="chart-tooltip-value">{formatMetricValue("num_gaussians", p.value)}</span>
        </div>
      </div>
    </div>
  );
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
      // Cap at MAX_OVERLAY_RUNS so every run has a distinct color. Exceeding
      // the cap just drops the oldest selection.
      if (prev.length >= MAX_OVERLAY_RUNS) return [...prev.slice(1), tag];
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

  // Which metrics have at least one non-null sample among selected runs?
  // Keeps empty charts from showing up when a backend doesn't emit a
  // particular metric.
  const visibleMetrics = useMemo(() => {
    return METRIC_DEFS.filter(({ key }) => {
      const points = overlayDataByMetric[key] || [];
      return points.some((p) => selectedRunTags.some((tag) => p[tag] !== undefined));
    });
  }, [overlayDataByMetric, selectedRunTags]);

  // PSNR vs wall-time chart data. For each selected run we zip the series'
  // wall_seconds + psnr into (x, y) points. This is the Shorter-Splatting
  // paper's headline comparison - speedup at constant quality shows up as
  // two curves reaching the same PSNR but the faster one using less wall time.
  const psnrVsTimeByRun = useMemo(() => {
    const out = {};
    for (const tag of selectedRunTags) {
      const series = seriesByRun[tag]?.records || [];
      const pts = [];
      for (const rec of series) {
        const t = rec.wall_seconds;
        const p = rec.psnr;
        if (t == null || p == null) continue;
        pts.push({ wall_seconds: Number(t), psnr: Number(p) });
      }
      pts.sort((a, b) => a.wall_seconds - b.wall_seconds);
      out[tag] = pts;
    }
    return out;
  }, [selectedRunTags, seriesByRun]);
  const anyWallTimeData = useMemo(
    () => Object.values(psnrVsTimeByRun).some((arr) => arr.length > 0),
    [psnrVsTimeByRun]
  );

  // Histogram data for scale + opacity distributions (from the run summary).
  // Rendering: one bar chart per run per histogram kind, where the bars
  // show count per bin. We don't overlay histograms on the same chart
  // because the bin edges typically differ run-to-run.
  const histogramsByRun = useMemo(() => {
    const out = {};
    const summaryByTag = Object.fromEntries(runs.map((r) => [r.run_tag, r.summary || {}]));
    for (const tag of selectedRunTags) {
      out[tag] = (summaryByTag[tag] || {}).histograms || null;
    }
    return out;
  }, [runs, selectedRunTags]);
  const anyScaleHist = useMemo(
    () => Object.values(histogramsByRun).some((h) => h && h.scale && h.scale.counts?.length),
    [histogramsByRun]
  );
  const anyOpacityHist = useMemo(
    () => Object.values(histogramsByRun).some((h) => h && h.opacity && h.opacity.counts?.length),
    [histogramsByRun]
  );

  // Deterministic color assignment per run tag. Baseline runs land on blue,
  // shortgs runs on red; anything beyond one of each gets a fallback palette
  // color. Recomputed whenever the selection or the run summaries change.
  const colorByRunTag = useMemo(() => {
    const summaryByTag = Object.fromEntries(runs.map((r) => [r.run_tag, r.summary || {}]));
    const occurrences = {};
    const out = {};
    for (const tag of selectedRunTags) {
      out[tag] = colorForRun(tag, summaryByTag[tag], occurrences);
    }
    return out;
  }, [selectedRunTags, runs]);

  // Per-dataset run numbering: "Run 1" is the earliest run for this dataset.
  // We only ever look at one dataset at a time on this page, so a single
  // ascending sort is enough. computeRunNumberMap returns {tag -> N}.
  const runNumberByTag = useMemo(() => {
    const ascending = [...runs]
      .map((r) => ({ tag: r.run_tag, when: parseRunTag(r.run_tag).datetime }))
      .filter((x) => x.when instanceof Date && !Number.isNaN(x.when.getTime()))
      .sort((a, b) => a.when - b.when)
      .map((x) => x.tag);
    return computeRunNumberMap(ascending);
  }, [runs]);

  const hasRunsSelected = selectedRunTags.length > 0;

  return (
    <main className="reports-page">
      <div className="page-header">
        <h2>Reports / Metrics</h2>
        <p>
          Pick a dataset and one or more training runs to compare PSNR,
          loss, gaussian count, and wall time.
        </p>
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
                  {runs.map((run, idx) => (
                    <RunPickerCard
                      key={run.run_tag || idx}
                      run={run}
                      dataset={selectedDataset}
                      selected={selectedRunTags.includes(run.run_tag)}
                      color={colorByRunTag[run.run_tag] || null}
                      runNumber={runNumberByTag[run.run_tag]}
                      onToggle={() => toggleRun(run.run_tag)}
                    />
                  ))}
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
          <SummaryTable
            runs={runs.filter((r) => selectedRunTags.includes(r.run_tag))}
            runNumberByTag={runNumberByTag}
          />

          {mode === "overlay" ? (
            <OverlayCharts
              metrics={visibleMetrics}
              runTags={selectedRunTags}
              dataByMetric={overlayDataByMetric}
              colorByRunTag={colorByRunTag}
              runNumberByTag={runNumberByTag}
            />
          ) : (
            <StackedCharts
              metrics={visibleMetrics}
              runTags={selectedRunTags}
              seriesByRun={seriesByRun}
              colorByRunTag={colorByRunTag}
              runNumberByTag={runNumberByTag}
            />
          )}

          {anyWallTimeData && (
            <PsnrVsWallTimeChart
              runTags={selectedRunTags}
              pointsByRun={psnrVsTimeByRun}
              colorByRunTag={colorByRunTag}
              runNumberByTag={runNumberByTag}
            />
          )}

          {(anyScaleHist || anyOpacityHist) && (
            <HistogramCharts
              runTags={selectedRunTags}
              histogramsByRun={histogramsByRun}
              showScale={anyScaleHist}
              showOpacity={anyOpacityHist}
              colorByRunTag={colorByRunTag}
              runNumberByTag={runNumberByTag}
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

// One card in the run picker grid. Shows the scannable "dataset - Run N"
// label + a compact stats strip. Reveals the full RunDetailsCard on hover.
// The popover is rendered through a portal attached to document.body so it
// escapes the scrolling run-list container's overflow clipping.
function RunPickerCard({ run, dataset, selected, color, runNumber, onToggle }) {
  const [hoverOpen, setHoverOpen] = useState(false);
  const [popoverPos, setPopoverPos] = useState(null);
  const cardRef = useRef(null);
  const tag = run.run_tag;
  const s = run.summary || {};
  const displayName = formatRunDisplay(tag, runNumber);

  // Position the portal popover relative to the card. Prefer placing it
  // to the right, fall back to left, fall back to below, so the card is
  // always visible regardless of where the card sits on screen.
  const onEnter = () => {
    const card = cardRef.current;
    if (!card) return;
    const rect = card.getBoundingClientRect();
    const popWidth = 320; // matches .run-details-card--popover
    const popEstHeight = 380;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let left = rect.right + 10;
    let top = rect.top;
    if (left + popWidth > vw - 12) {
      // not enough room on the right, try left
      left = rect.left - popWidth - 10;
      if (left < 12) {
        // fall back to below the card
        left = Math.max(12, Math.min(rect.left, vw - popWidth - 12));
        top = rect.bottom + 8;
      }
    }
    // clamp vertically so tall cards don't overflow the viewport
    top = Math.max(12, Math.min(top, vh - popEstHeight - 12));
    setPopoverPos({ top, left });
    setHoverOpen(true);
  };
  const onLeave = () => setHoverOpen(false);

  return (
    <label
      ref={cardRef}
      className={`run-card${selected ? " selected" : ""}`}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
    >
      <div className="run-card-head">
        <input type="checkbox" checked={selected} onChange={onToggle} />
        <span
          aria-hidden
          className="run-card-swatch"
          style={{
            background: selected ? (color || "transparent") : "transparent",
            borderColor: selected && color ? color : "var(--line)",
          }}
        />
        <span className="run-tag">{displayName}</span>
      </div>
      <div className="run-meta">
        {s.final_psnr != null && (
          <span className="run-meta-item">
            <span className="run-meta-key">PSNR</span>
            <span className="run-meta-val">{formatNumber(s.final_psnr, 2)}</span>
          </span>
        )}
        {s.total_wall_seconds != null && (
          <span className="run-meta-item">
            <span className="run-meta-key">Wall</span>
            <span className="run-meta-val">{formatNumber(s.total_wall_seconds, 0)}s</span>
          </span>
        )}
        {s.final_num_gaussians != null && (
          <span className="run-meta-item">
            <span className="run-meta-key">Gauss</span>
            <span className="run-meta-val">{formatNumber(s.final_num_gaussians, 0)}</span>
          </span>
        )}
        {s.iterations != null && (
          <span className="run-meta-item">
            <span className="run-meta-key">Iter</span>
            <span className="run-meta-val">{s.iterations}</span>
          </span>
        )}
      </div>
      <div className="run-actions">
        <a
          href={`/api/datasets/${encodeURIComponent(dataset)}/metrics/${encodeURIComponent(tag)}/download`}
          onClick={(e) => e.stopPropagation()}
        >
          Download zip
        </a>
      </div>
      {hoverOpen && popoverPos && createPortal(
        <div
          className="run-card-popover run-card-popover--portal"
          role="tooltip"
          style={{ top: popoverPos.top, left: popoverPos.left }}
        >
          <RunDetailsCard summary={s} runTag={tag} runNumber={runNumber} variant="popover" />
        </div>,
        document.body,
      )}
    </label>
  );
}

function SummaryTable({ runs, runNumberByTag }) {
  if (!runs.length) return null;
  return (
    <table className="summary-table">
      <thead>
        <tr>
          <th>Run</th>
          <th>Backend</th>
          <th>Iters</th>
          <th>PSNR</th>
          <th>Gaussians</th>
          <th>Wall (s)</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => {
          const s = run.summary || {};
          return (
            <tr key={run.run_tag}>
              <td className="run-cell" title={run.run_tag}>
                {shortLabelForTag(run.run_tag, (runNumberByTag || {})[run.run_tag])}
              </td>
              <td>{s.backend || "-"}</td>
              <td>{s.iterations ?? "-"}</td>
              <td>{formatNumber(s.final_psnr, 2)}</td>
              <td>{formatNumber(s.final_num_gaussians, 0)}</td>
              <td>{formatNumber(s.total_wall_seconds, 0)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OverlayCharts({ metrics, runTags, dataByMetric, colorByRunTag, runNumberByTag }) {
  // Loss is noisy (opacity-reset spikes at multiples of ~3000 iters stretch
  // the axis). Give just the loss chart a log-scale toggle so the reader
  // can see the descent without the spikes dominating.
  // Hook must be called unconditionally, before any early return.
  const [lossLog, setLossLog] = useState(false);
  if (!metrics.length) {
    return <div className="reports-empty">No metric data recorded for the selected runs.</div>;
  }
  return (
    <div className="charts-grid">
      <div className="charts-row">
        {metrics.map(({ key, label }) => {
          const isLoss = key === "loss";
          const yScale = isLoss && lossLog ? "log" : "auto";
          // Non-log charts pin the y-axis to zero so line curves sit
          // against a real baseline instead of a stretched auto range.
          // Log scale can't include zero, so the loss-log view keeps
          // its tiny epsilon floor.
          const yDomain = isLoss && lossLog ? [0.001, "auto"] : [0, "auto"];
          return (
            <div key={key} className="chart-card">
              <div className="chart-head">
                <h4>{label}</h4>
                {isLoss && (
                  <button
                    type="button"
                    className="chart-toggle"
                    onClick={() => setLossLog((v) => !v)}
                  >
                    {lossLog ? "Linear" : "Log"}
                  </button>
                )}
              </div>
              <div className="chart-sub">
                {runTags.length} run(s) overlaid
                {isLoss && " · spikes at ~3k/6k/9k are opacity resets (expected)"}
              </div>
              {METRIC_EXPLAINERS[key] && (
                <div className="chart-explainer">{METRIC_EXPLAINERS[key]}</div>
              )}
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dataByMetric[key]} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5ecf6" />
                    {/* type="number" + explicit domain forces recharts to
                        treat iteration as a numeric axis and start at 0. */}
                    <XAxis
                      dataKey="iteration"
                      type="number"
                      domain={[0, "dataMax"]}
                      tick={{ fontSize: 11 }}
                      tickFormatter={formatAxisTick}
                    />
                    <YAxis
                      tick={{ fontSize: 11 }}
                      width={72}
                      scale={yScale}
                      domain={yDomain}
                      tickFormatter={formatAxisTick}
                    />
                    <Tooltip
                      content={<ChartTooltip metricKey={key} labelKey="iteration" />}
                      cursor={{ stroke: "#cbd5e1", strokeWidth: 1 }}
                      isAnimationActive={false}
                      wrapperStyle={{ pointerEvents: "none" }}
                      allowEscapeViewBox={{ x: true, y: true }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {runTags.map((tag) => (
                      <Line
                        key={tag}
                        type="monotone"
                        name={shortLabelForTag(tag, (runNumberByTag || {})[tag])}
                        dataKey={tag}
                        stroke={colorByRunTag[tag] || BASELINE_COLOR}
                        dot={false}
                        strokeWidth={1.75}
                        connectNulls
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// PSNR plotted against wall-clock time instead of iteration. The paper's
// main claim is speedup - if a technique converges to the same PSNR in less
// wall time than the baseline, its curve sits to the left of the baseline's.
function PsnrVsWallTimeChart({ runTags, pointsByRun, colorByRunTag, runNumberByTag }) {
  return (
    <div className="charts-grid section-spaced">
      <h3 className="section-title">PSNR vs wall time</h3>
      <div className="charts-row">
        <div className="chart-card">
          <h4>PSNR (dB) vs wall time (s)</h4>
          <div className="chart-sub">
            Speedup view. Curves hitting the same PSNR faster land farther left.
          </div>
          <div className="chart-explainer">
            PSNR over elapsed wall-clock time instead of iteration. A technique
            is faster-at-equal-quality if its curve reaches a given PSNR to the
            left of the baseline.
          </div>
          <div className="chart-wrap tall">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5ecf6" />
                <XAxis
                  type="number"
                  dataKey="wall_seconds"
                  domain={[0, "dataMax"]}
                  label={{ value: "Wall time (s)", position: "insideBottom", offset: -2, fontSize: 11 }}
                  tick={{ fontSize: 11 }}
                  tickFormatter={formatAxisTick}
                />
                <YAxis
                  type="number"
                  dataKey="psnr"
                  domain={[0, "auto"]}
                  tick={{ fontSize: 11 }}
                  width={60}
                  tickFormatter={formatAxisTick}
                />
                <Tooltip
                  content={<ChartTooltip metricKey="psnr" labelKey="wall_seconds" />}
                  cursor={{ stroke: "#cbd5e1", strokeWidth: 1 }}
                  isAnimationActive={false}
                  wrapperStyle={{ pointerEvents: "none" }}
                  allowEscapeViewBox={{ x: true, y: true }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {runTags.map((tag) => (
                  <Line
                    key={tag}
                    data={pointsByRun[tag] || []}
                    type="monotone"
                    name={shortLabelForTag(tag, (runNumberByTag || {})[tag])}
                    dataKey="psnr"
                    stroke={colorByRunTag[tag] || BASELINE_COLOR}
                    dot={false}
                    strokeWidth={1.75}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

// Scale + opacity histograms from the final PLY of each run. Each run is
// rendered as its own chart (bin edges typically differ run-to-run so
// overlaying on a shared axis doesn't make sense). Paper validation lens:
// - Scale reset should move the scale histogram leftward: more gaussians
//   in the smallest bins, fewer big outliers in the right tail.
// - Entropy constraint polarizes opacity: a tall spike near 0, a tall
//   spike near 1, and a visibly empty middle band.
function HistogramCharts({ runTags, histogramsByRun, showScale, showOpacity, colorByRunTag, runNumberByTag }) {
  return (
    <div className="charts-grid section-spaced">
      <h3 className="section-title">Model distributions (from final PLY)</h3>
      {showScale && (
        <>
          <div className="chart-sub section-hint">
            <strong>Scale distribution.</strong> x-axis is the average size of
            each gaussian (bigger = covers more pixels). A run where scale
            reset is doing its job has most of its mass piled up on the left
            (tiny gaussians) with very little on the right.
          </div>
          <div className="charts-row">
            {runTags.map((tag) => {
              const hist = histogramsByRun[tag]?.scale;
              if (!hist || !hist.counts?.length) return null;
              return (
                <HistogramCard
                  key={`scale-${tag}`}
                  tag={tag}
                  runNumber={(runNumberByTag || {})[tag]}
                  color={colorByRunTag[tag] || BASELINE_COLOR}
                  hist={hist}
                  xLabel="Mean scale"
                />
              );
            })}
          </div>
        </>
      )}
      {showOpacity && (
        <>
          <div className="chart-sub section-hint">
            <strong>Opacity distribution.</strong> x-axis is each gaussian's
            opacity in [0, 1]. A run where the entropy constraint is working
            has a big bump near 0 (transparent), a big bump near 1 (fully
            opaque), and a clear valley in the middle.
          </div>
          <div className="charts-row">
            {runTags.map((tag) => {
              const hist = histogramsByRun[tag]?.opacity;
              if (!hist || !hist.counts?.length) return null;
              return (
                <HistogramCard
                  key={`opacity-${tag}`}
                  tag={tag}
                  runNumber={(runNumberByTag || {})[tag]}
                  color={colorByRunTag[tag] || BASELINE_COLOR}
                  hist={hist}
                  xLabel="Opacity"
                />
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function HistogramCard({ tag, runNumber, color, hist, xLabel }) {
  // Turn parallel bins[]/counts[] arrays into Recharts-shaped bars.
  // XAxis uses type="category" so bars are evenly spaced and the full
  // width is used regardless of how the distribution is shaped. We
  // thin out tick labels so the axis stays readable even with 30+ bins.
  const data = [];
  const bins = hist.bins || [];
  const counts = hist.counts || [];
  for (let i = 0; i < counts.length && i + 1 < bins.length; i++) {
    const mid = (bins[i] + bins[i + 1]) / 2;
    data.push({
      binLabel: formatNumber(mid, 3),
      bin: mid,
      count: counts[i],
      range: `${formatNumber(bins[i], 3)}–${formatNumber(bins[i + 1], 3)}`,
    });
  }
  // Show ~6 tick labels max so they don't collide. First and last are
  // always included so the axis range is obvious.
  const tickTargets = 6;
  const step = Math.max(1, Math.ceil(data.length / tickTargets));
  const visibleTicks = data
    .map((d, i) => (i % step === 0 || i === data.length - 1 ? d.binLabel : null))
    .filter(Boolean);
  const shortLabel = shortLabelForTag(tag, runNumber);
  return (
    <div className="chart-card" style={{ flex: 1 }}>
      <h4 title={tag}>{shortLabel}</h4>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 10, bottom: 18, left: 4 }} barCategoryGap={1}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5ecf6" />
            <XAxis
              dataKey="binLabel"
              type="category"
              interval={0}
              ticks={visibleTicks}
              tick={{ fontSize: 10 }}
              label={{ value: xLabel, position: "insideBottom", offset: -6, fontSize: 10 }}
            />
            <YAxis
              domain={[0, "auto"]}
              tick={{ fontSize: 10 }}
              width={64}
              tickFormatter={formatAxisTick}
            />
            <Tooltip
              content={<HistogramTooltip />}
              cursor={{ fill: "rgba(15, 107, 216, 0.06)" }}
              isAnimationActive={false}
              wrapperStyle={{ pointerEvents: "none" }}
              allowEscapeViewBox={{ x: true, y: true }}
            />
            <Bar dataKey="count" fill={color} isAnimationActive={false}>
              {data.map((_, i) => (
                <Cell key={i} fill={color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StackedCharts({ metrics, runTags, seriesByRun, colorByRunTag, runNumberByTag }) {
  return (
    <div className="charts-grid">
      {runTags.map((tag) => {
        const entry = seriesByRun[tag];
        const records = entry?.records || [];
        const color = colorByRunTag[tag] || BASELINE_COLOR;
        return (
          <div key={tag}>
            <h3 className="stacked-run-title" title={tag}>{shortLabelForTag(tag, (runNumberByTag || {})[tag])}</h3>
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
                      {METRIC_EXPLAINERS[key] && (
                        <div className="chart-explainer">{METRIC_EXPLAINERS[key]}</div>
                      )}
                      <div className="chart-wrap">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={records} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5ecf6" />
                            <XAxis
                              dataKey="iteration"
                              type="number"
                              domain={[0, "dataMax"]}
                              tick={{ fontSize: 11 }}
                              tickFormatter={formatAxisTick}
                            />
                            <YAxis
                              tick={{ fontSize: 11 }}
                              width={72}
                              domain={[0, "auto"]}
                              tickFormatter={formatAxisTick}
                            />
                            <Tooltip
                              content={<ChartTooltip metricKey={key} labelKey="iteration" />}
                              cursor={{ stroke: "#cbd5e1", strokeWidth: 1 }}
                              isAnimationActive={false}
                              wrapperStyle={{ pointerEvents: "none" }}
                              allowEscapeViewBox={{ x: true, y: true }}
                            />
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
