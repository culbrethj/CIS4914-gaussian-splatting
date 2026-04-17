import React from "react";
import { summaryGroups, parseRunTag, formatRunDatetime } from "../utils/runDisplay";
import "./RunDetailsCard.css";

// Structured details card for a training run. Shows, top to bottom:
//   * header: "dataset · Run N" + backend chip + date/time
//   * Pipeline (backend, iterations, seed, partition)
//   * Preprocessing (fps, downscale, blur/duplicate thresholds, max width)
//   * Shorter-Splatting (only the techniques that were actually active)
//   * Results (final PSNR / SSIM / LPIPS / gaussians / wall time)
//   * raw internal run_tag at the bottom (copy-pasteable id, never the
//     primary label in the UI)
//
// Props:
//   summary    - the run's metrics_summary.json object (may be partial)
//   runTag     - full run_tag string; shown in the footer only
//   runNumber  - optional 1-based number within its dataset; if present
//                the header reads "Run N" instead of datetime
//   variant    - "popover" (floating, compact) | "inline" (block, roomier)

export default function RunDetailsCard({ summary, runTag, runNumber, variant = "popover" }) {
  const { pipeline, preprocessing, shortgs, results } = summaryGroups(summary);
  const { dataset, datetime } = parseRunTag(runTag);
  const when = formatRunDatetime(datetime);

  return (
    <div className={`run-details-card run-details-card--${variant}`} role="group" aria-label="Run details">
      <div className="run-details-head">
        <div className="run-details-head-main">
          {dataset && <span className="run-details-dataset">{dataset}</span>}
          {typeof runNumber === "number" && runNumber > 0 && (
            <span className="run-details-run-chip">Run {runNumber}</span>
          )}
        </div>
        {summary?.backend && (
          <span className="run-details-backend-chip">{summary.backend}</span>
        )}
      </div>
      {when && <div className="run-details-when">{when}</div>}

      {pipeline.length > 0 && (
        <Section title="Pipeline">
          {pipeline.map((row) => (
            <Row key={row.key} label={row.key} value={row.value} />
          ))}
        </Section>
      )}

      {preprocessing.length > 0 && (
        <Section title="Preprocessing">
          {preprocessing.map((row) => (
            <Row key={row.key} label={row.key} value={row.value} />
          ))}
        </Section>
      )}

      <Section title="Shorter-Splatting">
        {shortgs.length > 0 ? (
          shortgs.map((row) => (
            <Row key={row.key} label={row.key} value={row.value} />
          ))
        ) : (
          <div className="run-details-empty">No techniques active (baseline run)</div>
        )}
      </Section>

      {results.length > 0 && (
        <Section title="Results">
          {results.map((row) => (
            <Row key={row.key} label={row.key} value={row.value} highlight />
          ))}
        </Section>
      )}

      {runTag && (
        <div className="run-details-tag">
          <span className="run-details-tag-label">id</span>
          <code className="run-details-tag-value" title={runTag}>{runTag}</code>
        </div>
      )}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="run-details-section">
      <div className="run-details-section-title">{title}</div>
      <div className="run-details-section-body">{children}</div>
    </div>
  );
}

function Row({ label, value, highlight }) {
  return (
    <div className={`run-details-row${highlight ? " is-highlight" : ""}`}>
      <span className="run-details-row-key">{label}</span>
      <span className="run-details-row-val">{value}</span>
    </div>
  );
}
