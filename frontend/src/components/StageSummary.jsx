import React from "react";
import { formatEta } from "../utils/pipelineLog";
import "./StageSummary.css";

// Renders the four-stage pipeline overview (Preprocess / SfM / Training /
// Publish) computed from the raw log stream. Each stage shows an icon,
// label, and a one-line detail; training + SfM show an ETA when active.

const STATUS_GLYPH = {
  pending: "·",
  active: "▶",
  done: "✓",
  skipped: "⤼",
  failed: "!",
};

export default function StageSummary({ stages, trainingLive, sfmEtaSeconds }) {
  return (
    <div className="stage-summary">
      {stages.map((stage) => (
        <StageCard
          key={stage.id}
          stage={stage}
          trainingLive={stage.id === "training" ? trainingLive : null}
          sfmEtaSeconds={stage.id === "sfm" ? sfmEtaSeconds : null}
        />
      ))}
    </div>
  );
}

function StageCard({ stage, trainingLive, sfmEtaSeconds }) {
  const { id, label, status, detail } = stage;
  const cls = `stage-card stage-card--${status}`;

  // Training card: show live iteration + ETA when active.
  let liveLine = null;
  if (id === "training" && status === "active" && trainingLive) {
    const { iteration, total, itsPerSec, psnr, etaSeconds } = trainingLive;
    const parts = [];
    if (iteration != null && total != null) parts.push(`${iteration.toLocaleString()} / ${total.toLocaleString()}`);
    if (itsPerSec != null) parts.push(`${itsPerSec.toFixed(0)} it/s`);
    if (psnr != null) parts.push(`PSNR ${psnr.toFixed(2)}`);
    const eta = formatEta(etaSeconds);
    liveLine = (
      <div className="stage-live">
        <span>{parts.join(" · ")}</span>
        {eta && <span className="stage-eta">~{eta} left</span>}
      </div>
    );
  }

  // SfM card: show rough time-remaining when active.
  let sfmLine = null;
  if (id === "sfm" && status === "active" && sfmEtaSeconds != null) {
    sfmLine = <div className="stage-eta">~{formatEta(sfmEtaSeconds)} (typical)</div>;
  }

  return (
    <div className={cls}>
      <div className="stage-icon" aria-hidden>{STATUS_GLYPH[status] || "·"}</div>
      <div className="stage-body">
        <div className="stage-label-row">
          <span className="stage-label">{label}</span>
          <span className="stage-status">{statusText(status)}</span>
        </div>
        {detail && <div className="stage-detail">{detail}</div>}
        {liveLine}
        {sfmLine}
      </div>
    </div>
  );
}

function statusText(s) {
  switch (s) {
    case "active": return "running";
    case "done": return "done";
    case "skipped": return "cached";
    case "failed": return "failed";
    default: return "waiting";
  }
}
