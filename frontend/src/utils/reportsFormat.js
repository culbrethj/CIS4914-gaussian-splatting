// Number formatters shared by Reports.jsx charts, tables, and tooltips.
//
// formatMetric is metric-aware: given a metric key (psnr, ssim, lpips,
// loss, num_gaussians, wall_seconds, ...) it formats the value in the
// unit users expect. In axis mode it returns a short form suitable for
// Recharts tickFormatter callbacks.
//
// formatNumber is the metric-agnostic fallback: thousands-separator
// commas for big numbers, fixed digits for small ones.

export function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 1000) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return n.toFixed(digits);
}

export function formatMetric(value, metric, { axis = false } = {}) {
  if (value === null || value === undefined) return axis ? "" : "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return axis ? "" : "-";

  if (axis) {
    if (Math.abs(n) >= 1000) {
      return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
    if (Math.abs(n) >= 10) return n.toFixed(1);
    return n.toFixed(3);
  }

  switch (metric) {
    case "psnr":
      return `${n.toFixed(2)} dB`;
    case "ssim":
    case "lpips":
      return n.toFixed(3);
    case "loss":
      return n.toFixed(4);
    case "num_gaussians":
    case "splats_per_frame":
      return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    case "wall_seconds": {
      const secs = Math.round(n);
      if (secs < 60) return `${secs}s`;
      const m = Math.floor(secs / 60);
      const s = secs % 60;
      return s === 0 ? `${m}m` : `${m}m ${s}s`;
    }
    default:
      if (Math.abs(n) >= 1000) {
        return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
      }
      return n.toFixed(3);
  }
}
