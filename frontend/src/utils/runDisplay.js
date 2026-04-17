// Shared helpers for rendering run_tags as human-friendly labels.
//
// A run_tag looks like one of:
//   <dataset>_<stage>_<YYYYMMDD>_<HHMMSS>                     (old format)
//   <dataset>_<label>_<stage>_<YYYYMMDD>_<HHMMSS>             (new format)
//
// Examples:
//   test123456_train_20260416_160635
//   test123456_s1-shortgs-sr-ent-pr_train_20260416_195114
//
// parseRunTag() pulls the pieces apart so UI code can show a scannable
// "<dataset> — <datetime>" label while preserving the raw tag for API
// paths and file identity.

const STAGES = new Set(["train", "smoke", "validate", "setup"]);

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

// Parse a "YYYYMMDD_HHMMSS" string into a JS Date.
function parseStamp(date, time) {
  if (!/^\d{8}$/.test(date) || !/^\d{6}$/.test(time)) return null;
  const y = Number(date.slice(0, 4));
  const m = Number(date.slice(4, 6)) - 1;
  const d = Number(date.slice(6, 8));
  const h = Number(time.slice(0, 2));
  const mi = Number(time.slice(2, 4));
  const s = Number(time.slice(4, 6));
  const dt = new Date(y, m, d, h, mi, s);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

// Split an unknown run_tag into dataset / label / stage / date / time parts.
// Returns the raw tag as a fallback if the shape isn't recognised.
export function parseRunTag(tag) {
  if (!tag || typeof tag !== "string") {
    return { raw: tag || "", dataset: "", label: "", stage: "", datetime: null };
  }
  const parts = tag.split("_");
  if (parts.length < 4) {
    return { raw: tag, dataset: tag, label: "", stage: "", datetime: null };
  }
  // Last two parts are always the timestamp.
  const time = parts[parts.length - 1];
  const date = parts[parts.length - 2];
  const stageCandidate = parts[parts.length - 3];
  const stage = STAGES.has(stageCandidate) ? stageCandidate : "";
  // Everything before stage (or timestamp if stage unknown) is dataset + optional label
  const headEnd = stage ? parts.length - 3 : parts.length - 2;
  const head = parts.slice(0, headEnd);
  // New format: head is ["dataset", "label"] — label can itself contain hyphens.
  // Old format: head is just ["dataset"].
  let dataset = head[0] || tag;
  let label = "";
  if (head.length > 1) {
    label = head.slice(1).join("_");
  }
  return { raw: tag, dataset, label, stage, datetime: parseStamp(date, time) };
}

// "Apr 16 5:51pm" — short, scannable, no seconds. Falls back to the raw
// stamp if we couldn't parse it.
export function formatRunDatetime(dt) {
  if (!(dt instanceof Date) || Number.isNaN(dt.getTime())) return "";
  const month = MONTHS[dt.getMonth()];
  const day = dt.getDate();
  let hour = dt.getHours();
  const minutes = String(dt.getMinutes()).padStart(2, "0");
  const ampm = hour >= 12 ? "pm" : "am";
  hour = hour % 12 || 12;
  return `${month} ${day} ${hour}:${minutes}${ampm}`;
}

// Primary display name: "<dataset> — Run N" when a run number is known,
// otherwise "<dataset> — Apr 16 5:51pm" as a datetime fallback. If the
// timestamp won't parse we just show the dataset name. Callers that know
// the run's position within its dataset pass runNumber; callers that only
// have the tag can call without it.
export function formatRunDisplay(tag, runNumber) {
  const { dataset, datetime, raw } = parseRunTag(tag);
  if (!dataset) return raw || "run";
  if (typeof runNumber === "number" && runNumber > 0) {
    return `${dataset} — Run ${runNumber}`;
  }
  const when = formatRunDatetime(datetime);
  if (!when) return dataset;
  return `${dataset} — ${when}`;
}

// Given an ordered list of run_tags (sorted by datetime ascending) AND a
// specific run_tag, returns that run's 1-based index within the list.
// Used to render "Run 3" labels that increase chronologically per dataset.
// Caller is responsible for filtering to a single dataset + sorting ascending.
export function computeRunNumberMap(sortedTagsAscending) {
  const out = {};
  sortedTagsAscending.forEach((tag, idx) => {
    out[tag] = idx + 1;
  });
  return out;
}

// Small descriptors used in the tooltip / details card to say what shortgs
// techniques were active.
function describeShortgs(shortgs) {
  if (!shortgs) return [];
  const lines = [];
  const every = Number(shortgs.scale_reset_every || 0);
  if (every > 0) {
    lines.push({
      key: "Scale reset",
      value: `every ${every.toLocaleString()} iters × factor ${shortgs.scale_reset_factor}`,
    });
  }
  const ent = Number(shortgs.entropy_weight || 0);
  if (ent > 0) {
    lines.push({
      key: "Entropy constraint",
      value: `λ = ${ent}`,
    });
  }
  const pr = shortgs.progressive_resolution || "";
  if (pr) {
    lines.push({
      key: "Progressive res",
      value: pr,
    });
  }
  return lines;
}

const NICE_NUMBER = (n, digits = 2) => {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
  const v = Number(n);
  if (!Number.isFinite(v)) return "-";
  if (Math.abs(v) >= 1000) {
    return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return v.toFixed(digits);
};

// Groups the summary into four readable sections for the details card.
// Pipeline (backend/iters/seed) | Preprocessing (FPS/blur/duplicate/etc)
// | Shorter-Splatting (only active techniques) | Results (final metrics).
// Skips fields that are null/empty so empty runs don't display blank rows.
export function summaryGroups(summary) {
  const s = summary || {};

  const pipeline = [];
  if (s.backend) pipeline.push({ key: "Backend", value: s.backend });
  if (s.iterations != null) pipeline.push({ key: "Iterations", value: s.iterations.toLocaleString() });
  if (s.partition) pipeline.push({ key: "Partition", value: s.partition });
  if (s.seed != null && s.seed !== "") pipeline.push({ key: "Seed", value: s.seed });

  // Preprocessing settings are only present if the summary captured them
  // (newer runs write preprocess_report.json separately). We tolerate both
  // shapes - nested `prep` object or flat fields - so older summaries
  // still render something meaningful.
  const preprocessing = [];
  const prep = s.prep || s.preprocess || {};
  const prepGet = (keys) => {
    for (const k of keys) {
      if (prep[k] != null) return prep[k];
      if (s[k] != null) return s[k];
    }
    return null;
  };
  const fps = prepGet(["fps", "target_fps"]);
  const blur = prepGet(["blur_threshold"]);
  const dup = prepGet(["duplicate_threshold"]);
  const ds = prepGet(["downscale"]);
  const mw = prepGet(["max_width", "max_output_width"]);
  if (fps != null) preprocessing.push({ key: "FPS", value: NICE_NUMBER(fps, 1) });
  if (ds != null) preprocessing.push({ key: "Downscale", value: NICE_NUMBER(ds, 2) });
  if (mw != null) preprocessing.push({ key: "Max width", value: NICE_NUMBER(mw, 0) });
  if (blur != null) preprocessing.push({ key: "Blur threshold", value: NICE_NUMBER(blur, 1) });
  if (dup != null) preprocessing.push({ key: "Duplicate threshold", value: NICE_NUMBER(dup, 2) });

  const shortgs = describeShortgs(s.shortgs);

  const results = [];
  if (s.final_psnr != null) results.push({ key: "PSNR", value: `${NICE_NUMBER(s.final_psnr, 2)} dB` });
  if (s.final_ssim != null) results.push({ key: "SSIM", value: NICE_NUMBER(s.final_ssim, 3) });
  if (s.final_lpips != null) results.push({ key: "LPIPS", value: NICE_NUMBER(s.final_lpips, 3) });
  if (s.final_num_gaussians != null) results.push({ key: "Gaussians", value: NICE_NUMBER(s.final_num_gaussians, 0) });
  if (s.final_splats_per_frame != null) results.push({ key: "Splats / frame", value: NICE_NUMBER(s.final_splats_per_frame, 0) });
  if (s.total_wall_seconds != null) results.push({ key: "Wall time", value: `${NICE_NUMBER(s.total_wall_seconds, 0)} s` });

  return { pipeline, preprocessing, shortgs, results };
}
