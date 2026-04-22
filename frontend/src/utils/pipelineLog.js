// Parses the flat WebSocket log stream from the pipeline subprocess into a
// structured snapshot of what each stage is doing. The parser runs over
// all logs on every update so it's a stateless pure function — no React
// state drift, no partial updates to reconcile.
//
// Returned shape:
//   {
//     stages: [{ id, label, status, detail }],
//     trainingLive: { iteration, total, itsPerSec, loss, psnr, etaSeconds } | null,
//     sfmEtaSeconds: number | null,  // rough time-remaining estimate when SfM is active
//   }

export const STAGE_ORDER = ["preprocess", "sfm", "training", "publish"];

const STAGE_LABELS = {
  preprocess: "Preprocess",
  sfm: "SfM + undistort",
  training: "Training",
  publish: "Publish",
};

// Rough static fallbacks used when we have no history. Actual ETA for
// training is computed from the running it/s rate. SfM is tough to predict
// so we just say "~8 min" when we see it start and clear it when we see
// it end.
const SFM_ETA_DEFAULT_SECONDS = 8 * 60;

function blank(id) {
  return { id, label: STAGE_LABELS[id] || id, status: "pending", detail: "" };
}

// tqdm progress line: "...| 1000/10000 [00:20<03:30, 48.19it/s, Loss=0.045, Depth Loss=0.000]"
// Also catches the shorter form without ETA on very first lines.
const TRAIN_PROGRESS_RE = /(\d+)\s*\/\s*(\d+)\s*\[[^\]]*?(\d+(?:\.\d+)?)it\/s[^\]]*?Loss\s*[:=]\s*([0-9.eE+-]+)/;
// Eval line: "[ITER 7000] Evaluating train: L1 0.017 PSNR 32.26 [...]"
const EVAL_RE = /\[ITER\s+(\d+)\].*?PSNR\s+([0-9.eE+-]+)/i;
// pipeline.py finished-prepare line: "Preprocessing finished (138/200 frames kept in 12.34s)"
const PREP_FINISHED_RE = /Preprocessing finished \((\d+)\/(\d+) frames kept in/;
// pipeline.py prep cache hit: "Preprocessing cache hit — reusing 138 images"
const PREP_CACHE_RE = /Preprocessing cache hit.*reusing\s+(\d+)\s+images/i;

export function parsePipelineLog(logs) {
  const stages = {
    preprocess: blank("preprocess"),
    sfm: blank("sfm"),
    training: blank("training"),
    publish: blank("publish"),
  };

  let trainingLive = null;
  let sfmEtaSeconds = null;
  let sawSfmStart = false;
  let sawSfmEnd = false;

  // Walk every logged line and update stage state. Latest-wins: later
  // lines about the same stage overwrite earlier ones.
  for (const entry of logs) {
    const line = (entry && entry.text) || "";
    if (!line) continue;
    const lower = line.toLowerCase();

    // --- preprocess ---
    if (lower.includes("starting preprocessing") || lower.includes("starting video slicing")) {
      if (stages.preprocess.status === "pending") {
        stages.preprocess.status = "active";
        stages.preprocess.detail = "Extracting + filtering frames";
      }
    }
    const mCache = line.match(PREP_CACHE_RE);
    if (mCache) {
      stages.preprocess.status = "skipped";
      stages.preprocess.detail = `Cached — reusing ${mCache[1]} frames`;
    }
    const mPrep = line.match(PREP_FINISHED_RE);
    if (mPrep) {
      const kept = Number(mPrep[1]);
      const total = Number(mPrep[2]);
      stages.preprocess.status = "done";
      stages.preprocess.detail = `${kept} of ${total} frames passed filters`;
    }

    // --- sfm ---
    if (lower.includes("sfm cache hit")) {
      stages.sfm.status = "skipped";
      stages.sfm.detail = "Cached — reusing existing sparse model";
      sawSfmEnd = true;
    } else if (lower.includes("skipping image upload and hpg sfm")) {
      stages.sfm.status = "skipped";
      stages.sfm.detail = "Cached — skipped image upload and remote SfM";
      sawSfmEnd = true;
    } else if (lower.includes("starting vggt sfm step")) {
      // VGGT path: feed-forward transformer on a GPU, typically under
      // a minute of wall time (a few minutes with bundle adjustment).
      // This line wins over the neutral "uploading" detail below.
      if (!sawSfmEnd) {
        stages.sfm.status = "active";
        stages.sfm.detail = "Running VGGT on HPG (feed-forward transformer, GPU)";
        sawSfmStart = true;
        sfmEtaSeconds = 180; // ~3 min with BA on; ~1 min without
      }
    } else if (lower.includes("starting sfm step on hipergator")) {
      // COLMAP-specific line from the fastergs path.
      if (!sawSfmEnd) {
        stages.sfm.status = "active";
        stages.sfm.detail = "Running COLMAP on HPG (queue + SIFT + matching + mapping + undistort)";
        sawSfmStart = true;
        sfmEtaSeconds = SFM_ETA_DEFAULT_SECONDS;
      }
    } else if (lower.includes("starting faster-gs sync")) {
      // Sync fires before the SfM method branches, so keep the label
      // method-agnostic until "starting vggt/colmap sfm" arrives and
      // overwrites it.
      if (!sawSfmEnd) {
        stages.sfm.status = "active";
        stages.sfm.detail = "Uploading preprocessed images to HPG";
        sawSfmStart = true;
        sfmEtaSeconds = SFM_ETA_DEFAULT_SECONDS;
      }
    } else if (lower.includes("starting sfm step")) {
      // Generic opensplat SfM line (single-machine COLMAP).
      if (!sawSfmEnd) {
        stages.sfm.status = "active";
        stages.sfm.detail = "Running COLMAP locally";
        sawSfmStart = true;
        sfmEtaSeconds = SFM_ETA_DEFAULT_SECONDS;
      }
    } else if (
      lower.includes("sfm step finished successfully") ||
      lower.includes("gs_final prepare complete") ||
      lower.includes("[ok] gs_final prepare complete") ||
      lower.includes("[ok] vggt sfm stage completed")
    ) {
      stages.sfm.status = "done";
      stages.sfm.detail = stages.sfm.detail || "Sparse model ready";
      sawSfmEnd = true;
      sfmEtaSeconds = null;
    }

    // --- training ---
    const mEval = line.match(EVAL_RE);
    if (mEval) {
      if (stages.training.status === "pending") stages.training.status = "active";
      const iter = Number(mEval[1]);
      const psnr = Number(mEval[2]);
      trainingLive = {
        ...(trainingLive || {}),
        iteration: iter,
        psnr,
      };
    }
    const mTrain = line.match(TRAIN_PROGRESS_RE);
    if (mTrain) {
      if (stages.training.status === "pending") stages.training.status = "active";
      const iter = Number(mTrain[1]);
      const total = Number(mTrain[2]);
      const rate = Number(mTrain[3]);
      const loss = Number(mTrain[4]);
      const remaining = Math.max(0, total - iter);
      const eta = rate > 0 ? remaining / rate : null;
      trainingLive = {
        iteration: iter,
        total,
        itsPerSec: rate,
        loss,
        psnr: trainingLive?.psnr ?? null,
        etaSeconds: eta,
      };
      stages.training.detail = `iter ${iter.toLocaleString()} / ${total.toLocaleString()}`;
    }
    if (
      lower.includes("training complete") ||
      lower.includes("gs_final train stage completed") ||
      lower.includes("[ok] gs_final train stage completed")
    ) {
      stages.training.status = "done";
      if (trainingLive?.iteration && trainingLive?.total) {
        stages.training.detail = `Completed ${trainingLive.total.toLocaleString()} iterations`;
      } else {
        stages.training.detail = "Training finished";
      }
      trainingLive = null;
    }

    // --- publish / post-processing ---
    if (lower.includes("final .splat published")) {
      stages.publish.status = "done";
      stages.publish.detail = "Splat copied to viewer + gallery";
    } else if (
      lower.includes("rendering metrics pngs") ||
      lower.includes("metrics_plotter] wrote")
    ) {
      if (stages.publish.status === "pending") {
        stages.publish.status = "active";
        stages.publish.detail = "Rendering metrics charts";
      }
    } else if (lower.includes("converter did not create expected splat")) {
      stages.publish.status = "failed";
      stages.publish.detail = "PLY → splat conversion failed";
    }

    // --- top-level failure ---
    const trimmed = line.trim();
    if (
      trimmed.startsWith("ERROR:") ||
      trimmed.startsWith("[error]") ||
      trimmed.includes("<<ERROR:") ||
      trimmed.match(/<<DONE:\s*[1-9]/)
    ) {
      // Mark whichever stage is currently active as failed with a useful
      // message, so the card for the in-flight step shows the failure.
      for (const id of STAGE_ORDER) {
        if (stages[id].status === "active") {
          stages[id].status = "failed";
          stages[id].detail = trimmed;
          break;
        }
      }
    }
  }

  // If SfM is skipped and training is active, the overall flow looks like
  // "SfM skipped (cached) → Training..." which we indicate by giving the
  // SfM card a skipped status but keeping it visible.
  if (stages.sfm.status === "pending" && stages.training.status !== "pending") {
    // Prepare done + training running without seeing explicit SfM events
    // (e.g. when OpenSplat path blurs SfM and training into one step):
    // mark SfM done implicitly so the timeline doesn't look stuck.
    stages.sfm.status = "done";
    stages.sfm.detail = "Completed";
  }

  // If publish is still pending but training finished, we're probably
  // mid-conversion. Keep it active so the user sees something's happening.
  if (stages.training.status === "done" && stages.publish.status === "pending") {
    stages.publish.status = "active";
    stages.publish.detail = stages.publish.detail || "Converting PLY → splat";
  }

  return {
    stages: STAGE_ORDER.map((id) => stages[id]),
    trainingLive,
    sfmEtaSeconds:
      stages.sfm.status === "active" && sawSfmStart && !sawSfmEnd
        ? sfmEtaSeconds
        : null,
  };
}

// "3 min", "12 sec", "1h 5m" — rough, no faked precision.
export function formatEta(seconds) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "";
  const s = Math.round(seconds);
  if (s < 60) return `${s} sec`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem ? `${h}h ${rem}m` : `${h}h`;
}
