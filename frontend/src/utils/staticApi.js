// Fetch shim for the VITE_STATIC_MODE GitHub Pages build. Monkey-patches
// window.fetch so the existing `fetch("/api/datasets/...")` call sites in
// Gallery, Reports, GaussViewer, etc. keep working without refactor.
//
// Every supported endpoint is synthesized from backend/datasets/ assets that
// were mirrored into the static bundle at build time (see vite.config.js
// copyShowcasePlugin + config/showcase.js).
//
// Anything we don't explicitly intercept falls through to the real fetch,
// which in static mode will simply fail - intended for the endpoints the
// Live Demos page already guards against with its static banner.

import { SHOWCASE_DATASETS } from "../config/showcase.js";

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// BASE_URL is the Vite base the site is served under (e.g.
// "/CIS4914-gaussian-splatting/" on GitHub Pages, "/" locally). All static
// asset URLs need to be resolved against it.
function withBase(path) {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

// Build the /api/datasets response. The Gallery filters this by run_count>0
// and the Reports page reads only .name, so we set the fields the UI
// actually uses; the rest are here to preserve shape compatibility.
function buildDatasetList() {
  return SHOWCASE_DATASETS.map((d) => ({
    name: d.name,
    has_splat: true,
    splat_path: `/api/datasets/${d.name}/splat`,
    has_images: true,
    has_raw_video: false,
    has_sfm: true,
    run_count: 1,
  }));
}

// Build the /api/datasets/{name}/runs response used by the Gallery picker.
// One real run per dataset, pointed at the dataset-root splat (which is what
// a fresh clone ships with - the hipergator/gs_final/ path is gitignored).
function buildRunsList(name) {
  const entry = SHOWCASE_DATASETS.find((d) => d.name === name);
  if (!entry) return [];
  return [
    {
      run_tag: entry.run_tag,
      splat_path: `/api/datasets/${name}/splat`,
      splat_filename: "splat.splat",
      metrics_summary: null,
      created_at_epoch: 0,
      is_showcase: false,
      run_number: 1,
      display_label: "Run 1",
    },
  ];
}

// Fetch the committed metrics_summary.json for a dataset's run from the
// static bundle. Used by /api/datasets/{name}/metrics to fill the Reports
// page's RunPickerCard + summary table.
async function loadSummary(origFetch, datasetName, runTag) {
  const path = withBase(
    `/datasets/${encodeURIComponent(datasetName)}/metrics/${encodeURIComponent(runTag)}/metrics_summary.json`
  );
  try {
    const res = await origFetch(path);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// Fetch and parse the committed metrics.jsonl time-series for a run. The
// backend returns a JSON array; we keep that shape.
async function loadSeries(origFetch, datasetName, runTag) {
  const path = withBase(
    `/datasets/${encodeURIComponent(datasetName)}/metrics/${encodeURIComponent(runTag)}/metrics.jsonl`
  );
  try {
    const res = await origFetch(path);
    if (!res.ok) return [];
    const text = await res.text();
    const records = [];
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        records.push(JSON.parse(trimmed));
      } catch {
        // skip malformed line; matches backend's "return what parses" spirit
      }
    }
    return records;
  } catch {
    return [];
  }
}

export function installStaticApi() {
  if (typeof window === "undefined" || !window.fetch) return;
  const origFetch = window.fetch.bind(window);

  window.fetch = async (input, init) => {
    const rawUrl = typeof input === "string" ? input : input?.url || "";
    // Only intercept same-origin API paths. Any absolute URL (CDN, gsplat
    // example hosts, etc.) skips the shim.
    const isApi = rawUrl.startsWith("/api/");
    const isDatasetsMount = rawUrl.startsWith("/datasets/");

    if (!isApi && !isDatasetsMount) {
      return origFetch(input, init);
    }

    // /api/health: report healthy so BackendHealthBanner stays hidden.
    if (rawUrl.startsWith("/api/health")) {
      return json({ ok: true, mode: "static" });
    }

    // /api/datasets
    if (rawUrl === "/api/datasets" || rawUrl.startsWith("/api/datasets?")) {
      return json(buildDatasetList());
    }

    // /api/datasets/{name}/runs
    let m = rawUrl.match(/^\/api\/datasets\/([^/?]+)\/runs(?:\?.*)?$/);
    if (m) {
      return json(buildRunsList(decodeURIComponent(m[1])));
    }

    // /api/datasets/{name}/metrics (list of runs with their summaries)
    m = rawUrl.match(/^\/api\/datasets\/([^/?]+)\/metrics(?:\?.*)?$/);
    if (m) {
      const name = decodeURIComponent(m[1]);
      const entry = SHOWCASE_DATASETS.find((d) => d.name === name);
      if (!entry) return json([]);
      const summary = await loadSummary(origFetch, name, entry.run_tag);
      return json([
        {
          run_tag: entry.run_tag,
          summary,
          has_series: true,
        },
      ]);
    }

    // /api/datasets/{name}/metrics/{tag}/series
    m = rawUrl.match(/^\/api\/datasets\/([^/?]+)\/metrics\/([^/?]+)\/series(?:\?.*)?$/);
    if (m) {
      const name = decodeURIComponent(m[1]);
      const tag = decodeURIComponent(m[2]);
      const records = await loadSeries(origFetch, name, tag);
      return json(records);
    }

    // /api/datasets/{name}/metrics/{tag}/summary
    m = rawUrl.match(/^\/api\/datasets\/([^/?]+)\/metrics\/([^/?]+)\/summary(?:\?.*)?$/);
    if (m) {
      const name = decodeURIComponent(m[1]);
      const tag = decodeURIComponent(m[2]);
      const summary = await loadSummary(origFetch, name, tag);
      return summary ? json(summary) : json({}, 404);
    }

    // /api/datasets/{name}/splat  -> static /datasets/{name}/splat.splat
    m = rawUrl.match(/^\/api\/datasets\/([^/?]+)\/splat(?:\?.*)?$/);
    if (m) {
      const name = decodeURIComponent(m[1]);
      return origFetch(withBase(`/datasets/${encodeURIComponent(name)}/splat.splat`), init);
    }

    // Pre-existing /datasets/... paths (the backend's StaticFiles mount) need
    // the GitHub Pages base prefix. Rewrite and forward.
    if (isDatasetsMount) {
      return origFetch(withBase(rawUrl), init);
    }

    // Unsupported /api/... call. Return a 503 with a clear body so callers
    // see a consistent "backend off" signal instead of a raw network error.
    return json(
      { detail: "backend disabled in static build", mode: "static" },
      503
    );
  };
}
