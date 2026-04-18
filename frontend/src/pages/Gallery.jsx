import React, { useEffect, useMemo, useState } from "react";
import GaussViewer from "../components/GaussViewer";
import SideBySideViewer from "../components/SideBySideViewer";
import "./Gallery.css";

// Gallery uses a two-level selector: pick a dataset, then pick a run
// within that dataset. Single-run datasets skip the run picker.
// Compare mode shows two runs from the SAME dataset side-by-side — the
// only comparison that actually tells you something about training
// settings. Runs come from GET /api/datasets/{name}/runs which groups
// per-run splat files with their metrics so the dropdown can show
// "Run 1", "Run 2", etc. rather than raw filenames.

function displayRunLabel(run) {
  if (!run) return "";
  if (run.is_showcase) return "Showcase";
  return run.display_label || `Run ${run.run_number || "?"}`;
}

export default function Gallery() {
  const [datasets, setDatasets] = useState([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);

  // Scene A: dataset name + run_tag (null run_tag = showcase).
  const [sceneADataset, setSceneADataset] = useState("");
  const [sceneARunTag, setSceneARunTag] = useState("");
  // Scene B only has a run picker — its dataset follows Scene A's.
  const [sceneBRunTag, setSceneBRunTag] = useState("");

  const [compareMode, setCompareMode] = useState(false);
  const [matchCamera, setMatchCamera] = useState(false);

  // Runs keyed by dataset so we don't re-fetch when flipping between
  // scenes in the same dataset.
  const [runsByDataset, setRunsByDataset] = useState({});

  useEffect(() => {
    setDatasetsLoading(true);
    fetch("/api/datasets")
      .then((r) => r.json())
      .then((list) => {
        // Only datasets that have something viewable: either a root
        // splat (showcase) or at least one training run.
        const usable = (list || []).filter((d) => d.has_splat || (d.run_count || 0) > 0);
        setDatasets(usable);
      })
      .catch(() => setDatasets([]))
      .finally(() => setDatasetsLoading(false));
  }, []);

  // When the user picks a dataset, fetch its runs if we haven't yet.
  useEffect(() => {
    if (!sceneADataset || runsByDataset[sceneADataset]) return;
    let cancelled = false;
    fetch(`/api/datasets/${encodeURIComponent(sceneADataset)}/runs`)
      .then((r) => (r.ok ? r.json() : []))
      .then((runs) => {
        if (cancelled) return;
        setRunsByDataset((prev) => ({ ...prev, [sceneADataset]: runs || [] }));
      })
      .catch(() => {
        if (!cancelled) setRunsByDataset((prev) => ({ ...prev, [sceneADataset]: [] }));
      });
    return () => { cancelled = true; };
  }, [sceneADataset, runsByDataset]);

  const sceneARuns = useMemo(() => {
    const all = runsByDataset[sceneADataset] || [];
    // Only expose runs that actually have a splat on disk.
    return all.filter((r) => r.splat_path);
  }, [runsByDataset, sceneADataset]);

  // Single-run datasets auto-select their one run so the user doesn't
  // have to click through a dropdown with one option.
  useEffect(() => {
    if (!sceneADataset) {
      setSceneARunTag("");
      return;
    }
    if (sceneARuns.length === 1) {
      const only = sceneARuns[0];
      setSceneARunTag(only.run_tag || "__showcase__");
    } else if (sceneARuns.length > 1) {
      // Multi-run dataset: clear any stale selection that doesn't belong
      // to the current dataset.
      const still = sceneARuns.some((r) => (r.run_tag || "__showcase__") === sceneARunTag);
      if (!still) setSceneARunTag("");
    }
  }, [sceneADataset, sceneARuns, sceneARunTag]);

  const sceneARun = useMemo(
    () => sceneARuns.find((r) => (r.run_tag || "__showcase__") === sceneARunTag) || null,
    [sceneARuns, sceneARunTag],
  );

  // Scene B: only meaningful within Scene A's dataset, excluding Scene A.
  const sceneBOptions = useMemo(() => {
    if (!sceneARun) return [];
    return sceneARuns.filter((r) => (r.run_tag || "__showcase__") !== sceneARunTag);
  }, [sceneARun, sceneARuns, sceneARunTag]);

  const canCompare = sceneBOptions.length > 0;

  // Scene B auto-reset when Scene A changes dataset/run.
  useEffect(() => {
    if (!compareMode) return;
    if (!sceneBRunTag) return;
    if (!sceneBOptions.some((r) => (r.run_tag || "__showcase__") === sceneBRunTag)) {
      setSceneBRunTag("");
    }
  }, [sceneARunTag, sceneADataset, compareMode, sceneBOptions, sceneBRunTag]);

  const sceneBRun = useMemo(
    () => sceneBOptions.find((r) => (r.run_tag || "__showcase__") === sceneBRunTag) || null,
    [sceneBOptions, sceneBRunTag],
  );

  function onToggleCompare() {
    const next = !compareMode;
    setCompareMode(next);
    if (next) {
      if (!sceneBRunTag && sceneBOptions.length > 0) {
        setSceneBRunTag(sceneBOptions[0].run_tag || "__showcase__");
      }
    } else {
      setMatchCamera(false);
    }
  }

  const showSceneARunPicker = sceneADataset && sceneARuns.length > 1;
  const splatAPath = sceneARun?.splat_path || "";
  const splatBPath = sceneBRun?.splat_path || "";
  const sceneALabel = sceneARun
    ? `${sceneADataset} — ${displayRunLabel(sceneARun)}`
    : "Scene A";
  const sceneBLabel = sceneBRun
    ? `${sceneADataset} — ${displayRunLabel(sceneBRun)}`
    : "Scene B";

  return (
    <main className="gallery-page">
      <header className="gallery-head">
        <h2>Gallery</h2>
        <p className="gallery-sub">
          Pick a dataset, then a run within it. Turn on Compare to view two
          runs from the same dataset side-by-side.
        </p>
      </header>

      {datasets.length === 0 ? (
        <div className="gallery-empty">
          {datasetsLoading
            ? "Loading datasets..."
            : "No datasets yet. If you expect datasets here, check the backend health banner at the top of the page. The backend may not be running."}
        </div>
      ) : (
        <section className="gallery-controls">
          <label className="gallery-picker">
            <span className="gallery-picker-label">Dataset</span>
            <select
              value={sceneADataset}
              onChange={(e) => {
                setSceneADataset(e.target.value);
                setSceneARunTag("");
                setSceneBRunTag("");
              }}
            >
              <option value="">-- select a dataset --</option>
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name}
                  {d.run_count ? ` · ${d.run_count} run${d.run_count === 1 ? "" : "s"}` : ""}
                </option>
              ))}
            </select>
          </label>

          {showSceneARunPicker && (
            <label className="gallery-picker">
              <span className="gallery-picker-label">Run</span>
              <select
                value={sceneARunTag}
                onChange={(e) => setSceneARunTag(e.target.value)}
              >
                <option value="">-- select a run --</option>
                {sceneARuns.map((r) => (
                  <option
                    key={r.run_tag || "__showcase__"}
                    value={r.run_tag || "__showcase__"}
                    title={r.run_tag || r.splat_filename || ""}
                  >
                    {displayRunLabel(r)}
                  </option>
                ))}
              </select>
            </label>
          )}

          <div className="gallery-actions">
            <button
              type="button"
              className={`gallery-btn${compareMode ? " is-active" : ""}`}
              onClick={onToggleCompare}
              disabled={!canCompare}
              title={
                canCompare
                  ? "Compare two runs from the same dataset"
                  : sceneARun
                    ? "This dataset only has one run; need at least two"
                    : "Pick a dataset + run first"
              }
            >
              {compareMode ? "Exit Compare" : "Compare"}
            </button>
            {compareMode && (
              <button
                type="button"
                className={`gallery-btn${matchCamera ? " is-active" : ""}`}
                onClick={() => setMatchCamera((v) => !v)}
                title="Lock the right viewer to follow the left viewer's camera"
              >
                {matchCamera ? "Unlock Camera" : "Match Camera"}
              </button>
            )}
          </div>

          {compareMode && (
            <label className="gallery-picker">
              <span className="gallery-picker-label">
                Scene B <span className="gallery-picker-hint">(same dataset)</span>
              </span>
              <select
                value={sceneBRunTag}
                onChange={(e) => setSceneBRunTag(e.target.value)}
              >
                <option value="">-- select a second run --</option>
                {sceneBOptions.map((r) => (
                  <option
                    key={r.run_tag || "__showcase__"}
                    value={r.run_tag || "__showcase__"}
                    title={r.run_tag || r.splat_filename || ""}
                  >
                    {displayRunLabel(r)}
                  </option>
                ))}
              </select>
            </label>
          )}
        </section>
      )}

      {datasets.length > 0 && (
        <section className="gallery-viewer">
          {compareMode ? (
            <SideBySideViewer
              leftApiPath={splatAPath}
              rightApiPath={splatBPath}
              leftLabel={sceneALabel}
              rightLabel={sceneBLabel}
              matchCamera={matchCamera}
            />
          ) : (
            <GaussViewer splatApiPath={splatAPath || null} />
          )}
        </section>
      )}
    </main>
  );
}
