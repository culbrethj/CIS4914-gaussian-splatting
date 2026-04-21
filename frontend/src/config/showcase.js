// Showcase datasets for the GitHub Pages static build (VITE_STATIC_MODE=true).
// Only used by utils/staticApi.js to synthesize the /api/datasets/* responses
// the frontend otherwise fetches from the FastAPI backend. Local dev and any
// deploy with a real backend ignore this file entirely.
//
// Each entry's shape matches what the backend currently returns from
// /api/datasets and /api/datasets/{name}/runs and /api/datasets/{name}/metrics,
// so existing page code (Gallery, Reports, GaussViewer) works unchanged.
//
// To publish a new showcase scene:
//   1. Commit its splat.splat + metrics/<run_tag>/ under backend/datasets/.
//   2. Add an entry below with the folder name + run_tag.
//   3. The build copies the matching files into the static bundle.

export const SHOWCASE_DATASETS = [
  {
    name: "cone",
    run_tag: "cone_colmap_c9-vanilla_train_20260418_000034",
  },
  {
    name: "snowboard",
    run_tag: "snowboard_colmap_s1-baseline_train_20260420_181230",
  },
  {
    name: "gator_statue",
    run_tag: "gator_statue_s1-baseline_train_20260421_055348",
  },
];

export function isStaticMode() {
  return import.meta.env.VITE_STATIC_MODE === "true";
}
