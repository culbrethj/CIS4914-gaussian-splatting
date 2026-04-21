# Faster-GS backend notes

This is the other training backend in the project, next to OpenSplat. It's
what `backend="fastergs"` selects on the Live Demos page. This README covers
what it is, why the pipeline looks the way it does, and how to get it
running if you're picking the project up fresh.

## What Faster-GS is

Faster-GS is a set of optimizations on top of the original Inria
gaussian-splatting trainer (faster rasterizer, faster Adam, a custom CUDA
backend). We use the fork at
[fhahlbohm/gaussian-splatting](https://github.com/fhahlbohm/gaussian-splatting)
which pulls in the Faster-GS techniques.

We picked it because:

- OpenSplat's local binary has been flaky across machines: missing dylibs,
  runtime segfaults. Faster-GS on HPG is where we actually get reliable
  training runs.
- It produces standard `point_cloud.ply` output, which plugs straight into
  our existing PLY→SPLAT converter and viewer.
- It's actively maintained, and the fork is close enough to the canonical
  Inria code that the camera model and dataset layout match COLMAP's
  convention.

## Why this runs on HiPerGator, not locally

Training needs a GPU with CUDA. Most of our dev machines don't have one. HPG
has an account for this class (`cis4914`) with GPU partitions, so we push
the compute-heavy steps there and keep the lightweight bits (video
preprocessing, file publishing) local.

The split:

| Step                       | Where |
|---------------------------|-------|
| Frame extraction + filtering | Local |
| Upload cleaned images to HPG | Local (`rsync` over SSH) |
| SfM (COLMAP or VGGT)       | HPG (SLURM; CPU for COLMAP, GPU for VGGT) |
| Faster-GS training         | HPG (SLURM GPU job) |
| PLY → SPLAT                | HPG (right after training) |
| Fetch result + publish     | Local |

## SfM methods

Two SfM backends are selectable in the Advanced Settings panel and via
the `--sfm-method` CLI flag:

- **COLMAP (default)**: traditional feature-extraction + sequential
  matching + incremental mapping + `image_undistorter`, all on CPU.
  Typical wall time: 8–12 minutes. Reliable, well-understood, no extra
  env setup needed.
- **VGGT**: [Visual Geometry Grounded Transformer](https://github.com/facebookresearch/vggt)
  (CVPR 2025 Best Paper, Facebook Research). Feed-forward transformer
  that infers camera intrinsics/extrinsics + point cloud from the
  image set in one GPU pass. Typical wall time: under 1 minute (plus
  queue wait on `hpg-b200`). Output cameras are PINHOLE-family so no
  undistort step is needed afterwards.

Pick COLMAP when you want the familiar pipeline. Pick VGGT when you want
to iterate fast on training settings. A full run (prepare → SfM →
training) drops from ~25 min to ~8 min on our 48-image smoke set.

VGGT bundle adjustment is ON by default (`--vggt-use-ba`). Without BA,
VGGT's demo writes zero 3D points into `points3D.bin` and the trainer
has nothing to seed gaussians from. BA takes ~3 min on a B200 vs ~45 s
for the raw feed-forward output. The `--no-vggt-ba` flag is there for
timing comparisons, not for actual training.

### VGGT: one-time HPG setup

Commands below use `$USER` as your gatorlink. Export `FASTERGS_REMOTE_ROOT`
once before running them so paths resolve to your own workspace:

```bash
export FASTERGS_REMOTE_ROOT=/blue/cis4914/$USER/gs_final
```

Only needed once per remote workspace (`$FASTERGS_REMOTE_ROOT`):

```bash
# Clone the VGGT repo into $REMOTE_ROOT/src/
ssh hpg 'cd /blue/cis4914/$USER/gs_final/src && git clone --depth=1 https://github.com/facebookresearch/vggt.git'

# Install vggt + its extra deps into the fastergs conda env. We pass
# --no-deps on vggt itself because its pyproject pins torch==2.3.1 which
# would downgrade our CUDA 12.8 build; vggt works fine with newer torch.
# pycolmap is pinned to 3.x because VGGT's demo_colmap.py builds
# pycolmap.Image(cam_from_world=...) and pycolmap 4.x made that attribute
# read-only (it became a method), which can't be patched around.
ssh hpg 'bash -lc "source /apps/conda/25.7.0/etc/profile.d/conda.sh && \
  conda activate /blue/cis4914/$USER/gs_final/envs/fastergs_cuda128 && \
  pip install --no-deps /blue/cis4914/$USER/gs_final/src/vggt && \
  pip install einops safetensors huggingface_hub trimesh \"pycolmap==3.10.0\" hydra-core omegaconf scipy tqdm && \
  pip install git+https://github.com/cvg/LightGlue"'

# Prime the VGGT-1B weight cache. Takes ~1 min; subsequent jobs hit the cache.
ssh hpg 'bash -lc "source /apps/conda/25.7.0/etc/profile.d/conda.sh && \
  conda activate /blue/cis4914/$USER/gs_final/envs/fastergs_cuda128 && \
  export HF_HOME=/blue/cis4914/$USER/gs_final/models/hf && \
  python -c \"from vggt.models.vggt import VGGT; VGGT.from_pretrained(\\\"facebook/VGGT-1B\\\"); print(\\\"ok\\\")\""'
```

Everything goes under `$REMOTE_ROOT` so teammates overriding that env
var get a parallel workspace automatically.

## Prerequisites

Before any of this works you need:

1. **HPG account on the `cis4914` SLURM account.** Ask the instructor if
   you don't have one.
2. **SSH alias `hpg`** in `~/.ssh/config` that points to HPG, with key-based
   auth set up. The pipeline shells out to `ssh hpg ...` and `rsync ...`
   with this alias. Example entry:
   ```
   Host hpg
     HostName hpg.rc.ufl.edu
     User your-gatorlink
     IdentityFile ~/.ssh/id_ed25519
   ```
   Test with `ssh hpg hostname`. It should print a login node without
   prompting for a password.
3. **A workspace on `/blue/cis4914/` that you own.** The scripts ship with
   a default root they'll fall back to (see the `FASTERGS_REMOTE_ROOT` row
   in "Configuration knobs" below), but every reader should point them at
   their own `/blue/cis4914/<your-gatorlink>/gs_final` via one env var
   override. All HPG paths the pipeline touches are derived from that root.
4. **conda on HPG.** The training job loads `conda/25.7.0` as a module. You
   don't need to set up anything manually. The job script creates an env
   at `$REMOTE_ROOT/envs/fastergs_cuda128` on the first run and reuses it
   after that.

## Configuration knobs

All the settings below have sensible defaults. You only need to touch them
if you're running under a different HPG account or on a different partition.

| Env var | Used by | Default |
|---|---|---|
| `FASTERGS_REMOTE` | fastergs_pipeline.py | `hpg` (your SSH alias for HiPerGator) |
| `FASTERGS_REMOTE_ROOT` | all HPG scripts | `/blue/cis4914/joshuabowman/gs_final` |
| `FASTERGS_SLURM_ACCOUNT` | fastergs_pipeline.py | `cis4914` |
| `FASTERGS_PREP_PARTITION` | SfM/undistort SLURM job | `hpg-default` |
| `FASTERGS_TRAIN_PARTITION` | Training SLURM job | `hpg-turin` |
| `FASTERGS_COLMAP_CONTAINER` | HPG prepare job | `/apps/colmap/3.11/container.sif` |

Override `FASTERGS_REMOTE_ROOT` before running anything. The default shown
above is the original developer workspace and should not be relied on.

To use your own `/blue` space, export `FASTERGS_REMOTE_ROOT` once:

```bash
export FASTERGS_REMOTE_ROOT=/blue/cis4914/your-gatorlink/gs_final
```

Every derived path (repo, conda env, logs, output) follows automatically.
CLI flags override env vars on a per-path basis if you need to mix and match.

## Remote workspace layout

After a few runs, `$REMOTE_ROOT` looks like this:

```
gs_final/
  datasets/<dataset>/
    images/             # cleaned frames we rsynced up
    database.db         # COLMAP database from SfM
    sparse/0/           # rebuilt COLMAP model (SIMPLE_RADIAL)
  experiments/faster-gs/datasets/<dataset>/
    images/             # undistorted images (PINHOLE)
    sparse/0/           # undistorted COLMAP model (PINHOLE)
  outputs/faster-gs/<run_tag>/
    point_cloud/iteration_N/point_cloud.ply
    metrics/            # metrics.jsonl, summary JSON, PNGs
    train_stdout.log
  outputs/<run_tag>.splat
  src/                  # our helper scripts copied here
  envs/fastergs_cuda128 # conda env
  logs/                 # SLURM job outputs
```

## The camera-model gotcha

This was the biggest integration headache and it's worth knowing about if
something breaks:

- **pycolmap** (which our local SfM uses for the OpenSplat path) defaults
  to `SIMPLE_RADIAL` cameras: one radial distortion parameter per camera.
- The **Faster-GS dataset loader** only accepts `SIMPLE_PINHOLE` or
  `PINHOLE`. No distortion term, just `fx fy cx cy`.
- If you hand it a `SIMPLE_RADIAL` model, it either crashes inside scene
  setup with a cryptic tensor-shape error or trains on garbage.

The fix happens on HPG inside
`scripts/hpg_gs_final_prepare.py`: after COLMAP builds the sparse model
with `SIMPLE_RADIAL`, we run `colmap image_undistorter`. That rewrites the
images (removing lens distortion) and the sparse model into an undistorted
PINHOLE space. After that step straight lines in the scene really are
straight, the focal length is still valid, and the distortion term is
gone, exactly what the trainer expects.

`scripts/fastergs_preflight.py` runs right after and verifies the output is
actually PINHOLE-family. If somehow a `SIMPLE_RADIAL` snuck through, it
exits with code 2 and the pipeline stops before wasting GPU time.

## Running it

### From the web UI

1. Start the backend: `cd backend && uvicorn main:app --reload`.
2. Start the frontend: `cd frontend && npm run dev`.
3. Open the Live Demos page, upload a video, name the dataset, pick
   **Faster-GS** as the backend, click Run.
4. Logs stream in real-time. End-to-end is typically 10–30 minutes
   depending on video length and HPG queue.

### From the CLI

If you want to skip the UI for a scripted run:

```bash
cd backend
python scripts/fastergs_pipeline.py <dataset-name> \
  --video path/to/video.mp4 \
  --iters 10000 \
  --remote hpg \
  --slurm-account cis4914
```

The script logs every step it takes and exits non-zero on any failure.

## Where outputs land

After a successful run:

- `backend/hipergator/gs_final/<run_tag>.splat`: the fetched trained splat.
- `backend/hipergator/gs_final/<run_tag>.ply`: the raw PLY.
- `backend/datasets/<dataset>/splat.splat`: the canonical "latest splat
  for this dataset". The viewer loads this.
- `backend/hipergator/<dataset>_fastergs_latest.splat`: a flat copy for
  the Gallery page.
- `backend/datasets/<dataset>/metrics/<run_tag>/`: metrics JSONL + summary
  + PNGs (see below).

## Metrics

Every training run writes structured metrics while it's running:

- `metrics.jsonl`: one JSON record per saved PLY checkpoint, with
  iteration, loss, PSNR, gaussian count, splats/frame, wall_seconds.
- `metrics_summary.json`: final values + metadata (backend, iterations,
  partition, dataset).
- `psnr.png`, `ssim.png`, `lpips.png`, `loss.png`, `num_gaussians.png`,
  `splats_per_frame.png`, `wall_seconds.png`: rendered by matplotlib on HPG.

SSIM and LPIPS only compute when the trainer writes eval images to
`test/ours_N/renders` and `gt` (it does this at `--test_iterations`
checkpoints). SSIM uses scikit-image; LPIPS uses the `lpips` pip package
with an AlexNet backbone. If either package isn't in the HPG conda env,
the collector leaves those fields null and the Reports page hides the
corresponding charts.

The Reports page at `/reports` in the frontend reads all of this and lets
you compare runs side-by-side.

## Troubleshooting

### "preflight failed: unsupported camera model"

The undistort step didn't produce PINHOLE cameras. Check the SLURM
`gsf_prepare_*.err` log on HPG under `$REMOTE_ROOT/logs/`. Most common
cause: COLMAP couldn't build a sparse model from the input frames (too
few frames, too little parallax, blurry video). Rerun with more or
better-quality frames.

### Training job sits in queue forever

Default is `hpg-turin` (L4, sm_89). If it's busy, swap to whatever
partition your class has access to:

```bash
FASTERGS_TRAIN_PARTITION=hpg-b200 uvicorn main:app --reload
```

Or pass `--train-partition hpg-b200` to `fastergs_pipeline.py`. Note
that B200 (sm_100) can't compile FasterGS's custom kernels yet, so you'll
see `rasterizer_mode=standard` in the log. Quality is unaffected but
you lose the FasterGS speedup. See the next section if you need to
re-enable the B200 fallback patches.

### "CUDA extension build failed" / "undefined symbol"

The conda env is out of sync. Easiest fix is to blow it away and let the
next run rebuild it:

```bash
ssh hpg 'rm -rf $FASTERGS_REMOTE_ROOT/envs/fastergs_cuda128'
```

The next training run will recreate it (adds ~15 min to that run).

### "USE_FASTERGS_RASTERIZER = False" in training logs

Used to be the intentional B200 workaround. B200 is sm_100 and the
FasterGS custom rasterizer doesn't compile there yet. The sed patches
that flipped those flags off are now commented out in
`scripts/hpg_gs_final_train.py` because the default partition is
`hpg-turin` (L4, sm_89) which compiles the real kernels cleanly. If you
need to run on B200, uncomment both sed lines and you'll see the
stock-Inria fallback again.

### VGGT job fails with `ModuleNotFoundError`

Means the one-time env setup was skipped or a dep got clobbered. Rerun
the install block in the "VGGT: one-time HPG setup" section above. Safe
to run repeatedly; pip will no-op anything already installed.

### VGGT job fails with `ConnectionError` during model download

The compute node couldn't reach HuggingFace. Weights should be cached
under `$REMOTE_ROOT/models/hf/` after the first successful run. Re-run
the priming step from a login node (which has outbound HTTPS). If the
cache gets stale, `rm -rf $REMOTE_ROOT/models/hf` and re-prime.

### VGGT SLURM log says "pycolmap X.Y.Z is incompatible with VGGT"

Someone reinstalled pycolmap (often as a transitive dep of a later pip
install) and bumped it past the 3.x line. The script bails early before
the long model download. Fix:

```bash
ssh hpg '/blue/cis4914/$USER/gs_final/envs/fastergs_cuda128/bin/pip \
  install --no-deps "pycolmap==3.10.0"'
```

Nothing else in the env uses pycolmap directly (the COLMAP path runs via
an apptainer container, and preflight reads the raw `.bin` files), so the
downgrade is safe.

### VGGT finishes but training errors out with "no points in sparse model"

Means `--vggt-use-ba` got disabled somewhere (payload, CLI, or default).
Without BA, VGGT writes cameras only and `points3D.bin` is 8 bytes (empty
header). The trainer needs at least a few thousand seed points. Re-run
with BA on (the default).

### Want to force a fresh SfM even though the fingerprint matches

`rm backend/datasets/<dataset>/remote_sfm_fingerprint.json` locally, or
flip the SfM method in the UI (COLMAP ↔ VGGT). Switching methods always
invalidates the cache and triggers a new SfM run.

### Local dev server can't reach the backend

The backend runs on `:8000`, frontend dev server on `:5173`. Vite's proxy
in `frontend/vite.config.js` forwards `/api/*` and `/datasets/*` to
`localhost:8000`. If that isn't happening, check `vite.config.js` wasn't
clobbered.
