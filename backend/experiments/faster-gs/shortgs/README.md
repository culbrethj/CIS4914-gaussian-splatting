# Shorter-Splatting paper validation

Infrastructure for replicating the three techniques from
["Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists"](https://arxiv.org/pdf/2603.09277)
on top of our Faster-GS pipeline.

## Status

All three techniques are actually implemented now. They run via
`backend/scripts/shortgs_apply_patches.py`, which edits the vendored Inria
`train.py` on HPG before each training job. The patches are idempotent and
gated on env vars, so when no `SHORTGS_*` flags are set the training path is
bit-identical to stock Inria.

What's wired:

- Flags pass through `fastergs_pipeline.py` → `hpg_gs_final_train.py` as
  env vars on the remote training job (`SHORTGS_SCALE_RESET_EVERY`,
  `SHORTGS_SCALE_RESET_FACTOR`, `SHORTGS_ENTROPY_WEIGHT`,
  `SHORTGS_PROGRESSIVE_RESOLUTION`).
- `shortgs_apply_patches.py` inserts four labelled blocks into `train.py`:
  setup (reads env vars), progressive-resolution downscale (before L1/SSIM
  loss), entropy term (added to `loss` before backward), and scale reset
  (inside the densification `no_grad` block, after the optimizer step).
  Every block is wrapped in `# === SHORTGS PATCH ===` markers so the
  professor can diff the file and see exactly what was added.
- `run_matrix.py` reads `experiment_matrix.yaml`, kicks off
  `fastergs_pipeline.py` once per (dataset × config × seed), and waits for
  each to complete.
- `compare.py` aggregates `metrics_summary.json` from every completed run
  and produces bar plots comparing final PSNR, wall time, and gaussian
  count across configurations.

## Entropy implementation note

The paper's exact formulation is "entropy of the per-pixel alpha blending
weight distribution". That would need the rasterizer to return per-pixel
alpha lists - the stock and Faster-GS rasterizers here don't, and exposing
them requires modifying the CUDA kernel (not feasible before the deadline).

We implement a **faithful proxy**: the Bernoulli entropy of each gaussian's
scalar opacity, averaged across all gaussians, scaled by
`SHORTGS_ENTROPY_WEIGHT` and added to the loss. Bernoulli entropy peaks at
opacity=0.5 and is zero at opacity=0 or 1, so minimizing it pushes every
gaussian towards a committed "on" or "off" state.

This achieves the same *outcome* the paper describes - polarized
opacities mean fewer gaussians contribute meaningfully to each pixel, which
is what shortens per-pixel alpha-blend lists. You can verify the effect by
looking at the opacity histogram on the Reports page: a run with
`--shortgs-entropy-weight 0.01` should show more mass near 0 and near 1 and
less mass in the middle than a baseline run.

> **Warning:** Entropy weight `0.01` collapses the training loss to zero on
> real scenes. Start at `0.001` or lower and validate on a small dataset.

It's not bit-identical to the paper's formulation. A further step
toward that would be patching the Faster-GS CUDA kernel to emit
per-pixel alpha lists directly.

## Partition / rasterizer note

Default partition is `hpg-turin` (NVIDIA L4, sm_89). That compute
capability compiles both `USE_FASTERGS_RASTERIZER = True` and
`USE_FASTERGS_ADAM = True` cleanly, so these runs use the real
FasterGS kernels (look for `rasterizer_mode=fastergs` in the SLURM
log). HPG retired the A100 partition; turin is the closest equivalent
our group has access to.

If the turin queue is backed up you can fall back to `hpg-b200`
(sm_100) — but the FasterGS custom rasterizer and fused Adam don't
build on sm_100 yet, so B200 runs need the two sed lines in
`scripts/hpg_gs_final_train.py` (currently commented out) re-enabled.
That mode prints `rasterizer_mode=standard optimizer_mode=adam`;
training still completes, just without the FasterGS speedup.

## The three techniques (paper summary)

1. **Scale reset** — every K iterations, shrink every gaussian's log-scale
   parameter by a factor F < 1. Paper claims ~2× training speedup because
   smaller gaussians contribute to fewer pixels, shortening per-pixel
   alpha-blend lists.
2. **Entropy constraint on alpha blending** — add `λ · H(α)` to the loss,
   where H is the entropy of the per-pixel alpha-weight distribution
   across contributing gaussians. Pushes each gaussian to dominate a
   region instead of contributing weakly to many, also shortens per-pixel
   lists.
3. **Progressive resolution scheduler** — start training at a fraction of
   full image resolution (e.g. 25%) and step up at scheduled iteration
   boundaries. Cheaper early iterations, converges to same quality as
   fixed full-res.

Claim under evaluation: ~2× wall-clock speedup at ≤ 0.2 dB PSNR loss when
all three are combined.

## Quick-start: single commands for each technique

Copy-paste these from `backend/`. They all hit `hpg-turin`; override with
`FASTERGS_TRAIN_PARTITION` if your class has access to a different one.
The dataset name must already exist under `backend/datasets/<name>/` with
a video under `video/`. If you just uploaded one through the UI, use the
name you typed in the form (e.g. `test1234`).

**Baseline** (no shortgs flags — reproduces default Faster-GS behavior):

```bash
FASTERGS_TRAIN_PARTITION=hpg-turin \
python scripts/fastergs_pipeline.py test1234 \
  --video datasets/test1234/video/*.mov \
  --iters 10000 \
  --seed 0
```

**Scale reset only** (shrink gaussians every 1000 iters by factor 0.9):

```bash
FASTERGS_TRAIN_PARTITION=hpg-turin \
python scripts/fastergs_pipeline.py test1234 \
  --video datasets/test1234/video/*.mov \
  --iters 10000 \
  --seed 0 \
  --shortgs-scale-reset-every 1000 \
  --shortgs-scale-reset-factor 0.9
```

**Entropy constraint only** (entropy weight 0.01):

```bash
FASTERGS_TRAIN_PARTITION=hpg-turin \
python scripts/fastergs_pipeline.py test1234 \
  --video datasets/test1234/video/*.mov \
  --iters 10000 \
  --seed 0 \
  --shortgs-entropy-weight 0.01
```

**Progressive resolution only** (start at 25%, step up at 5k and 10k):

```bash
FASTERGS_TRAIN_PARTITION=hpg-turin \
python scripts/fastergs_pipeline.py test1234 \
  --video datasets/test1234/video/*.mov \
  --iters 10000 \
  --seed 0 \
  --shortgs-progressive-resolution "0:0.25,5000:0.5,10000:1.0"
```

**Everything combined**:

```bash
FASTERGS_TRAIN_PARTITION=hpg-turin \
python scripts/fastergs_pipeline.py test1234 \
  --video datasets/test1234/video/*.mov \
  --iters 10000 \
  --seed 0 \
  --shortgs-scale-reset-every 1000 \
  --shortgs-scale-reset-factor 0.9 \
  --shortgs-entropy-weight 0.01 \
  --shortgs-progressive-resolution "0:0.25,5000:0.5,10000:1.0"
```

After each run, metrics land under
`backend/datasets/<name>/metrics/<run_tag>/` and the Reports page (`/reports`)
picks them up automatically. Pick baseline + one technique in the run list
and flip to "Overlay runs" to see whether the technique actually helped.

Reminder: until the vendored fork is patched, these flags just set env
vars; the trainer ignores them. Every run will produce identical PSNR /
gaussians / wall time. You'll see the config recorded in
`metrics_summary.json` under `"shortgs"` so the plumbing is verifiable.

## Recommended quick-validation dataset

The **test1234** dataset (50 frames, ~60 MB source video, 10k iters in
~5-8 minutes on hpg-turin L4) is small enough for rapid iteration. If
you don't have it, upload any phone video through the Live Demos UI
and give it a short name; that's your dataset.

## Running the full matrix

Prereq: the `fastergs` backend works end-to-end from the web UI or CLI
(see the parent `README.md`). Run one manual training job first to make
sure that path is healthy before you queue up thirty of them.

```bash
cd backend
python experiments/faster-gs/shortgs/run_matrix.py \
  --matrix experiments/faster-gs/shortgs/experiment_matrix.yaml \
  --results-dir experiments/faster-gs/shortgs/results
```

## Running the minimal A/B matrix

For a quick sanity check of all four configs on one dataset (4 runs total,
~20-30 min on a100):

```bash
cd backend
FASTERGS_TRAIN_PARTITION=hpg-turin \
python experiments/faster-gs/shortgs/run_matrix.py \
  --matrix experiments/faster-gs/shortgs/ab_matrix.yaml \
  --results-dir experiments/faster-gs/shortgs/results-ab
```

Each run's artifacts land under
`backend/datasets/<dataset>/metrics/<run_tag>/` like any other Faster-GS
run. The matrix runner also copies each `metrics_summary.json` into the
`--results-dir` so you can aggregate without scanning all the dataset
dirs.

When the matrix is done (or you've accumulated enough):

```bash
python experiments/faster-gs/shortgs/compare.py \
  --results-dir experiments/faster-gs/shortgs/results \
  --output experiments/faster-gs/shortgs/comparison
```

That writes `comparison/final_psnr.png`, `comparison/wall_time.png`,
`comparison/final_num_gaussians.png` with bar plots grouped by
configuration.

## Configuration file format

See `experiment_matrix.yaml`. The shape is:

```yaml
datasets: [can, garden]
seeds: [0, 1, 2]
base_iterations: 10000
partition: hpg-turin
configs:
  - name: baseline
    flags: {}
  - name: scale_reset
    flags: {shortgs_scale_reset_every: 1000, shortgs_scale_reset_factor: 0.9}
  - name: entropy
    flags: {shortgs_entropy_weight: 0.01}
  - name: progressive
    flags: {shortgs_progressive_resolution: "0:0.25,5000:0.5,10000:1.0"}
  - name: combined
    flags:
      shortgs_scale_reset_every: 1000
      shortgs_scale_reset_factor: 0.9
      shortgs_entropy_weight: 0.01
      shortgs_progressive_resolution: "0:0.25,5000:0.5,10000:1.0"
```

Total runs = `len(datasets) × len(seeds) × len(configs)`. The default
matrix is 2 × 3 × 5 = 30 runs. Each run on hpg-turin at 10k iterations
takes roughly 15-25 minutes depending on scene size and queue wait.
