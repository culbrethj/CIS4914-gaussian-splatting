# Experimental Faster-GS Path (Non-Blocking)

This is an **experimental** track for Faster-GS.  
It does **not** replace or modify the main OpenSplat workflow.

## Repos Inspected

- Official Faster-GS (NeRFICG method): https://github.com/nerficg-project/faster-gaussian-splatting
- Faster-GS integration into Inria 3DGS codebase: https://github.com/fhahlbohm/gaussian-splatting

## 1) Install Requirements on HiPerGator

For fastest low-risk testing with your current project, use the **Inria integration fork**.

Core requirements from upstream:
- NVIDIA GPU + CUDA-capable node
- Linux + Conda
- PyTorch + CUDA-compatible compiler toolchain
- Repo submodules (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`)

Practical HPG module baseline (adjust if your account has different module names):

```bash
module purge
module load git cmake gcc/12.2.0 conda/25.7.0 cuda/12.4.1
```

Create env (CUDA12 variant from upstream fork):

```bash
conda env create -f environment_cuda12.yml -p /blue/cis4914/joshuabowman/conda/fastergs_inria
conda activate /blue/cis4914/joshuabowman/conda/fastergs_inria
```

## 2) Expected Input Format

Faster-GS Inria fork expects COLMAP-style source path:

```text
<scene_root>/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

## 3) Does Current SfM/COLMAP Output Work?

Partially:
- Your structure is close (`images/`, `sparse/0/*` exists).
- But Faster-GS Inria loader only supports `SIMPLE_PINHOLE` or `PINHOLE`.
- Your current `backend/datasets/can/sparse/0/cameras.bin` is `SIMPLE_RADIAL`.

Use preflight checker:

```bash
cd backend
python scripts/fastergs_preflight.py can
```

## 4) Training Command

From the Inria integration fork:

```bash
python train.py \
  -s /path/to/scene_root \
  -m /path/to/output/model_dir \
  --iterations 1000 \
  --disable_viewer
```

## 5) Output Format

Saved model outputs include:

```text
<model_dir>/point_cloud/iteration_<N>/point_cloud.ply
```

The exported Gaussian PLY contains fields like:
- `x y z`
- `f_dc_0..2`
- `f_rest_*`
- `opacity`
- `scale_0..2`
- `rot_0..3`

## 6) Can Output Be Converted to Our Viewer Format?

Yes, likely directly.

Your converter (`backend/scripts/converter.py`) already supports:
- `f_dc_*`
- `opacity`
- `scale_*`
- `rot_*`

Conversion:

```bash
python backend/scripts/converter.py /path/to/point_cloud.ply
```

Then place `.splat` under:
- dataset root (`backend/datasets/<name>/...`) for dataset API usage, or
- `backend/hipergator/<name>.splat` for Gallery-style loading.

## 7) Estimated Integration Risk

Low-risk to pilot (experimental script + docs only): **Low**

Runtime integration risk (without replacing OpenSplat): **Medium**
- Need camera undistortion / camera-model compatibility step (`SIMPLE_RADIAL` -> `PINHOLE/SIMPLE_PINHOLE`)
- Different training runtime profile than OpenSplat
- Additional dependency stack (submodules + CUDA extensions)

Full production switch risk (replace OpenSplat): **Medium-High**
- New failure modes, tuning differences, and pipeline validation needed

## Suggested Pilot Plan

1. Keep OpenSplat as default.
2. Add undistorted COLMAP export path for one test scene.
3. Train Faster-GS Inria fork for ~1000 iterations.
4. Convert `point_cloud.ply` to `.splat`.
5. Compare quality/time against current OpenSplat baseline.

## Experimental Dataset Conversion (Undistort + Verify)

Use the helper script in this repo:

```bash
cd backend
python scripts/fastergs_prepare_dataset.py can
```

What it does:
1. Reads `backend/datasets/can` (or any dataset you pass).
2. Runs `colmap image_undistorter` if COLMAP is available.
3. Writes to a separate output dataset by default:
   - `backend/experiments/faster-gs/datasets/can`
4. Normalizes output layout to `sparse/0/*` if needed.
5. Runs `fastergs_preflight` checks on the output dataset.

It does **not** modify the original dataset.

Alternative output location next to source dataset:

```bash
python scripts/fastergs_prepare_dataset.py can --output-mode adjacent
```

This writes:
- `backend/datasets/can_undistorted`

Explicit output path:

```bash
python scripts/fastergs_prepare_dataset.py can --output /blue/cis4914/joshuabowman/gaussian-splatting/datasets/can_fastergs
```

If output exists:

```bash
python scripts/fastergs_prepare_dataset.py can --force
```

### HiPerGator SLURM Submit Helper

To run undistortion on a compute node (not login), use:

```bash
cd backend
python scripts/hpg_fastergs_undistort.py can \
  --remote hpg \
  --slurm-account cis4914 \
  --source-root /blue/cis4914/joshuabowman/datasets
```

Submit-only mode (no polling/fetch from local script):

```bash
python scripts/hpg_fastergs_undistort.py can \
  --remote hpg \
  --slurm-account cis4914 \
  --source-root /blue/cis4914/joshuabowman/datasets \
  --no-wait
```

### HiPerGator SLURM Training Helper

After undistortion/preflight succeeds, submit Faster-GS training on a GPU compute node:

```bash
cd backend
python scripts/hpg_fastergs_train.py can \
  --remote hpg \
  --slurm-account cis4914 \
  --slurm-gpus 1 \
  --iterations 1000
```

This helper will:
1. Ensure Faster-GS repo checkout exists on `/blue`.
2. Create/activate isolated conda env at `/blue/cis4914/joshuabowman/conda/fastergs_inria` if missing.
3. Train using scene root:
   - `/blue/cis4914/joshuabowman/gaussian-splatting/experiments/faster-gs/datasets/<dataset>`
4. Convert latest `point_cloud.ply` to `.splat` using this repo's converter.
5. Publish `.splat` to:
   - `/blue/cis4914/joshuabowman/gaussian-splatting/hipergator/<run_tag>.splat`
6. Fetch the published `.splat` back to local:
   - `backend/hipergator/<run_tag>.splat`
