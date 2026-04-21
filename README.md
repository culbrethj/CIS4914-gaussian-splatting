# Gaussian Splatting Studio: CIS4914 Senior Project

A web app for turning a phone video into a photorealistic 3D scene you can
explore in the browser.

Gaussian Splatting is a 2023 rendering technique that produces remarkably
realistic 3D reconstructions from a set of photos or a video. It's powerful
but historically hard to use: the reference implementations need CUDA
hardware, Python environments, and command-line workflows. Our project
makes it approachable.

You can see a hosted version of the site at https://culbrethj.github.io/CIS4914-gaussian-splatting/. The Gallery, Reports, and Viewer pages all work there with our trained showcase scenes. The Live Demos page is shown for preview but the upload + training workflow needs the backend running locally. Instructions below.

## What the app does

1. Upload a video (or pick one of the bundled samples).
2. The backend extracts frames, runs Structure-from-Motion, and trains a
   3D Gaussian Splatting model. Training can run locally via OpenSplat or
   remotely on a GPU cluster via our Faster-GS integration.
3. The browser renders the resulting 3D scene. Drag to orbit, scroll to
   zoom, tweak splat scale and camera in real time.

## Prerequisites

Before cloning + running this repo, make sure you have:

- **Python 3.10–3.13** for the backend + pipeline scripts. The HPG conda
  env is pinned at 3.10.14; matching locally avoids subtle import quirks.
  Python 3.14 is not yet supported (open3d has no 3.14 wheels).
- **Node.js 20+** for the Vite frontend.
- **An HPG account on the `cis4914` SLURM allocation** if you want to
  run the Faster-GS backend (the default for training). OpenSplat
  runs locally and doesn't need HPG.
- **An SSH alias named `hpg`** in your `~/.ssh/config` pointing at
  the HiPerGator login node with key-based auth already set up. The
  pipeline scripts shell out to `ssh hpg …` and `rsync … hpg:…`;
  without the alias every HPG step breaks. One-time config example
  and full setup walkthrough lives in
  [backend/experiments/faster-gs/README.md](backend/experiments/faster-gs/README.md).

If you're only running OpenSplat locally, skip the HPG items and install
OpenSplat per the [pierotofy/OpenSplat](https://github.com/pierotofy/OpenSplat)
README (final binary goes in `backend/binaries/`).

Faster-GS backend setup (camera-model undistort, conda env on HPG,
troubleshooting) is documented in detail at
[backend/experiments/faster-gs/README.md](backend/experiments/faster-gs/README.md),
worth skimming before the first training run.

## See it running

To see what a splat looks like without running anything locally, just open the [hosted version](https://culbrethj.github.io/CIS4914-gaussian-splatting/) and click into Gallery. Otherwise, to run it yourself:

1. `./scripts/setup-local.sh` (installs deps; requires Python 3.10–3.13)
2. Terminal 1: `cd frontend && npm run dev`
3. Terminal 2: `source venv/bin/activate && cd backend && uvicorn main:app --reload`
4. Open `http://localhost:5173/gallery`, pick `cone`, `snowboard`, or `gator_statue`, drag to spin.

See [SETUP.md](SETUP.md) for full installation across Mac, Windows, and
HiPerGator.

## Project structure

Three-stage pipeline (preprocess → SfM → training) wrapped in a FastAPI
backend and a React frontend.

| Stage | OpenSplat (local) | Faster-GS (remote on HPG) |
|-------|-------------------|---------------------------|
| Preprocess | [backend/scripts/preprocessor.py](backend/scripts/preprocessor.py) | same |
| SfM | [backend/scripts/sfm.py](backend/scripts/sfm.py) | [hpg_gs_final_prepare.py](backend/scripts/hpg_gs_final_prepare.py) (COLMAP) or [hpg_gs_vggt_sfm.py](backend/scripts/hpg_gs_vggt_sfm.py) (VGGT) |
| Training | [backend/binaries/opensplat](backend/binaries/) | [hpg_gs_final_train.py](backend/scripts/hpg_gs_final_train.py) (SLURM job) |

Frontend pages (under [frontend/src/pages/](frontend/src/pages/)):

- `/`: Landing + feature cards
- `/demos`: Upload video, run pipeline, watch live logs ([LiveDemos.jsx](frontend/src/pages/LiveDemos.jsx)). On the hosted site this page is read-only since there's no backend; locally it works fully.
- `/gallery`: Browse trained splats, compare runs side-by-side ([Gallery.jsx](frontend/src/pages/Gallery.jsx))
- `/reports`: Training metrics and charts ([Reports.jsx](frontend/src/pages/Reports.jsx))
- `/converter`: Client-side PLY → SPLAT conversion ([Converter.jsx](frontend/src/components/Converter.jsx))
- `/documentation`: Viewer controls and pipeline overview ([Documentation.jsx](frontend/src/pages/Documentation.jsx))

CLI entry points:

- Full pipeline: `python backend/scripts/fastergs_pipeline.py <dataset> --video <path>`
- Experiment matrix: `python backend/experiments/faster-gs/shortgs/run_matrix.py`

Deeper docs:

- [SETUP.md](SETUP.md): cross-platform setup
- [backend/experiments/faster-gs/README.md](backend/experiments/faster-gs/README.md): HPG + Faster-GS
- [backend/experiments/faster-gs/shortgs/README.md](backend/experiments/faster-gs/shortgs/README.md): Shorter-Splatting ablation harness

## Running tests

```bash
pytest backend/tests/
```

## Testing and validation

We validated the pipeline by running a matrix of training experiments across 4
scenes on NVIDIA B200 GPUs. The experiments cover backend comparison
(stock Adam vs fused Adam vs OpenSplat), paper-flag ablation from the
Shorter-Splatting paper, and iteration-count scaling. Findings are
discussed in the written capstone report.

## Setup

Full cross-platform setup lives in [SETUP.md](SETUP.md). Quick version for
Mac/Linux from the repo root:

```bash
./scripts/setup-local.sh          # installs deps (Python 3.10-3.13, Node 20+)
source venv/bin/activate && cd backend && uvicorn main:app --reload  # terminal 1
cd frontend && npm run dev        # terminal 2
```

Open http://localhost:5173. Windows and HPG-direct setup in SETUP.md.

## Training backends

The Live Demos page lets you pick between two training backends:

- **OpenSplat**: runs locally against the bundled binary under
  `backend/binaries/`. No GPU required (CPU or CUDA).
- **Faster-GS**: sends the job to HiPerGator and trains there on a GPU
  partition. This is the default and tends to be more reliable.

OpenSplat is simpler to set up (no HPG account needed) but has had
dylib/runtime issues on some machines. Faster-GS needs an HPG account and
an SSH alias `hpg`, but gives you the full pipeline with metrics + charts
on the Reports page.

Full Faster-GS setup instructions (HPG workspace layout, camera-model
undistort step, troubleshooting) live in
[backend/experiments/faster-gs/README.md](backend/experiments/faster-gs/README.md).

## Team

- **Jackson Culbreth**, Team Leader
- **Joshua Bowman**, Scrum Master
- **Jackson Kelly**, Developer
- **Nicolas Desmornes**, Developer

## Credits

Third-party code and papers this project builds on:

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., SIGGRAPH 2023)
- [Faster Gaussian Splatting](https://github.com/nerficg-project/faster-gaussian-splatting) (Hahlbohm et al., CVPR 2026)
- [OpenSplat](https://github.com/pierotofy/OpenSplat), an open-source 3DGS training binary in C++
- [VGGT: Visual Geometry Grounded Transformer](https://github.com/facebookresearch/vggt): feed-forward SfM alternative
- Shorter-Splatting training techniques ([arXiv 2603.09277](https://arxiv.org/pdf/2603.09277)): scale reset, entropy regularization, progressive resolution
- [COLMAP](https://colmap.github.io/): Structure-from-Motion pipeline
- [@react-three/drei](https://github.com/pmndrs/drei) `<Splat>` component: browser splat renderer

---

## Due dates

- ~~Feb 01: Project Proposal~~
- ~~Feb 06: Week 4 reports~~
- ~~Feb 13: Week 5 reports~~
- ~~Feb 15: Presentation 1 video~~
- ~~Feb 20: Week 6 reports~~
- ~~Feb 27: Week 7 reports~~
- ~~Mar 06: Week 8 reports~~
- ~~Mar 13: Week 9 reports~~
- ~~Mar 13: Presentation 2 video~~
- ~~Mar 27: Week 11 reports~~
- ~~Apr 03: Week 12 reports~~
- ~~Apr 10: Week 13 reports~~
- ~~Apr 14: Senior Showcase~~
- ~~Apr 21: Final Presentation video~~
- ~~Apr 21: GitHub repo submission~~
