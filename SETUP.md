# Setup Guide

Full cross-platform setup for this repo. Quick-start per platform below, then prerequisites, validation steps, and troubleshooting.

**Feature branch**: check out `jb/fastergs-opensplat-integration` before following this guide. Don't set up off `main` — the current experiment work lives on the feature branch.

```bash
git clone <repo-url>
cd CIS4914-gaussian-splatting
git checkout jb/fastergs-opensplat-integration
```

---

## Quick start

- **Mac / Linux**: run `./scripts/setup-local.sh` from the repo root. See [Mac / Linux setup (automated)](#mac--linux-setup-automated).
- **Windows**: manual flow (PowerShell recommended). See [Windows setup](#windows-setup-manual).
- **VS Code Remote on HiPerGator**: most things are already built on the cluster. See [HPG-direct setup](#hpg-direct-setup-vs-code-on-hipergator).

## Running tests

Backend test suite (21 tests, runs in under a second):

```bash
cd backend && pytest tests/
```

Run this after setup to confirm the install works, and before committing
backend changes.

---

## What you need regardless of platform

- **Python 3.10–3.13**. The pinned dependency stack in `requirements.txt` (open3d 0.19.0, numpy 2.4.2, pandas 3.0.1, …) has wheels through Python 3.13 only — 3.14 fails to resolve a pinned wheel for `open3d` as of writing. 3.10.14 is what the HPG conda env uses, so matching locally avoids subtle import quirks. The setup script auto-picks the first working `python3.13` / `python3.12` / `python3.11` / `python3.10` on PATH if your default `python3` is too new.
- **Node.js 20+** with npm (Vite 7 needs a modern Node runtime).
- **Git**.
- **HiPerGator `cis4914` SLURM group** access, if you plan to run the Faster-GS backend or the experiment matrix. OpenSplat runs locally and doesn't need HPG.
- **SSH alias `hpg`** in `~/.ssh/config` (Mac/Linux) or `C:\Users\<you>\.ssh\config` (Windows) pointing at the HiPerGator login node, with key-based auth already working. The pipeline shells out to `ssh hpg …`, `scp …`, and `rsync …` with that alias. Without it, every HPG step breaks.

A minimal SSH config entry:

```ssh-config
Host hpg
    HostName hpg.rc.ufl.edu
    User your-gatorlink
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm_%C.sock
    ControlPersist 8h
```

Run `ssh hpg` once interactively (Duo push + key unlock) so the persistent SSH mux is live — the pipeline's subsequent ssh/scp/rsync calls piggyback on it without re-authing every step.

---

## Mac / Linux setup (automated)

From the repo root:

```bash
./scripts/setup-local.sh
```

That handles the Python venv, `pip install -r requirements.txt`, and `npm install` in `frontend/`. It's idempotent — safe to re-run whenever you pull new commits or update `requirements.txt`.

After setup completes, in **separate terminals**:

```bash
# Terminal 1: activate venv + start the backend API
source venv/bin/activate
cd backend
uvicorn main:app --reload
# API up at http://localhost:8000

# Terminal 2: start the frontend dev server
cd frontend
npm run dev
# UI up at http://localhost:5173
```

Open http://localhost:5173 — that's the main app.

---

## Windows setup (manual)

PowerShell is recommended over `cmd` (better venv activation and rich error output). WSL2 is a reasonable fallback if native Windows gets painful — inside WSL2, just follow the Mac/Linux path above.

### 1. Install prerequisites

- **Python 3.10+**: install from [python.org](https://www.python.org/downloads/windows/). During install, check "Add Python to PATH".
- **Node 20+**: install from [nodejs.org](https://nodejs.org/) (LTS). Ships with `npm`.
- **Git**: install from [git-scm.com](https://git-scm.com/download/win).
- **OpenSSH client**: Windows 10/11 ships with it; if `ssh` isn't available from PowerShell, install "OpenSSH Client" via *Settings → Apps → Optional Features*.

Verify in PowerShell:

```powershell
python --version     # 3.10 or higher
node --version       # v20 or higher
git --version
ssh -V
```

### 2. Create + activate the venv (repo root)

```powershell
cd C:\path\to\CIS4914-gaussian-splatting
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation errors with *"execution of scripts is disabled on this system,"* run once per user account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then re-run `.\venv\Scripts\Activate.ps1`.

### 3. Install Python deps (from repo root, venv activated)

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install frontend deps

```powershell
cd frontend
npm install
cd ..
```

### 5. Configure SSH

Create (or edit) `C:\Users\<you>\.ssh\config`. Windows does not auto-create the `.ssh` folder — you may need to create it manually, and set permissions so only your user can read it:

```powershell
mkdir $HOME\.ssh -ErrorAction SilentlyContinue
notepad $HOME\.ssh\config
```

Paste the SSH config block from the "What you need" section above. Save + close. Then:

```powershell
ssh hpg
```

to perform the first-time Duo + key auth and verify connectivity.

**Windows-specific gotcha**: the `ControlMaster` / `ControlPath` / `ControlPersist` keys in the config block are Unix-only — OpenSSH for Windows ignores them. That's fine; the pipeline scripts fall back to per-call auth if no mux is available. You'll just get prompted for Duo more often. Not a blocker.

### 6. Run backend + frontend

In one PowerShell window (with venv active):

```powershell
cd backend
uvicorn main:app --reload
```

In a second PowerShell window:

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173.

---

## HPG-direct setup (VS Code on HiPerGator)

For teammates running VS Code Remote SSH against HPG, or working directly in the cluster shell, and skipping the local frontend.

1. **SSH into HPG**, clone the repo into your own `/blue/cis4914/<gatorlink>/` workspace (or reuse `/blue/cis4914/joshuabowman/` for read-only access to shared artifacts).
2. **Check out the feature branch**:

   ```bash
   git checkout jb/fastergs-opensplat-integration
   ```

3. **Conda env**: the pinned Faster-GS env already exists on HPG at `/blue/cis4914/joshuabowman/gs_final/envs/fastergs_cuda128`. You can either:
   - Activate the shared env directly:
     ```bash
     module load conda/25.7.0 cuda/12.8.1 gcc/12.2.0
     conda activate /blue/cis4914/joshuabowman/gs_final/envs/fastergs_cuda128
     ```
   - Build your own copy following `backend/experiments/faster-gs/README.md` (section "VGGT: one-time HPG setup" + the surrounding env setup). Takes ~15 min for a fresh install.

4. **Override defaults if your workspace differs from `/blue/cis4914/joshuabowman/gs_final`**:

   ```bash
   export FASTERGS_REMOTE_ROOT=/blue/cis4914/<your-gatorlink>/gs_final
   export FASTERGS_OPENSPLAT_ROOT=$FASTERGS_REMOTE_ROOT/src/OpenSplat
   ```

   All HPG paths in the pipeline scripts default via `FASTERGS_REMOTE_ROOT` + `FASTERGS_OPENSPLAT_ROOT`, so exporting these is usually all you need.

5. **Run experiments** directly (no frontend needed):

   ```bash
   # Single run against an already-preprocessed dataset on HPG:
   python backend/scripts/fastergs_pipeline.py test123456 \
     --use-existing-frames \
     --iters 2000 \
     --seed 1 \
     --backend opensplat \
     --sfm-method vggt

   # Full matrix (datasets × seeds × configs from experiment_matrix.yaml):
   python backend/experiments/faster-gs/shortgs/run_matrix.py \
     --matrix backend/experiments/faster-gs/shortgs/experiment_matrix.yaml \
     --results-dir $FASTERGS_REMOTE_ROOT/results
   ```

   > **Note**: `fastergs_pipeline.py` takes `dataset` as a positional arg (no `--dataset`), and the flag is `--iters` (not `--iterations`). `--use-existing-frames` reuses the preprocessed `images/` dir on disk so you don't need a video file.

6. **Results land** locally (to the machine running `fastergs_pipeline.py`) under `backend/datasets/<dataset>/` and `backend/hipergator/gs_final/<run_tag>.splat`. Run the matrix from HPG → results live in `$FASTERGS_REMOTE_ROOT/results/`. Run it from your laptop that's SSHed into HPG → results sync back to your laptop via `scp`/`rsync` steps inside the pipeline.

7. **To view results in the gallery**, the frontend is optional — if you want it, start Vite on your laptop as usual and point `/api/datasets/<name>/splat` + `/api/hpg/gs_final/<tag>.splat/splat` at the locally-fetched artifacts. Port-forwarding VS Code Remote SSH's 5173/8000 tunnels back to localhost works too.

---

## Validating your setup

### Mac / Linux / Windows (with frontend)

1. Start backend (`uvicorn main:app --reload`) and frontend (`npm run dev`) in separate terminals.
2. Open http://localhost:5173.
3. Go to the Gallery page and pick one of the pre-existing datasets from the dropdown (e.g. `banana`, `truck`, `test123456`).
4. The splat should load in the viewer within a second or two. You should be able to orbit with the mouse, and the controls panel (top-left of the viewer) should expand when you click its caret.

If all of that works, the local setup is good.

### HPG-direct

1. SSH to HPG, activate the conda env:
   ```bash
   conda activate /blue/cis4914/joshuabowman/gs_final/envs/fastergs_cuda128
   ```
2. Dry-run the matrix:
   ```bash
   python backend/experiments/faster-gs/shortgs/run_matrix.py \
     --matrix backend/experiments/faster-gs/shortgs/experiment_matrix.yaml \
     --results-dir /tmp/dry-run-results \
     --dry-run
   ```

   This should print the queued commands for every (dataset × config × seed) without submitting anything to SLURM.

If the dry-run prints planned invocations cleanly, the HPG setup is good.

---

## Running experiments

Full documentation lives in [backend/experiments/faster-gs/README.md](backend/experiments/faster-gs/README.md) — covers the HPG workspace layout, camera-model undistort step, shortgs paper flags, partition defaults, and troubleshooting. Read it before kicking off the matrix.

The variants in the current `experiment_matrix.yaml`:

| Variant | Backend | Notes |
|---|---|---|
| `stock_baseline` | FasterGS | Stock Inria rasterizer + `torch.optim.Adam`. Reference point. |
| `fastergs_adam` | FasterGS | Stock rasterizer + FasterGS fused Adam (the portion of FasterGS that compiles on L4/B200). |
| `scale_reset` / `entropy` / `progressive` / `combined` | FasterGS | Shorter-Splatting paper techniques. Off unless a `shortgs_*` flag is set. |
| `opensplat` | OpenSplat | C++ binary trainer on HPG; ignores FasterGS/shortgs flags. |

---

## Troubleshooting

- **`pip install` fails on `open3d==0.19.0` or `numpy==2.4.2` with "No matching distribution found"**: your Python is too new. The pinned stack targets Python 3.10–3.13; Python 3.14 has no matching wheels for these packages yet. Install a supported interpreter (pyenv / `brew install python@3.12` / python.org) and re-run `./scripts/setup-local.sh` — it'll pick up the versioned binary automatically.
- **`pip install` errors with "Invalid requirement: 'requirements.txt'"**: you forgot the `-r` flag. Run `pip install -r requirements.txt`, not `pip install requirements.txt`.
- **`pip install` installs into the system Python instead of the venv**: venv isn't active. `source venv/bin/activate` (Mac/Linux) or `.\venv\Scripts\Activate.ps1` (Windows) before re-running.
- **Node version mismatch**: use `nvm use 20` (or install Node 20 from nodejs.org). Vite 7 uses syntax not supported on older Node.
- **SSH alias not resolving**: check `~/.ssh/config` (Mac/Linux) or `C:\Users\<you>\.ssh\config` (Windows) contains a `Host hpg` block. Run `ssh hpg` directly — if you get the HPG login banner, the alias works; if you get "could not resolve hostname", the config file isn't being read.
- **Frontend port 5173 already in use**: kill the other process (`lsof -iTCP:5173 -sTCP:LISTEN -P` on Mac/Linux, `Get-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess` on Windows) or start Vite on a different port: `npm run dev -- --port 5174`.
- **Backend port 8000 already in use**: kill the other process or run `uvicorn main:app --reload --port 8001`.
- **PowerShell execution policy blocks venv activation**: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (runs once per user).
- **HPG conda env is broken or out of sync**: wipe and rebuild:
  ```bash
  ssh hpg 'rm -rf /blue/cis4914/<you>/gs_final/envs/fastergs_cuda128'
  # Next fastergs_pipeline.py run with --stage setup rebuilds it; ~15 min.
  ```
  The details are in `backend/experiments/faster-gs/README.md`.
- **OpenSplat binary not found on HPG**: verify `/blue/cis4914/joshuabowman/gs_final/src/OpenSplat/build_b200/opensplat` exists and is executable. If it doesn't, the shared build isn't set up yet — ping the maintainer or build from source per pierotofy/OpenSplat's README.
- **Duo prompts on every ssh call**: make sure `ControlMaster auto` + `ControlPath` + `ControlPersist 8h` are in your `~/.ssh/config` `Host hpg` block (Mac/Linux only — Windows OpenSSH ignores them). Then run `ssh hpg` once interactively, authenticate, and exit; subsequent ssh/scp/rsync reuse the mux for 8 hours.
- **`pip install` of `lpips` pulls a huge torch wheel**: expected. LPIPS needs torch + torchvision. If you only care about PSNR (the other SSIM/LPIPS metrics are optional), you can comment out the `lpips` line in `requirements.txt` — `metrics_collector.py` lazy-imports it and gracefully skips LPIPS if unavailable.

---

## Platform notes

- **OpenSplat and FasterGSCudaBackend are HPG-only.** The prebuilt binaries live under `/blue/cis4914/joshuabowman/gs_final/src/OpenSplat` and the conda env has FasterGSCudaBackend compiled for `sm_89` (L4) + `sm_100` (B200). Local machines (Mac / Windows / Linux laptops) **do not build these** — training jobs are dispatched to HPG via SLURM.
- **Frontend gallery reads locally-synced results.** The Faster-GS pipeline fetches `.splat` + metrics back to `backend/hipergator/gs_final/` and `backend/datasets/<dataset>/metrics/` on the machine that kicked off the run; the frontend Gallery page reads those. If you're running the matrix on HPG-direct and want to view results, either fetch them down with `rsync` or run Vite inside VS Code Remote SSH and use port forwarding.
- **The FasterGS custom rasterizer is currently disabled** on both `hpg-turin` (sm_89) and `hpg-b200` (sm_100) because of an upstream launch-config bug. The fused Adam kernel works on both. See the comment block in `backend/scripts/hpg_gs_final_train.py` above the `sed` lines for the full story.

---

## Branch + testing notes

- The feature branch is **`jb/fastergs-opensplat-integration`**. Check it out before following any of the setup above — don't set up off `main`.
- **Don't merge to `main`** until the matrix results have been reviewed (planned for Saturday morning).
- Report setup friction promptly so these docs can improve. If something surprises you or the scripts/README.md lies about a command, raise it — fixing docs costs ~5 min each.
