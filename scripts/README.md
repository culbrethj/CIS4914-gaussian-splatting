# `scripts/` — setup + tooling

## `setup-local.sh`

One-command bootstrap for a Mac / Linux checkout.

**Run it from the repo root:**

```bash
./scripts/setup-local.sh
```

**What it does, in order:**

1. Verifies a Python interpreter in the 3.10–3.13 range is on PATH (tries `python3.13` → `python3.12` → `python3.11` → `python3.10` → `python3`), plus `node` ≥ 20 and `git`. Fails with a specific install hint if any are missing. 3.14 is too new for the pinned `open3d` / `numpy` wheels.
2. Creates a Python venv at `./venv` (repo root) if it doesn't exist, activates it, and `pip install -r requirements.txt`. Skips the pip install on repeat runs when `requirements.txt` hasn't changed (stamp file stored in the venv).
3. Runs `npm install` in `frontend/` if `package-lock.json` has been touched more recently than `node_modules/`. Otherwise skips.
4. Greps `~/.ssh/config` for an `hpg` alias; prints a sample config block + setup hint if missing. **Does not block setup** — users who only want to run OpenSplat locally don't need HPG.
5. Prints the next-step commands (venv activation, backend, frontend) and a pointer to `SETUP.md`.

**Idempotent.** Safe to re-run whenever; steps that already completed are skipped.

**Never requires `sudo`.** All installs go into `./venv` and `frontend/node_modules`.

## Known limitations

- **Mac / Linux only.** The script uses `bash`-isms (`set -euo pipefail`, `source bin/activate`, unix paths). Windows users should follow the manual section in `SETUP.md`.
- **Does not touch HPG.** Conda env on HiPerGator (`fastergs_cuda128`) is set up separately via the training script's `--stage setup` flow; the local script only cares about local dev prerequisites. See `backend/experiments/faster-gs/README.md` for HPG env details.
- **Does not install system-level tools.** If Python / Node / git aren't installed, the script fails with an install hint and exits — it won't try to install them for you.

## Adding new scripts

Anything tooling-adjacent that a user would invoke by hand belongs here. Keep each script self-documenting (top-of-file comment block describing what it does, required env vars, and safe-to-re-run semantics).
