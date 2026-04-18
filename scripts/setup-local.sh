#!/usr/bin/env bash
# setup-local.sh — first-time + repeat setup for a Mac/Linux checkout.
#
# What it does:
#   1. Verifies Python 3.10+, Node 20+, git are on PATH.
#   2. Creates ./venv (at repo root) if missing, activates it, installs
#      requirements.txt.
#   3. Runs `npm install` inside frontend/.
#   4. Checks for an `hpg` alias in ~/.ssh/config. Prints a one-paragraph
#      how-to if missing; does NOT block setup (OpenSplat-only runs
#      don't need HPG).
#   5. Prints the next-step commands (venv activation + backend/frontend).
#
# Safe properties:
#   - set -e so any failure stops the script.
#   - Idempotent: re-running skips the venv creation / npm install when
#     they've already completed successfully.
#   - Never needs sudo.
#   - Never touches system-level tools. All installs go into ./venv or
#     frontend/node_modules.

set -euo pipefail

# --- pretty logging ---------------------------------------------------------
BOLD="$(printf '\033[1m')"
GREEN="$(printf '\033[32m')"
YELLOW="$(printf '\033[33m')"
RED="$(printf '\033[31m')"
DIM="$(printf '\033[2m')"
RESET="$(printf '\033[0m')"

step()  { printf '%s==> %s%s\n' "${BOLD}${GREEN}" "$*" "${RESET}"; }
warn()  { printf '%s!!  %s%s\n' "${BOLD}${YELLOW}" "$*" "${RESET}"; }
fail()  { printf '%sxx  %s%s\n' "${BOLD}${RED}"    "$*" "${RESET}" >&2; exit 1; }
info()  { printf '    %s%s%s\n' "${DIM}"          "$*" "${RESET}"; }

# --- locate repo root (the dir ABOVE scripts/) ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
step "Repo root: ${REPO_ROOT}"

# --- 1. prerequisite checks -------------------------------------------------
step "Checking prerequisites"

command -v git >/dev/null 2>&1 \
  || fail "git not found on PATH. Install git (https://git-scm.com/) and re-run."
info "git: $(git --version)"

# Pinned requirements.txt (open3d 0.19.0, numpy 2.4.2, …) has wheels for
# Python 3.10–3.13 today; 3.14 isn't supported upstream yet. If the
# default `python3` on PATH is too new we try to fall back to a
# versioned binary, then fail with a clear message.
pick_python() {
  # $1 = candidate interpreter name (e.g. "python3.11"). Echoes the path
  # if the interpreter exists AND its version is 3.10–3.13 inclusive.
  local candidate="$1"
  command -v "${candidate}" >/dev/null 2>&1 || return 1
  local v
  v="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" || return 1
  local mj="${v%%.*}"
  local mn="${v##*.}"
  [ "${mj}" = "3" ] && [ "${mn}" -ge 10 ] && [ "${mn}" -le 13 ] || return 1
  echo "${candidate}"
}

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if PYTHON_BIN="$(pick_python "${candidate}")"; then
    break
  fi
done

if [ -z "${PYTHON_BIN}" ]; then
  CURRENT="$(python3 --version 2>/dev/null || echo 'not installed')"
  fail "Need Python 3.10–3.13 on PATH. Found: ${CURRENT}. Install a supported version (pyenv, brew install python@3.12, or python.org) and re-run. Pinned deps (open3d, numpy, pandas) don't yet ship wheels for 3.14+."
fi
info "python: $(${PYTHON_BIN} --version) -> $(command -v ${PYTHON_BIN})"

if ! command -v node >/dev/null 2>&1; then
  fail "node not found on PATH. Install Node 20+ via nvm (recommended) or https://nodejs.org/ and re-run."
fi

# Node prints like "v20.11.0"; strip the leading v and keep major.
NODE_FULL="$(node --version)"
NODE_MAJOR="${NODE_FULL#v}"
NODE_MAJOR="${NODE_MAJOR%%.*}"
if [ "${NODE_MAJOR}" -lt 20 ]; then
  fail "Node 20+ required; found ${NODE_FULL}. Upgrade (nvm: 'nvm install 20 && nvm use 20') and re-run."
fi
info "node: ${NODE_FULL}"

if ! command -v npm >/dev/null 2>&1; then
  fail "npm not found on PATH (should ship with Node). Reinstall Node and re-run."
fi
info "npm: $(npm --version)"

# --- 2. python venv + pip install -------------------------------------------
VENV_DIR="${REPO_ROOT}/venv"
VENV_PY="${VENV_DIR}/bin/python"

if [ -x "${VENV_PY}" ]; then
  step "Reusing existing venv at ./venv"
else
  step "Creating venv at ./venv (using ${PYTHON_BIN})"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# Activate the venv for the rest of this script. Using the activate
# script (vs. plain path-prefixing) so child tools see the expected
# VIRTUAL_ENV env var.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
info "venv python: $(python -c 'import sys; print(sys.executable)')"

step "Upgrading pip inside venv"
python -m pip install --upgrade pip >/dev/null

step "Installing Python requirements (requirements.txt)"
# Capture the requirements.txt mtime so we can fingerprint the "last
# installed" state and skip this step on repeat runs when nothing's
# changed. Stamp lives inside the venv so it's per-checkout.
STAMP="${VENV_DIR}/.requirements.stamp"
REQ_MTIME="$(stat -f %m requirements.txt 2>/dev/null || stat -c %Y requirements.txt)"
PREV_STAMP="$(cat "${STAMP}" 2>/dev/null || echo 0)"

if [ "${PREV_STAMP}" = "${REQ_MTIME}" ]; then
  info "requirements.txt unchanged since last install — skipping pip install"
else
  python -m pip install -r requirements.txt
  echo "${REQ_MTIME}" > "${STAMP}"
fi

# --- 3. npm install ---------------------------------------------------------
step "Installing frontend dependencies (npm install)"
cd "${REPO_ROOT}/frontend"
# npm ci would be faster but fails if the lockfile is missing or out of
# sync — npm install is more forgiving for teammates who may rebase.
if [ -d node_modules ] && [ -f package-lock.json ] \
   && [ node_modules -nt package-lock.json ]; then
  info "node_modules newer than package-lock.json — skipping npm install"
else
  npm install
fi
cd "${REPO_ROOT}"

# --- 4. SSH alias check -----------------------------------------------------
step "Checking for 'hpg' SSH alias"
SSH_CONFIG="${HOME}/.ssh/config"
if [ -f "${SSH_CONFIG}" ] && grep -qE '^\s*Host\s+.*\bhpg\b' "${SSH_CONFIG}"; then
  info "Found 'hpg' Host entry in ${SSH_CONFIG}"
else
  warn "No 'hpg' SSH alias found in ${SSH_CONFIG}"
  cat <<'HINT'
    Add a block like this to ~/.ssh/config to enable HPG pipeline steps:

      Host hpg
        HostName hpg.rc.ufl.edu
        User your-gatorlink
        IdentityFile ~/.ssh/id_ed25519
        ControlMaster auto
        ControlPath ~/.ssh/cm_%C.sock
        ControlPersist 8h

    Then run `ssh hpg` once from your terminal to authenticate (Duo push,
    key unlock) so the persistent mux is live. The pipeline scripts
    piggyback on that mux for ssh/scp/rsync calls.

    Skip this only if you plan to run OpenSplat locally and never touch
    Faster-GS / the experiment matrix.
HINT
fi

# --- 5. summary -------------------------------------------------------------
step "Setup complete"
cat <<NEXT

Next steps (run these in separate terminals):

  # Activate the Python venv (required before using backend tools):
  source venv/bin/activate

  # Start the backend API (FastAPI + uvicorn, port 8000):
  cd backend && uvicorn main:app --reload

  # Start the frontend dev server (Vite, port 5173):
  cd frontend && npm run dev

  # Open the UI:
  open http://localhost:5173      # macOS
  xdg-open http://localhost:5173  # Linux

For full details, platform notes, and troubleshooting, see SETUP.md.
NEXT
