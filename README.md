# CIS4914 Project: End-to-end Gaussian Splatting Implementation

### Due Dates
- ~~**Feb 01: Project Proposal**~~
- ~~Feb 06: Week 4 reports~~
- ~~Feb 13: Week 5 reports~~
- ~~**Feb 15: Presentation 1 video**~~
- ~~Feb 20: Week 6 reports~~
- ~~Feb 27: Week 7 reports~~
- ~~Mar 06: Week 8 reports~~
- ~~Mar 13: Week 9 reports~~
- ~~**Mar 13: Presentation 2 video**~~
- ~~Mar 27: Week 11 reports~~
- ~~Apr 03: Week 12 reports~~
- ~~Apr 10: Week 13 reports~~
- **Apr 14: Senior Showcase**
- **Apr 21: Final Presentation video**

---

### Frontend Setup and Run

NodeJs is required (v20+). Check your Node version.

```bash
node -v
```

Install frontend dependencies and run.

```bash
cd frontend
npm install
```

Run the app.

```bash
npm run dev
```
Visit `http://localhost:5173` in browser.

--- 

### Backend Setup and Run

This is needed for the current video upload implementation. Python 3 is required.

Create and activate a virtual environment **from the repo root** (`requirements.txt` lives here, not under `backend/`):

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

Install backend dependencies (still from repo root):

```bash
pip install -r requirements.txt
```

Then run the server from the `backend/` dir:

```bash
cd backend
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`.

---

### OpenSplat install

Follow platform instructions on [https://github.com/pierotofy/OpenSplat](https://github.com/pierotofy/OpenSplat)

Final binaries go in backend/binaries

---

### Training backends

The Live Demos page lets you pick between two training backends:

- **OpenSplat** — runs locally against the bundled binary under
  `backend/binaries/`. No GPU required (CPU or CUDA).
- **Faster-GS** — sends the job to HiPerGator and trains there on a GPU
  partition. This is the default and tends to be more reliable.

OpenSplat is simpler to set up (no HPG account needed) but has had
dylib/runtime issues on some machines. Faster-GS needs an HPG account and
an SSH alias `hpg`, but gives you the full pipeline with metrics + charts
on the Reports page.

Full Faster-GS setup instructions (HPG workspace layout, camera-model
undistort step, troubleshooting) live in
[backend/experiments/faster-gs/README.md](backend/experiments/faster-gs/README.md).
