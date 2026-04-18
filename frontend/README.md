# Frontend

React + Vite single-page app for the Gaussian Splatting Studio. Talks to the
FastAPI backend in `../backend` and renders trained splats in the browser.

Main pages (under `src/pages/`):

- `/` — landing page
- `/demos` — upload a video and run the pipeline with a live log stream
- `/gallery` — browse pre-trained splats and compare two runs side-by-side
- `/reports` — charts of training metrics across runs
- `/documentation` — viewer controls and backend overview
- `/converter` — client-side PLY → .splat conversion

For install, run, and dev instructions (including how to start the backend),
see [../SETUP.md](../SETUP.md).
