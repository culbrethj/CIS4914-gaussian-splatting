import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { existsSync, mkdirSync, copyFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

// Showcase datasets that ship with the GitHub Pages build. Must stay in sync
// with frontend/src/config/showcase.js (same names + run_tags). Two parallel
// lists is cheap vs. teaching Vite's Node-land config to consume an ESM
// source file that uses import.meta.env.
const SHOWCASE = [
  { name: 'cone', run_tag: 'cone_colmap_c9-vanilla_train_20260418_000034' },
  { name: 'snowboard', run_tag: 'snowboard_colmap_s1-baseline_train_20260420_181230' },
  { name: 'gator_statue', run_tag: 'gator_statue_s1-baseline_train_20260421_055348' },
]

// Vite plugin: at buildStart when VITE_STATIC_MODE=true, mirror each
// showcase dataset's committed splat.splat + metrics_summary.json +
// metrics.jsonl from backend/datasets/ into frontend/public/datasets/ so
// Vite picks them up as static assets in the Pages bundle.
// Metrics PNGs are deliberately skipped - the Reports page renders its own
// charts from the JSONL, so shipping the PNGs too would just bloat the
// artifact.
function copyShowcasePlugin() {
  return {
    name: 'copy-showcase-assets',
    buildStart() {
      if (process.env.VITE_STATIC_MODE !== 'true') return
      const repoRoot = resolve(__dirname, '..')
      const src = join(repoRoot, 'backend', 'datasets')
      const dst = resolve(__dirname, 'public', 'datasets')
      let copied = 0
      let missing = 0
      for (const { name, run_tag } of SHOWCASE) {
        const dsSrc = join(src, name)
        const dsDst = join(dst, name)
        const splatSrc = join(dsSrc, 'splat.splat')
        const splatDst = join(dsDst, 'splat.splat')
        if (existsSync(splatSrc)) {
          mkdirSync(dirname(splatDst), { recursive: true })
          copyFileSync(splatSrc, splatDst)
          copied++
        } else {
          this.warn(`[showcase] missing splat for ${name}: ${splatSrc}`)
          missing++
        }
        const metricsSrc = join(dsSrc, 'metrics', run_tag)
        const metricsDst = join(dsDst, 'metrics', run_tag)
        for (const fname of ['metrics_summary.json', 'metrics.jsonl']) {
          const f = join(metricsSrc, fname)
          const to = join(metricsDst, fname)
          if (existsSync(f)) {
            mkdirSync(dirname(to), { recursive: true })
            copyFileSync(f, to)
            copied++
          } else {
            this.warn(`[showcase] missing ${fname} for ${name}/${run_tag}`)
            missing++
          }
        }
      }
      this.info(`[showcase] copied ${copied} files, missing ${missing}`)
    },
  }
}

// https://vite.dev/config/
// The base path only shifts under VITE_STATIC_MODE because GitHub Pages
// serves the site from https://<user>.github.io/<repo>/. Local dev and any
// server-backed deploy keep the default "/" so Vite's proxy + absolute /api
// paths keep working unchanged.
export default defineConfig({
  base: process.env.VITE_STATIC_MODE === 'true' ? '/CIS4914-gaussian-splatting/' : '/',
  plugins: [react(), copyShowcasePlugin()],
  assetsInclude: ['**/*.ply', '**/*.splat'],
  server: {
    proxy: {
      // proxy HTTP and WebSocket requests under /api to your FastAPI backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        ws: true, // <-- enable websocket proxying
      },
      // allow frontend dev server to load backend static preview assets
      '/datasets': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    }
  }
})
