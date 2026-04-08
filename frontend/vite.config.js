import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  assetsInclude: ['**/*.ply', '**/*.splat'],
  server: {
    proxy: {
      // proxy HTTP and WebSocket requests under /api to your FastAPI backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        ws: true, // <-- enable websocket proxying
      }
    }
  }
})