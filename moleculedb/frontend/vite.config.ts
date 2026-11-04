import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Forward API calls to the FastAPI backend (uv run moleculedb) during dev.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
