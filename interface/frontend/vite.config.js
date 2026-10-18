import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// interface/assets (samples, figures, team photos, PDFs) is served as the public folder.
export default defineConfig({
  plugins: [react()],
  publicDir: '../assets',
  server: { port: 5173 },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          motion: ['framer-motion'],
          charts: ['recharts'],
        },
      },
    },
  },
})
