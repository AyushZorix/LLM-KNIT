import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: 'frontend', // Point to the frontend directory
  server: {
    port: 3001, // Changed to port 3001
    strictPort: false,
    host: '0.0.0.0', // Allow external connections
  },
})

