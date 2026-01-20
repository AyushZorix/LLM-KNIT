import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: 'frontend', // Point to the frontend directory
  server: {
    port: 3001,
    strictPort: false,
    host: '0.0.0.0',
    hmr: {
      port: 3001
    },
    // Add compatibility for newer Node.js versions
    fs: {
      strict: false
    }
  },
  // Add build optimizations for Node.js v25 compatibility
  optimizeDeps: {
    include: ['react', 'react-dom']
  },
  define: {
    global: 'globalThis'
  }
})

