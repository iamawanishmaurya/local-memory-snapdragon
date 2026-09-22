/// <reference types="vitest/config" />
import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import { playwright } from '@vitest/browser-playwright'

// https://vite.dev/config/
export default defineConfig({
  // Relative base so the built bundle is served by the local FastAPI server
  // (http://127.0.0.1:8787) with no external asset hosting.
  base: './',
  server: {
    proxy: {
      // Dev convenience: proxy API calls to the running Local Memory backend.
      // The dev-only auth token (research §5) is injected from the
      // LOCAL_MEMORY_TOKEN env var — the one sanctioned override
      // (local_memory/config.py::auth_token). Production serves the token
      // invisibly inside index.html; nothing here affects the built app.
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
        headers: { Authorization: `Bearer ${process.env.LOCAL_MEMORY_TOKEN ?? ''}` },
      },
    },
  },
  plugins: [
    tanstackRouter({
      target: 'react',
      autoCodeSplitting: true,
    }),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    silent: 'passed-only',
    unstubEnvs: true,
    browser: {
      enabled: true,
      provider: playwright(),
      instances: [{ browser: 'chromium' }],
    },
    coverage: {
      // include: ['src/**/*.{js,jsx,ts,tsx}'], // Uncomment to expand the report to all src/**/* so untested modules appear as 0% coverage.
      exclude: [
        'src/components/ui/**',
        'src/assets/**',
        'src/tanstack-table.d.ts',
        'src/routeTree.gen.ts',
        'src/test-utils/**',
        'src/routes/**',
      ],
    },
  },
})
