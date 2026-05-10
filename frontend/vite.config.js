import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev`, the React dev server runs on http://localhost:5173
// and proxies API calls to the FastAPI server on http://localhost:8000.
// In production (Docker / Cloud Run), the FastAPI server serves the
// built `dist/` folder, so no proxy is needed.
const API_TARGET = process.env.VITE_API_TARGET || "http://localhost:8000";

const apiPaths = ["/health", "/listings", "/predict", "/flag_overpriced", "/rank"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      apiPaths.map((p) => [p, { target: API_TARGET, changeOrigin: true }])
    ),
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
