import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React app runs on :5173 and proxies /api to the Express server on :8787,
// so the browser never sees the Anthropic API key.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8787",
    },
  },
});
