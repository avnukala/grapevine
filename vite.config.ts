import path from "path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const api = `http://localhost:${process.env.PORT ?? 8790}`;

// The Python backend owns all server work. Vite only builds and serves the client.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": api,
      "/ws": { target: api.replace("http", "ws"), ws: true },
    },
  },
});
