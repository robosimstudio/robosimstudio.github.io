import { defineConfig } from "vite";
import sirv from "sirv";

// Demo data never lives in git or in public/: it is built under RSS_DATA_DIR
// (default ~/rss_ws/robosimstudio-website/build/site-data) and served at /demos in
// dev/preview, then rsynced into dist/demos by scripts/deploy.sh.
const DATA = process.env.RSS_DATA_DIR ?? new URL("../build/site-data", import.meta.url).pathname;

function demoData() {
  const serve = sirv(DATA, { dev: true, etag: true });
  // Braces matter: Vite calls a function returned from configureServer as a post
  // hook, and connect's use() returns the app.
  const mount = (server: { middlewares: { use: (p: string, h: unknown) => void } }) => {
    server.middlewares.use("/robosimstudio-website/data", serve);
  };
  return { name: "rsw-demo-data", configureServer: mount, configurePreviewServer: mount };
}

export default defineConfig({
  base: "/robosimstudio-website/",
  plugins: [demoData()],
  build: { target: "es2022", sourcemap: true, chunkSizeWarningLimit: 900 },
  worker: { format: "es" },
  test: { include: ["tests/unit/**/*.test.ts"], environment: "node" },
});
