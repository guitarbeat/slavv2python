import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  plugins: [react()],
  build: {
    outDir: "build",
    lib: {
      entry: "./src/index.tsx",
      formats: ["es"],
      fileName: "index-[hash]",
    },
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        assetFileNames: "style-[hash][extname]",
      },
    },
  },
});
