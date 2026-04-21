import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { isStaticMode } from "./config/showcase";
import { installStaticApi } from "./utils/staticApi";
import "./index.css";

// In VITE_STATIC_MODE (the GitHub Pages build) there is no backend, so patch
// window.fetch to synthesize /api/datasets/* responses from the mirrored
// showcase files. Must run before the router mounts so the first effect in
// Gallery/Reports already sees the shim.
if (isStaticMode()) {
  installStaticApi();
}

// The GitHub Pages build is served under /<repo>/; BrowserRouter needs that
// prefix or every in-app link 404s. Locally and for any backend-backed deploy
// BASE_URL is just "/".
const routerBase = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter basename={routerBase}>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
