import React, { useEffect, useState } from "react";
import GaussViewer from "../components/GaussViewer";
import { Link } from "react-router-dom";

export default function Gallery() {
  const [splats, setSplats] = useState([]);
  const [selectedApiPath, setSelectedApiPath] = useState(null);

  useEffect(() => {
    fetch("/api/hpg/splats")
      .then((r) => r.json())
      .then((list) => setSplats(list || []))
      .catch((e) => {
        console.error("hpg splats fetch failed", e);
        setSplats([]);
      });
  }, []);

  return (
    <main style={{ padding: 24 }}>
      <h2>Hipergator Gallery</h2>
      <p>Gallery of splats / pointclouds.</p>

      <section style={{ marginTop: 16 }}>
        <h3>Choose a splat</h3>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {splats.length === 0 ? (
            <div style={{ color: "#666" }}>No .splat files found in hpg</div>
          ) : (
            <select
              value={selectedApiPath || ""}
              onChange={(e) => setSelectedApiPath(e.target.value || null)}
              style={{ padding: "8px 10px", borderRadius: 8 }}
            >
              <option value="">-- select splat --</option>
              {splats.map((s) => (
                <option key={s.filename} value={s.api_path}>
                  {s.name}
                </option>
              ))}
            </select>
          )}
        </div>

        <div style={{ marginTop: 16 }}>
          <h3>Viewer</h3>
          <div
            style={{
              padding: 12,
              borderRadius: 12,
              background: "#fff",
              boxShadow: "0 4px 10px rgba(0,0,0,0.06)",
            }}
          >
            <GaussViewer splatApiPath={selectedApiPath} />
          </div>
        </div>
      </section>

      <Link to="/">Back</Link>
    </main>
  );
}