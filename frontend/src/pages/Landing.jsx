import React, { useState } from "react";
import { Link } from "react-router-dom";
import gaussGuide from '../assets/gauss_guide.gif';

export default function Landing() {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <main style={s.page}>
      <section style={s.hero}>
        <h1 style={s.title}>Gaussian Splatting</h1>
        <p style={s.sub}>Upload a video. Get a photorealistic 3D scene.</p>
        <Link to="/demos" style={s.cta}>Try the Live Demo →</Link>
      </section>

      <section style={s.grid}>
        <Link to="/docs" style={s.card}>
          <span style={s.cardTitle}>Documentation</span>
          <span style={s.cardDesc}>
            What the pipeline does, how the viewer renders splats, and what runs on CPU vs GPU.
          </span>
        </Link>
        <Link to="/reports" style={s.card}>
          <span style={s.cardTitle}>Reports</span>
          <span style={s.cardDesc}>
            Per-run metrics, overlay comparisons, and Shorter-Splatting experiment results.
          </span>
        </Link>
        <Link to="/gallery" style={s.card}>
          <span style={s.cardTitle}>Gallery</span>
          <span style={s.cardDesc}>
            Pre-trained splats from HiPerGator. Compare two scenes side-by-side.
          </span>
        </Link>
      </section>

      <section style={s.flowWrap}>
        <div style={s.flowHeader}>
          <h2 style={s.flowTitle}>How it works</h2>
          <p style={s.flowSub}>
            A simple pipeline from video to interactive 3D scene.
          </p>
        </div>

        <div style={s.flow}>
          {[
            ["01", "Upload", "Drop in video footage from a phone or camera"],
            ["02", "Process", "Frames are extracted, filtered, and deduplicated"],
            ["03", "Reconstruct", "SfM + Gaussian Splatting train the model"],
            ["04", "Explore", "View and interact with the scene in real time"],
          ].map(([n, title, desc], i, arr) => (
            <React.Fragment key={n}>
              <div style={{ ...s.flowCard, ...s.stepColors[i] }}>
                <div style={s.flowTop}>
                  <span style={s.flowBadge}>{n}</span>
                </div>
                <span style={s.flowCardTitle}>{title}</span>
                <span style={s.flowCardDesc}>{desc}</span>
              </div>

              {i < arr.length - 1 && <div style={s.flowArrow}>→</div>}
            </React.Fragment>
          ))}
        </div>

        <div style={s.showMoreSection}>
          <button onClick={() => setIsVisible(!isVisible)} style={s.showMoreBtn}>
            {isVisible ? "Hide example" : "Show example output"}
          </button>

          {isVisible && (
            <div style={s.expandedAssetWrap}>
              <img src={gaussGuide} style={s.expandedImg} alt="Example gaussian splatting output" />
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

const s = {
  page: {
    // Shallow top padding so the hero is visible on a normal laptop screen
    // without scrolling. Content flows top-down; no 100vh vertical centering.
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 36,
    padding: "32px 24px 64px",
    background: "#f8f9fb",
    fontFamily: "'Helvetica Neue', Arial, sans-serif",
    color: "#111",
  },
  hero: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 14,
    textAlign: "center",
    maxWidth: 760,
  },
  title: {
    fontSize: 44,
    fontWeight: 700,
    color: "#0f172a",
    margin: 0,
    letterSpacing: "-0.025em",
    lineHeight: 1.05,
  },
  sub: {
    fontSize: 17,
    color: "#475569",
    margin: 0,
    lineHeight: 1.5,
  },
  cta: {
    marginTop: 6,
    display: "inline-block",
    background: "#0f172a",
    color: "#fff",
    textDecoration: "none",
    padding: "12px 28px",
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 600,
    letterSpacing: "0.01em",
    transition: "transform 0.15s ease, box-shadow 0.15s ease",
    boxShadow: "0 4px 12px rgba(15, 23, 42, 0.18)",
  },
  grid: {
    // Cards stretch the full container width so the row reads as one unit
    // instead of three small tiles floating in whitespace.
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 16,
    width: "100%",
    maxWidth: 980,
  },
  card: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    background: "#fff",
    border: "1px solid #e5e8ee",
    borderRadius: 14,
    padding: "20px 22px",
    textDecoration: "none",
    color: "#0f172a",
    transition: "border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease",
    boxShadow: "0 1px 2px rgba(15, 23, 42, 0.03)",
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 700,
    letterSpacing: "-0.005em",
  },
  cardDesc: {
    fontSize: 13.5,
    color: "#64748b",
    lineHeight: 1.5,
  },
  flowWrap: {
    width: "100%",
    maxWidth: 980,
    background: "#ffffff",
    border: "1px solid #e5e8ee",
    borderRadius: 20,
    padding: "28px 24px 26px",
    boxShadow: "0 12px 40px rgba(15, 23, 42, 0.06)",
  },

  flowHeader: {
    textAlign: "center",
    marginBottom: 22,
  },

  flowTitle: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: "#0f172a",
  },

  flowSub: {
    margin: "6px 0 0",
    fontSize: 14,
    color: "#64748b",
  },

  flow: {
    display: "grid",
    gridTemplateColumns: "1fr auto 1fr auto 1fr auto 1fr",
    alignItems: "center",
    gap: 12,
  },

  flowCard: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    minHeight: 140,
    padding: "16px 14px",
    borderRadius: 14,
    background: "#fafbfd",
    border: "1px solid #ececf2",
    transition: "all 0.2s ease",
  },

  flowTop: {
    display: "flex",
    justifyContent: "space-between",
  },

  flowBadge: {
    fontSize: 11,
    fontWeight: 700,
    color: "#475569",
    background: "#eef2f7",
    borderRadius: 999,
    padding: "4px 9px",
    letterSpacing: "0.08em",
  },

  flowCardTitle: {
    fontSize: 15,
    fontWeight: 700,
    color: "#0f172a",
  },

  flowCardDesc: {
    fontSize: 13,
    color: "#64748b",
    lineHeight: 1.5,
  },

  flowArrow: {
    fontSize: 18,
    color: "#c5c9d3",
  },

  stepColors: [
    { borderTop: "3px solid #6366f1" },
    { borderTop: "3px solid #22c55e" },
    { borderTop: "3px solid #f59e0b" },
    { borderTop: "3px solid #ef4444" },
  ],
  showMoreSection: {
    paddingTop: 22,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 18,
  },
  showMoreBtn: {
    background: "#f5f5f7",
    border: "1px solid #e5e5e7",
    color: "#1d1d1f",
    padding: "8px 16px",
    borderRadius: 99,
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.2s ease",
  },
  expandedAssetWrap: {
    width: "100%",
    overflow: "hidden",
    borderRadius: 12,
    border: "1px solid #ececec",
    animation: "fadeIn 0.4s ease-out",
  },
  expandedImg: {
    width: "100%",
    display: "block",
    height: "auto",
  },
};
