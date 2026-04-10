import React from "react";
import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <main style={s.page}>
      <div style={s.hero}>
        <h1 style={s.title}>Gaussian Splatting</h1>
        <p style={s.sub}>Upload a video. Get a photorealistic 3D scene.</p>
        <Link to="/demos" style={s.cta}>Try the Live Demo →</Link>
      </div>

      <div style={s.grid}>
        <Link to="/docs" style={s.card}>
          <span style={s.cardTitle}>Documentation</span>
          <span style={s.cardDesc}>Setup guides and pipeline details</span>
        </Link>
        <Link to="/reports" style={s.card}>
          <span style={s.cardTitle}>Reports</span>
          <span style={s.cardDesc}>Benchmarks and experiment results</span>
        </Link>
        <Link to="/gallery" style={s.card}>
          <span style={s.cardTitle}>Gallery</span>
          <span style={s.cardDesc}>Pre-built splats from HiperGator</span>
        </Link>
      </div>

      <div style={s.flowWrap}>
        <div style={s.flowHeader}>
          <h2 style={s.flowTitle}>How it works</h2>
          <p style={s.flowSub}>
            A simple pipeline from video to interactive 3D scene
          </p>
        </div>

        <div style={s.flow}>
          {[
            ["01", "Upload", "Drop in video footage from a phone or camera"],
            ["02", "Process", "Frames are extracted and filtered"],
            ["03", "Reconstruct", "SfM + Gaussian Splatting build the scene"],
            ["04", "Explore", "View and interact in real time"],
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
      </div>
    </main>
  );
}

const s = {
  page: {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 64,
    padding: "60px 24px",
    background: "#f9f9f9",
    fontFamily: "'Helvetica Neue', Arial, sans-serif",
  },
  hero: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 16,
    textAlign: "center",
  },
  title: {
    fontSize: 40,
    fontWeight: 700,
    color: "#111",
    margin: 0,
    letterSpacing: "-0.02em",
  },
  sub: {
    fontSize: 16,
    color: "#666",
    margin: 0,
  },
  cta: {
    marginTop: 8,
    display: "inline-block",
    background: "#111",
    color: "#fff",
    textDecoration: "none",
    padding: "12px 28px",
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 500,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 16,
    width: "100%",
    maxWidth: 720,
  },
  card: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
    background: "#fff",
    border: "1px solid #e5e5e5",
    borderRadius: 12,
    padding: "20px 20px",
    textDecoration: "none",
    color: "#111",
  },
  cardTitle: {
    fontSize: 14,
    fontWeight: 600,
  },
  cardDesc: {
    fontSize: 13,
    color: "#888",
  },
  flowWrap: {
  width: "100%",
  maxWidth: 980,
  background: "#ffffff",
  border: "1px solid #e8e8e8",
  borderRadius: 20,
  padding: "32px 24px",
  boxShadow: "0 12px 40px rgba(0,0,0,0.05)",
},

flowHeader: {
  textAlign: "center",
  marginBottom: 28,
},

flowTitle: {
  margin: 0,
  fontSize: 22,
  fontWeight: 600,
  color: "#111",
},

flowSub: {
  margin: "8px 0 0 0",
  fontSize: 14,
  color: "#777",
},

flow: {
  display: "grid",
  gridTemplateColumns: "1fr auto 1fr auto 1fr auto 1fr",
  alignItems: "center",
  gap: 14,
},

flowCard: {
  display: "flex",
  flexDirection: "column",
  gap: 10,
  minHeight: 150,
  padding: "18px 16px",
  borderRadius: 16,
  background: "#fafafa",
  border: "1px solid #ececec",
  transition: "all 0.2s ease",
},

flowTop: {
  display: "flex",
  justifyContent: "space-between",
},

flowBadge: {
  fontSize: 11,
  fontWeight: 700,
  color: "#666",
  background: "#f1f1f1",
  borderRadius: 999,
  padding: "5px 9px",
  letterSpacing: "0.08em",
},

flowCardTitle: {
  fontSize: 16,
  fontWeight: 600,
  color: "#111",
},

flowCardDesc: {
  fontSize: 13,
  color: "#777",
  lineHeight: 1.5,
},

flowArrow: {
  fontSize: 22,
  color: "#c5c5c5",
},

stepColors: [
  {
    borderTop: "3px solid #6366f1",
  },
  {
    borderTop: "3px solid #22c55e",
  },
  {
    borderTop: "3px solid #f59e0b",
  },
  {
    borderTop: "3px solid #ef4444",
  },
],
};