import React from "react";
import { Link } from "react-router-dom";

export default function Landing() {
  const containerStyle = {
    padding: 24,
    maxWidth: 900,
    margin: "0 auto",
    display: "flex",
    flexDirection: "column",
    gap: 24,
    alignItems: "center",
  };

  const gridStyle = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: 16,
    width: "100%",
  };

  const buttonStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "14px 18px",
    borderRadius: 12,
    background: "#1f6feb",
    color: "#fff",
    textDecoration: "none",
    fontWeight: 700,
    fontSize: 16,
    boxShadow: "0 6px 18px rgba(31,111,235,0.12)",
    minHeight: 56,
    transition: "transform 120ms ease, box-shadow 120ms ease",
  };

  const smallNoteStyle = {
    color: "#555",
    fontSize: 14,
    marginTop: 8,
    textAlign: "center",
  };

  return (
    <main style={containerStyle}>
      <div style={{ textAlign: "center" }}>
        <h1 style={{ margin: 0 }}>Gaussian Splatting — Demo Hub</h1>
        <p style={{ marginTop: 8, color: "#444" }}>
          Pick a section to continue.
        </p>
      </div>

      <div style={gridStyle}>
        <Link to="/docs" style={buttonStyle} aria-label="Documentation">
          Documentation
        </Link>

        <Link to="/reports" style={buttonStyle} aria-label="Reports">
          Reports
        </Link>

        <Link to="/demos" style={buttonStyle} aria-label="Live Demos">
          Live Demos
        </Link>

        <Link to="/gallery" style={buttonStyle} aria-label="Gallery">
          Hipergator Gallery
        </Link>
      </div>

      <div style={smallNoteStyle}>
        most of this is temporary boilerplate, wanted to quickly organize our page layout. Feel free to redesign this page, we should probably add some basic info here as well like "About" 
      </div>
    </main>
  );
}