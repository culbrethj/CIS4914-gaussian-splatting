import React from "react";
import { Link } from "react-router-dom";

export default function Reports() {
  return (
    <main style={{padding:24}}>
      <h2>Reports / Metrics</h2>
      <p>Place charts, CSV viewers, and generated reports here.</p>
      <Link to="/">Back</Link>
    </main>
  );
}