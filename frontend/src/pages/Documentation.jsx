import React from "react";
import { Link } from "react-router-dom";

export default function Documentation() {
  return (
    <main style={{padding:24}}>
      <h2>Documentation</h2>
      <p>Project docs go here. Use markdown files converted to pages or embed static docs.</p>
      <Link to="/">Back</Link>
    </main>
  );
}