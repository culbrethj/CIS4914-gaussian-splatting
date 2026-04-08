import React from "react";
import GaussViewer from "../components/GaussViewer";
import { Link } from "react-router-dom";

export default function Gallery() {
  return (
    <main style={{padding:24}}>
      <h2>Hipergator Gallery</h2>
      <p>Gallery of splats / pointclouds.</p>

          <section style={{marginTop:16}}>
            <h3>Viewer</h3>
            <GaussViewer />
          </section>

      <Link to="/">Back</Link>
    </main>
  );
}