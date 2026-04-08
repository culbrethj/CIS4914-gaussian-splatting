import React from "react";
import { Link } from "react-router-dom";
import GaussViewer from "../components/GaussViewer";
import VideoUpload from "../components/VideoUpload";

export default function LiveDemos() {
  return (
    <main style={{padding:24}}>
      <h2>Live Demos</h2>
      <p>Demo page moved here.</p>
      
      <section style={{marginTop:16}}>
        <h3>Upload</h3>
        <VideoUpload />
      </section>

      <section style={{marginTop:16}}>
        <h3>Viewer</h3>
        <GaussViewer />
      </section>

      <div style={{marginTop:24}}>
        <Link to="/">Back</Link>
      </div>
    </main>
  );
}