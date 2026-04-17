import React from "react";
import { Routes, Route } from "react-router-dom";
import GaussViewer from './components/GaussViewer'
import VideoUpload from './components/VideoUpload'
import Converter from './components/Converter'
import NavBar from "./components/NavBar";
import Landing from "./pages/Landing";
import Documentation from "./pages/Documentation";
import Reports from "./pages/Reports";
import LiveDemos from "./pages/LiveDemos";
import Gallery from "./pages/Gallery";
import "./App.css";

// NavBar rendered at the app root so every route gets a consistent top nav.
// Replaces the per-page "Back" buttons/links that used to live inside each
// page's content.
export default function App() {
  return (
    <div className="app-root">
      <NavBar />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/docs" element={<Documentation />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/demos/*" element={<LiveDemos />} />
        <Route path="/gallery" element={<Gallery />} />
        <Route path="/converter" element={<Converter />} />
        <Route path="/upload" element={<VideoUpload />} />
        <Route path="/viewer" element={<GaussViewer />} />
      </Routes>
    </div>
  );
}