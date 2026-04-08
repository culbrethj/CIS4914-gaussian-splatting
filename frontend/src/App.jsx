import React from "react";
import { Routes, Route } from "react-router-dom";
import GaussViewer from './components/GaussViewer'
import VideoUpload from './components/VideoUpload'
import Landing from "./pages/Landing";
import Documentation from "./pages/Documentation";
import Reports from "./pages/Reports";
import LiveDemos from "./pages/LiveDemos";
import Gallery from "./pages/Gallery";
import "./App.css";

export default function App() {
  return (
    <div className="app-root">
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/docs" element={<Documentation />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/demos/*" element={<LiveDemos />} />
        <Route path="/gallery" element={<Gallery />} />
      </Routes>
    </div>
  );
}