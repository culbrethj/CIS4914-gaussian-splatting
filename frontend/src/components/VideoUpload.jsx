import { useState } from 'react'

export default function VideoUpload() {
  const [video, setVideo] = useState(null);
  const [status, setStatus] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if(file && file.type.startsWith("video/")){
      setVideo(file);
    }
  };

  const handleUpload = async () => {
    if(!video) return;

    try {
      setStatus("Uploading your video...");
      const formData = new FormData();
      formData.append("video", video);
      const response = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await response.json();
      setStatus("✓ Upload complete!");
    } catch (err) {
      setStatus("✗ Upload failed. Please try again.");
    }
  };

  return (
    <div className="upload-card">
      <h2 className="upload-title">UPLOAD VIDEO</h2>
      <p className="upload-subtitle">Supported formats: MP4, MOV, AVI</p>

      <label className="upload-dropzone">
        {video ? `✓ ${video.name}` : "Click to select or drag & drop"}
        <input type="file" accept="video/*" onChange={handleFileChange} style={{ display: "none" }}/>
      </label>

      <button className="upload-button" onClick={handleUpload} disabled={!video}>
        Upload
      </button>

      {status && <p className="upload-status">{status}</p>}
    </div>
  );
}