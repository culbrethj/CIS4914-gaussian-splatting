import { useState } from 'react'

export default function VideoUpload() {
  const [video, setVideo] = useState(null);
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if(file && file.type.startsWith("video/")){
      setVideo(file);
      setStatus("");
      setProgress(null);
    }
  };

  // fetch() does not expose upload progress events. XMLHttpRequest does,
  // so use it for the upload and translate the .progress event into a
  // React state update.
  const handleUpload = () => {
    if (!video) return;
    setStatus("");
    setProgress(0);
    setUploading(true);

    const formData = new FormData();
    formData.append("video", video);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        setProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      setUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        setProgress(100);
        setStatus("Upload complete.");
      } else {
        setStatus(`Upload failed: HTTP ${xhr.status}`);
      }
    };

    xhr.onerror = () => {
      setUploading(false);
      setStatus("Upload failed: network error.");
    };

    xhr.send(formData);
  };

  return (
    <div className="upload-card">
      <h2 className="upload-title">UPLOAD VIDEO</h2>
      <p className="upload-subtitle">Supported formats: MP4, MOV, AVI</p>

      <label className="upload-dropzone">
        {video ? video.name : "Click to select or drag & drop"}
        <input type="file" accept="video/*" onChange={handleFileChange} style={{ display: "none" }}/>
      </label>

      <button className="upload-button" type="button" onClick={handleUpload} disabled={!video || uploading}>
        {uploading && progress !== null ? `Uploading ${progress}%` : "Upload"}
      </button>

      {uploading && progress !== null ? (
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress}
          style={{
            marginTop: "1rem",
            height: 8,
            background: "#e2e8f0",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${progress}%`,
              height: "100%",
              background: "linear-gradient(135deg, #3182ce, #63b3ed)",
              transition: "width 120ms ease-out",
            }}
          />
        </div>
      ) : null}

      {status && <p className="upload-status">{status}</p>}
    </div>
  );
}
