import { useState } from 'react'
import * as SPLAT from "gsplat";

export default function Converter() {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleUploadClick = async () => {
        if (!selectedFile) return;

        try {
            const scene = new SPLAT.Scene()
            await SPLAT.PLYLoader.LoadFromFileAsync(selectedFile, scene);
            scene.saveToFile(selectedFile.name.replace(".ply", ".splat"));
        } catch (error) {
            console.error("PLY to SPLAT conversion error:", error);
            alert("Conversion failed: " + error.message);
        }
    };

    return (
        <>
            <input type="file" accept=".ply" required onChange={handleFileChange}/>
            <button onClick={handleUploadClick}>Upload/Render</button>
        </>
    )
}
