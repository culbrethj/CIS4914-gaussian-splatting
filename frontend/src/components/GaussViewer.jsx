import { useState, useEffect } from 'react'
import { Canvas } from "@react-three/fiber"
import { Splat, OrbitControls } from "@react-three/drei"

  // .ply -> .splat converter: https://huggingface.co/spaces/dylanebert/ply-to-splat
  // .splat files only, .ply not supported

  // Left-Click and drag to rotate
  // Right-Click and drag to pan
  // Scroll to zoom in/out

export default function GaussViewer() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loadedGaussModel, setLoadedGaussModel] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleUploadClick = () => {
        if (!selectedFile) return;

        // Load the selected file in the 3D renderer (blob url generated).
        const gaussModelURL = URL.createObjectURL(selectedFile);
        setLoadedGaussModel(gaussModelURL);
    };

    useEffect( () => { // Upon new file loaded, wipe previous loaded model's blob url, if any.
        return () => {
            if (loadedGaussModel && loadedGaussModel.startsWith('blob:')) { 
                URL.revokeObjectURL(loadedGaussModel);
            }
        }
    }, [loadedGaussModel]);

    return (
        <>
            <input type="file" accept=".splat" required onChange={handleFileChange}/>
            <button onClick={handleUploadClick}>Upload/Render</button>

            <Canvas>
                <OrbitControls />
                <Splat
                key={loadedGaussModel} // forces component cleanup when model changes
                src={loadedGaussModel}
                rotation={[0.75 * Math.PI, -.02 * Math.PI, .55 * Math.PI]} // optional: initial orientation
                />
            </Canvas>
        </>
    )
}