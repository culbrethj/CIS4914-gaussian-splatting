import React, { useState, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { Splat, OrbitControls } from "@react-three/drei";

export default function GaussViewer({ datasetName = null, splatApiPath = null }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loadedGaussModel, setLoadedGaussModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleLocalUpload = () => {
        if (!selectedFile) return;
        const gaussModelURL = URL.createObjectURL(selectedFile);
        setLoadedGaussModel(gaussModelURL);
        setError(null);
    };

    // load .splat from backend dataset via API proxy (served at /api/datasets/{name}/splat)
    async function loadSplatFromDataset(name) {
        if (!name) return;
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`/api/datasets/${encodeURIComponent(name)}/splat`);
            if (!res.ok) throw new Error(`failed to fetch splat file: ${res.status}`);
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);

            setLoadedGaussModel((prev) => {
                if (prev && typeof prev === "string" && prev.startsWith("blob:")) {
                    try { URL.revokeObjectURL(prev); } catch {}
                }
                return url;
            });
        } catch (e) {
            setError(String(e));
        } finally {
            setLoading(false);
        }
    }

    // load .splat from an explicit API path (e.g. /api/hpg/<file>/splat)
    async function loadSplatFromPath(path) {
        if (!path) return;
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(path);
            if (!res.ok) throw new Error(`failed to fetch splat file: ${res.status}`);
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);

            setLoadedGaussModel((prev) => {
                if (prev && typeof prev === "string" && prev.startsWith("blob:")) {
                    try { URL.revokeObjectURL(prev); } catch {}
                }
                return url;
            });
        } catch (e) {
            setError(String(e));
        } finally {
            setLoading(false);
        }
    }

    // auto-load when datasetName or splatApiPath prop provided/changes
    useEffect(() => {
        if (splatApiPath) {
            loadSplatFromPath(splatApiPath);
        } else if (datasetName) {
            loadSplatFromDataset(datasetName);
        }
    }, [datasetName, splatApiPath]);

    // revoke blob URL on unmount
    useEffect(() => {
        return () => {
            if (loadedGaussModel && typeof loadedGaussModel === "string" && loadedGaussModel.startsWith('blob:')) {
                try { URL.revokeObjectURL(loadedGaussModel); } catch {}
            }
        };
    }, [loadedGaussModel]);

    return (
        <>
            {/* <div style={{ marginBottom: 8 }}>
                <div style={{ marginBottom: 6 }}>
                    <strong>Load .splat</strong>
                </div>

                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="file" accept=".splat" onChange={handleFileChange}/>
                    <button onClick={handleLocalUpload}>Render local .splat</button>

                    {datasetName ? (
                        <button onClick={() => loadSplatFromDataset(datasetName)} disabled={loading} style={{ marginLeft: 8 }}>
                            {loading ? "Loading..." : `Load from dataset: ${datasetName}`}
                        </button>
                    ) : null}

                    {splatApiPath ? (
                        <div style={{ color: "#666", marginLeft: 8, fontSize: 13 }}>
                            {loading ? "Loading remote .splat..." : splatApiPath}
                        </div>
                    ) : null}
                </div>

                {error && <div style={{ color: "red", marginTop: 8 }}>{error}</div>}
            </div> */}

            <div style={{ width: "100%", height: 480 }}>
                <Canvas style={{ width: "100%", height: "100%" }}>
                    <OrbitControls />
                    {loadedGaussModel ? (
                        <Splat
                            key={loadedGaussModel}
                            src={loadedGaussModel}
                            rotation={[0.75 * Math.PI, -.02 * Math.PI, .55 * Math.PI]}
                        />
                    ) : null}
                </Canvas>
            </div>
        </>
    );
}