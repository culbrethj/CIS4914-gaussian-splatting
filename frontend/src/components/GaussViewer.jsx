import React, { useState, useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Splat, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

export default function GaussViewer({ datasetName = null, splatApiPath = null }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loadedGaussModel, setLoadedGaussModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // NEW: freecam toggle state
    const [freecamEnabled, setFreecamEnabled] = useState(false);

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

    // First-person freecam component
    function FreeCam({ enabled = true, speed = 3, sprintMultiplier = 3 }) {
        const { camera, gl } = useThree();
        const keys = useRef({ KeyW: false, KeyA: false, KeyS: false, KeyD: false, Space: false, ControlLeft: false, ShiftLeft: false });
        // store yaw/pitch and pointer lock state; init from camera quaternion for smooth start
        const look = useRef({ yaw: 0, pitch: 0, locked: false });

        useEffect(() => {
            // derive initial yaw/pitch from camera quaternion using YXZ order (common FPS)
            const e = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
            look.current.yaw = e.y;
            look.current.pitch = e.x;
        }, [camera]);

        useEffect(() => {
            if (!enabled) return;

            const onPointerLockChange = () => {
                look.current.locked = document.pointerLockElement === gl.domElement;
            };

            const onMouseMove = (e) => {
                if (!look.current.locked) return;
                const sensitivity = 0.0025; // tweak this for faster/slower look
                look.current.yaw -= e.movementX * sensitivity;
                look.current.pitch -= e.movementY * sensitivity;
                const max = Math.PI / 2 - 0.001;
                look.current.pitch = Math.max(-max, Math.min(max, look.current.pitch));

                // apply via quaternion to avoid Euler gimbal issues and keep smooth rotation
                const quat = new THREE.Quaternion().setFromEuler(new THREE.Euler(look.current.pitch, look.current.yaw, 0, "YXZ"));
                camera.quaternion.copy(quat);
            };

            const onKey = (e) => {
                if (e.code in keys.current) keys.current[e.code] = e.type === "keydown";
            };

            const onClick = () => {
                try { gl.domElement.requestPointerLock(); } catch {}
            };

            document.addEventListener("pointerlockchange", onPointerLockChange);
            document.addEventListener("mousemove", onMouseMove);
            document.addEventListener("keydown", onKey);
            document.addEventListener("keyup", onKey);
            gl.domElement.addEventListener("click", onClick);

            return () => {
                document.removeEventListener("pointerlockchange", onPointerLockChange);
                document.removeEventListener("mousemove", onMouseMove);
                document.removeEventListener("keydown", onKey);
                document.removeEventListener("keyup", onKey);
                gl.domElement.removeEventListener("click", onClick);
                try { if (document.pointerLockElement === gl.domElement) document.exitPointerLock(); } catch {}
            };
        }, [enabled, camera, gl]);

        useFrame((_, delta) => {
            if (!enabled || !look.current.locked) return;

            // get camera forward direction (includes vertical component when looking up/down)
            const dir = new THREE.Vector3();
            camera.getWorldDirection(dir);
            dir.normalize();
            // right vector perpendicular to forward and world up
            const right = new THREE.Vector3().crossVectors(dir, camera.up).normalize();

            const move = new THREE.Vector3();
            if (keys.current.KeyW) move.add(dir);
            if (keys.current.KeyS) move.sub(dir);
            if (keys.current.KeyA) move.sub(right);
            if (keys.current.KeyD) move.add(right);
            if (keys.current.Space) move.y += 1;
            if (keys.current.ControlLeft) move.y -= 1;

            if (move.lengthSq() > 0) {
                move.normalize();
                const s = speed * (keys.current.ShiftLeft ? sprintMultiplier : 1);
                camera.position.addScaledVector(move, s * delta);
            }
        });

        return null;
    }

    return (
        <>
            <div style={{ position: "relative", marginBottom: 8, minHeight: 36 }}>
                <div style={{ position: "absolute", left: 8, top: 8, zIndex: 5 }}>
                    <button
                        onClick={() => setFreecamEnabled((v) => !v)}
                        style={{
                            padding: "6px 12px",
                            borderRadius: 6,
                            border: "1px solid #0048a4",
                            background: freecamEnabled ? "#0048a4" : "#0f6bd8",
                            color: "white",
                            fontWeight: 600,
                            fontSize: 13,
                            cursor: "pointer",
                            boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
                        }}
                    >
                        {freecamEnabled ? "Disable Freecam" : "Enable Freecam"}
                    </button>
                </div>
                <div style={{ position: "absolute", right: 8, top: 12, zIndex: 5, color: "#666", fontSize: 13 }}>
                    {freecamEnabled ? "Click canvas to lock pointer. WASD + Space/Ctrl to move." : "Orbit controls active"}
                </div>
            </div>

            <div style={{ width: "100%", height: 480 }}>
                <Canvas style={{ width: "100%", height: "100%" }}>
                    {/* use freecam or orbit controls */}
                    {freecamEnabled ? <FreeCam enabled={true} /> : <OrbitControls />}

                    {loadedGaussModel ? (
                        <Splat
                            key={loadedGaussModel}
                            src={loadedGaussModel}
                            // rotation={[0.75 * Math.PI, -.02 * Math.PI, .55 * Math.PI]}
                        />
                    ) : null}
                </Canvas>
            </div>
        </>
    );
}