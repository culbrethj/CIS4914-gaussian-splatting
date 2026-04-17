import React, { useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Splat, OrbitControls } from "@react-three/drei";
import "./SideBySideViewer.css";

// Gallery comparison viewer. Renders two splats in side-by-side canvases
// with an optional "Match Camera" mode that locks both viewers to the
// same camera transform so you can compare the exact same pose between
// runs. When match is off, each canvas has its own OrbitControls.
//
// Props:
//   leftApiPath, rightApiPath  - /api/... paths to the two .splat files
//   leftLabel, rightLabel      - user-facing labels shown above each canvas
//   matchCamera                - true = primary (left) drives secondary (right)

export default function SideBySideViewer({
  leftApiPath,
  rightApiPath,
  leftLabel,
  rightLabel,
  matchCamera = false,
}) {
  const leftUrl = useBlobUrl(leftApiPath);
  const rightUrl = useBlobUrl(rightApiPath);

  // Shared state capsule: the left canvas writes its camera transform here
  // every frame, the right canvas reads from it when match mode is on. A
  // plain ref avoids a render-per-frame loop.
  const sharedCam = useRef({
    position: [0, 0, 3],
    quaternion: [0, 0, 0, 1],
  });

  return (
    <div className="sbs-viewer">
      <div className="sbs-pane">
        <div className="sbs-label" title={leftLabel}>{leftLabel || "Scene A"}</div>
        <div className="sbs-canvas-wrap">
          <Canvas camera={{ position: [0, 0, 3], fov: 55 }}>
            <color attach="background" args={["#f3f7ff"]} />
            <OrbitControls />
            {/* Publishes its camera to the shared ref every frame so the
                right pane can follow when matchCamera is on. */}
            <CameraReporter target={sharedCam} />
            {leftUrl && <Splat key={leftUrl} src={leftUrl} />}
          </Canvas>
        </div>
      </div>

      <div className="sbs-pane">
        <div className="sbs-label" title={rightLabel}>{rightLabel || "Scene B"}</div>
        <div className="sbs-canvas-wrap">
          <Canvas camera={{ position: [0, 0, 3], fov: 55 }}>
            <color attach="background" args={["#f3f7ff"]} />
            {matchCamera ? (
              // Drop OrbitControls on the right pane when matched - the
              // follower hook owns the camera and we don't want the user's
              // drag events to fight it.
              <CameraFollower source={sharedCam} />
            ) : (
              <OrbitControls />
            )}
            {rightUrl && <Splat key={rightUrl} src={rightUrl} />}
          </Canvas>
        </div>
      </div>
    </div>
  );
}

// Loads a .splat from an API path and returns a blob: URL that the drei
// <Splat> component can consume. Same pattern GaussViewer uses for the
// single-view case.
function useBlobUrl(apiPath) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    let cancelled = false;
    let currentUrl = null;
    if (!apiPath) {
      setUrl(null);
      return;
    }
    (async () => {
      try {
        const res = await fetch(apiPath);
        if (!res.ok) throw new Error(`${res.status}`);
        const blob = await res.blob();
        if (cancelled) return;
        currentUrl = URL.createObjectURL(blob);
        setUrl(currentUrl);
      } catch {
        if (!cancelled) setUrl(null);
      }
    })();
    return () => {
      cancelled = true;
      if (currentUrl) {
        try {
          URL.revokeObjectURL(currentUrl);
        } catch {
          /* ignore */
        }
      }
    };
  }, [apiPath]);
  return url;
}

// Writes the current camera transform into a shared mutable capsule each
// frame so another canvas can mirror it. `target` is a plain object (not a
// useRef); we intentionally mutate its `current` field - it's our own
// cross-canvas channel, not React state.
function CameraReporter({ target }) {
  const { camera } = useThree();
  useFrame(() => {
    const store = target.current;
    // eslint-disable-next-line react-hooks/immutability
    store.position = [camera.position.x, camera.position.y, camera.position.z];
    store.quaternion = [
      camera.quaternion.x,
      camera.quaternion.y,
      camera.quaternion.z,
      camera.quaternion.w,
    ];
  });
  return null;
}

// Reads the shared capsule each frame and applies it to this canvas's
// camera. Because we do this inside useFrame we follow even when the
// source pans smoothly via OrbitControls damping.
function CameraFollower({ source }) {
  const { camera } = useThree();
  useFrame(() => {
    const s = source.current;
    if (!s) return;
    camera.position.set(s.position[0], s.position[1], s.position[2]);
    camera.quaternion.set(s.quaternion[0], s.quaternion[1], s.quaternion[2], s.quaternion[3]);
  });
  return null;
}

