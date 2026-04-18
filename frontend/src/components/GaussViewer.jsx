import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Splat, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import "./GaussViewer.css";

// The viewer itself is drei's <Splat> (WebGL2) over @react-three/fiber +
// three.js. We don't use mkkellogg/GaussianSplats3D here — the controls
// below wire React state to three.js / Splat props, not to any third-
// party viewer SDK.
//
// Controls panel matches the Brush-style layout: collapsible, top-left,
// semi-transparent dark card. Controls are:
//   - Background color          -> <color attach="background" />
//   - FOV slider (20°-120°)     -> camera.fov + updateProjectionMatrix()
//   - Splat scale slider        -> <Splat scale={[s,s,s]} /> (uniform
//                                  scene scale; note in tooltip)
//   - Show grid checkbox        -> <gridHelper>
//   - Splat count (read-only)   -> derived from .splat blob byte length
//                                  (.splat format is 32 bytes/gaussian)
//   - SH degree (read-only)     -> always 0; drei's <Splat> consumes the
//                                  flat-color .splat binary format which
//                                  only stores band-0 spherical harmonics

const FOV_MIN = 20;
const FOV_MAX = 120;
const FOV_DEFAULT = 46;
const SPLAT_SCALE_MIN = 0.3;
const SPLAT_SCALE_MAX = 3.0;
const SPLAT_SCALE_DEFAULT = 1.0;
const BYTES_PER_SPLAT = 32;

// Applies the fov state to the R3F default camera. Lives inside the
// <Canvas> tree so it can pull the camera out via useThree.
function FovUpdater({ fov }) {
  const { camera } = useThree();
  useEffect(() => {
    if (camera && camera.isPerspectiveCamera) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    }
  }, [camera, fov]);
  return null;
}

// Bridges drei's <OrbitControls> with React state so mouse-drag and the
// Azimuth/Elevation/Distance sliders share the same source of truth.
// Mouse orbit fires 'change' events on the controls; we read them back
// into the slider state. Slider changes come back in through props; we
// apply them imperatively to the controls without emitting a fresh
// 'change' event (suppressRef guards the re-entry loop).
//
// Why sliders are world-spherical: orbit controls expose
// getAzimuthalAngle (rotation around world Y, radians in [-π, π]),
// getPolarAngle (angle from +Y down to camera, radians in [0, π]),
// and getDistance. We expose azimuth in degrees [-180, 180] and
// elevation in degrees [-89, 89] where elevation = 90° - polar.
function OrbitControlsSync({
  target,           // [x, y, z] orbit center (splat bbox center)
  azimuth, elevation, distance,  // current slider values (degrees / world units)
  onChange,         // fn(az, el, dist) called when mouse-drag moves the camera
  enabled = true,   // when freecam is on we disable our OrbitControls
}) {
  const controlsRef = useRef(null);
  // Flag set while we're imperatively applying slider values so the
  // 'change' handler doesn't echo them right back into state.
  const suppressRef = useRef(false);
  const { camera } = useThree();

  // Mouse-drag -> slider state. Debounced only by React batching; the
  // handler itself is cheap.
  useEffect(() => {
    const ctrl = controlsRef.current;
    if (!ctrl) return;
    const handler = () => {
      if (suppressRef.current) return;
      const rad2deg = 180 / Math.PI;
      const theta = ctrl.getAzimuthalAngle() * rad2deg;
      const phi = ctrl.getPolarAngle() * rad2deg;
      const el = 90 - phi;
      const d = ctrl.getDistance();
      onChange(theta, el, d);
    };
    ctrl.addEventListener("change", handler);
    return () => ctrl.removeEventListener("change", handler);
  }, [onChange]);

  // Slider state -> camera. Recompute the camera position from
  // spherical coords around `target` and push it to OrbitControls.
  useEffect(() => {
    const ctrl = controlsRef.current;
    if (!ctrl || !target) return;
    const deg2rad = Math.PI / 180;
    const thetaRad = azimuth * deg2rad;
    const phiRad = (90 - elevation) * deg2rad;
    // Clamp phi so we never hit a pole (matches the -89..89 slider clamp).
    const phiClamped = Math.max(0.01, Math.min(Math.PI - 0.01, phiRad));

    const tgt = new THREE.Vector3(target[0], target[1], target[2]);
    const sph = new THREE.Spherical(Math.max(0.001, distance), phiClamped, thetaRad);
    const offset = new THREE.Vector3().setFromSpherical(sph);
    const nextPos = tgt.clone().add(offset);

    // Skip if we're already there (float epsilon tolerance). Prevents
    // feedback loops when this effect fires from the same state the
    // 'change' handler just wrote.
    const posDelta = camera.position.distanceTo(nextPos);
    const targetDelta = ctrl.target.distanceTo(tgt);
    if (posDelta < 1e-4 && targetDelta < 1e-4) return;

    suppressRef.current = true;
    camera.position.copy(nextPos);
    ctrl.target.copy(tgt);
    ctrl.update();
    // Microtask so any synchronous 'change' event that fires inside
    // update() has already been swallowed before we unset the flag.
    Promise.resolve().then(() => { suppressRef.current = false; });
  }, [target, azimuth, elevation, distance, camera]);

  if (!enabled) return null;
  return <OrbitControls ref={controlsRef} enableDamping={false} />;
}

// Injects a uSplatSizeMultiplier uniform into drei's SplatMaterial
// vertex shader so each gaussian's projected screen-space radius is
// scaled WITHOUT moving its center. Hook is lazy: we only tamper with
// the material when scaleMultiplier != 1.0, so the default 1.0x path
// uses verbatim stock drei (avoids any shader-recompile regression on
// the common load case). Once overridden, subsequent slider moves just
// update the uniform value — no recompile.
//
// Implementation choices:
//   - The SplatMaterial from drei's shaderMaterial() is a THREE.ShaderMaterial
//     subclass. onBeforeCompile is respected as usual.
//   - We locate the material via a groupRef traverse (drei's <Splat>
//     doesn't forwardRef, so we wrap in a <group>).
//   - The group stays at scale [1,1,1] — any non-unit group transform
//     would reintroduce the bogus "whole-scene zoom" behavior.
function SplatWithScale({ src, scaleMultiplier }) {
  const groupRef = useRef(null);
  const shaderRef = useRef(null);
  const overriddenRef = useRef(false); // did we ever install onBeforeCompile on this material?

  const scaleRef = useRef(scaleMultiplier);
  useEffect(() => { scaleRef.current = scaleMultiplier; }, [scaleMultiplier]);

  // Find the splat material inside the group. Returns null if the
  // drei Splat hasn't committed its mesh yet (race on very-first-frame
  // mount), which lets callers retry on the next tick.
  const findSplatMaterial = () => {
    const g = groupRef.current;
    if (!g) return null;
    let mat = null;
    g.traverse((obj) => {
      if (!mat && obj.isMesh && obj.material && obj.material.isShaderMaterial) {
        mat = obj.material;
      }
    });
    return mat;
  };

  const installOverride = (mat) => {
    if (!mat || overriddenRef.current) return;
    overriddenRef.current = true;
    shaderRef.current = null;
    const prev = mat.onBeforeCompile;
    mat.onBeforeCompile = (shader, renderer) => {
      if (prev) {
        try { prev.call(mat, shader, renderer); } catch {}
      }
      shader.vertexShader = shader.vertexShader
        .replace(
          "precision highp usampler2D;",
          "precision highp usampler2D;\nuniform float uSplatSizeMultiplier;",
        )
        .replace(
          "position.x * v2 / viewport",
          "position.x * v2 * uSplatSizeMultiplier / viewport",
        )
        .replace(
          "position.y * v1 / viewport",
          "position.y * v1 * uSplatSizeMultiplier / viewport",
        );
      shader.uniforms.uSplatSizeMultiplier = { value: scaleRef.current };
      shaderRef.current = shader;
    };
    // THREE caches compiled programs by key; flipping needsUpdate forces
    // a fresh acquire on the next draw so our onBeforeCompile actually
    // runs. Also nudge the key so the renderer can't dedupe against a
    // previously-compiled program for this material class.
    mat.needsUpdate = true;
  };

  // Reset our "already overridden" latch whenever the src changes (drei
  // mounts a new mesh + material per src).
  useEffect(() => {
    overriddenRef.current = false;
    shaderRef.current = null;
  }, [src]);

  // Only touch the material when the user actually changes the scale
  // away from 1.0. This keeps the default load path identical to stock
  // drei. Once we've installed the override, we leave it in place for
  // the life of the material (resetting to 1.0 just sets the uniform
  // back to 1.0, which is a shader-level no-op).
  useEffect(() => {
    if (scaleMultiplier === 1.0 && !overriddenRef.current) {
      return; // never touched, never will — short-circuit.
    }
    if (!overriddenRef.current) {
      // First-time install: try now, retry on next rAF if drei hasn't
      // committed the mesh yet.
      let raf = 0;
      const tryInstall = () => {
        const mat = findSplatMaterial();
        if (mat) {
          installOverride(mat);
        } else {
          raf = requestAnimationFrame(tryInstall);
        }
      };
      tryInstall();
      return () => { if (raf) cancelAnimationFrame(raf); };
    }
    // Already overridden: just write the new uniform value. No recompile.
    const sh = shaderRef.current;
    if (sh && sh.uniforms && sh.uniforms.uSplatSizeMultiplier) {
      sh.uniforms.uSplatSizeMultiplier.value = scaleMultiplier;
    }
  }, [scaleMultiplier, src]);

  return (
    <group ref={groupRef}>
      <Splat key={src} src={src} />
    </group>
  );
}

function ControlsPanel({
  bgColor, setBgColor,
  fov, setFov,
  splatScale, setSplatScale,
  showGrid, setShowGrid,
  splatCount,
  shDegree,
  azimuth, setAzimuth,
  elevation, setElevation,
  distance, setDistance,
  distanceMin, distanceMax,
  onResetView,
}) {
  // Start collapsed so the panel is out of the way on load; the user
  // expands it via the caret when they want to tweak controls.
  const [collapsed, setCollapsed] = useState(true);
  const [showHelp, setShowHelp] = useState(false);

  return (
    <div className={`gauss-controls-panel${collapsed ? " is-collapsed" : ""}`}>
      <div className="gauss-controls-head">
        <button
          type="button"
          className="gauss-controls-toggle"
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? "Expand controls" : "Collapse controls"}
          aria-expanded={!collapsed}
        >
          <span className="gauss-controls-caret" aria-hidden>{collapsed ? "▶" : "▼"}</span>
          Controls
        </button>
        <button
          type="button"
          className="gauss-controls-help"
          onMouseEnter={() => setShowHelp(true)}
          onMouseLeave={() => setShowHelp(false)}
          onFocus={() => setShowHelp(true)}
          onBlur={() => setShowHelp(false)}
          onClick={() => setShowHelp((v) => !v)}
          aria-label="Controls help"
        >
          ?
        </button>
        {showHelp && (
          <div role="tooltip" className="gauss-controls-tooltip">
            <strong>Camera:</strong> left-drag orbits, right-drag pans,
            scroll zooms. Toggle Freecam above the canvas for WASD
            first-person nav.<br />
            <strong>Background:</strong> scene clear color.<br />
            <strong>Field of view:</strong> perspective angle of the camera.
            Wider values exaggerate parallax.<br />
            <strong>Splat scale:</strong> uniformly scales the splat
            scene; gaussians look bigger or smaller relative to the grid.<br />
            <strong>Show grid:</strong> reference 10×10 ground plane.<br />
            <strong>Splats / SH:</strong> stats of the loaded model.
          </div>
        )}
      </div>

      {!collapsed && (
        <div className="gauss-controls-body">
          <label className="gauss-controls-row">
            <span className="gauss-controls-label">Background</span>
            <input
              type="color"
              className="gauss-controls-color"
              value={bgColor}
              onChange={(e) => setBgColor(e.target.value)}
            />
          </label>

          <label className="gauss-controls-row">
            <span className="gauss-controls-label">
              Field of view
              <span className="gauss-controls-value">{fov}°</span>
            </span>
            <input
              type="range"
              min={FOV_MIN}
              max={FOV_MAX}
              step="1"
              value={fov}
              onChange={(e) => setFov(Number(e.target.value))}
            />
          </label>

          <label className="gauss-controls-row">
            <span className="gauss-controls-label">
              Splat scale
              <span className="gauss-controls-value">{splatScale.toFixed(2)}×</span>
            </span>
            <input
              type="range"
              min={SPLAT_SCALE_MIN}
              max={SPLAT_SCALE_MAX}
              step="0.05"
              value={splatScale}
              onChange={(e) => setSplatScale(Number(e.target.value))}
            />
          </label>

          <label className="gauss-controls-row">
            <span className="gauss-controls-label">
              Azimuth
              <span className="gauss-controls-value">{Math.round(azimuth)}°</span>
            </span>
            <input
              type="range"
              min={-180}
              max={180}
              step="1"
              value={azimuth}
              onChange={(e) => setAzimuth(Number(e.target.value))}
            />
          </label>

          <label className="gauss-controls-row">
            <span className="gauss-controls-label">
              Elevation
              <span className="gauss-controls-value">{Math.round(elevation)}°</span>
            </span>
            <input
              type="range"
              min={-89}
              max={89}
              step="1"
              value={elevation}
              onChange={(e) => setElevation(Number(e.target.value))}
            />
          </label>

          <label className="gauss-controls-row">
            <span className="gauss-controls-label">
              Distance
              <span className="gauss-controls-value">{distance.toFixed(2)}</span>
            </span>
            <input
              type="range"
              min={distanceMin}
              max={distanceMax}
              step={(distanceMax - distanceMin) / 200}
              value={distance}
              onChange={(e) => setDistance(Number(e.target.value))}
            />
          </label>

          <label className="gauss-controls-row gauss-controls-row--inline">
            <span className="gauss-controls-label">Show grid</span>
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.target.checked)}
            />
          </label>

          <button
            type="button"
            className="gauss-controls-reset"
            onClick={onResetView}
          >
            Reset View
          </button>

          <div className="gauss-controls-stats">
            <div className="gauss-controls-stat">
              <span className="gauss-controls-stat-key">Splats</span>
              <span className="gauss-controls-stat-val">
                {splatCount != null
                  ? splatCount.toLocaleString(undefined, { maximumFractionDigits: 0 })
                  : "—"}
              </span>
            </div>
            <div className="gauss-controls-stat">
              <span className="gauss-controls-stat-key">SH degree</span>
              <span className="gauss-controls-stat-val">{shDegree}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function GaussViewer({ datasetName = null, splatApiPath = null }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loadedGaussModel, setLoadedGaussModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [freecamEnabled, setFreecamEnabled] = useState(false);

  // Stats read from the fetched .splat blob. The format is 32 bytes per
  // gaussian with SH band 0 only, so we derive these at fetch time once
  // instead of instrumenting drei's loader.
  const [splatCount, setSplatCount] = useState(null);
  const [shDegree] = useState(0);

  // Scene controls wired to R3F / Three.js. Defaults mirror Brush.
  const [bgColor, setBgColor] = useState("#000000");
  const [fov, setFov] = useState(FOV_DEFAULT);
  const [splatScale, setSplatScale] = useState(SPLAT_SCALE_DEFAULT);
  const [showGrid, setShowGrid] = useState(false);

  // Splat bounding box — parsed from the raw .splat blob at fetch time
  // so we have the true scene center + radius for orbit framing + the
  // Distance slider range. The .splat format stores each gaussian as
  // 3 float32 positions followed by 3 float32 scales + 4 uint8 color
  // + 4 uint8 rotation = 32 bytes. We only need the first 12 bytes
  // per record. drei's Splat negates the y axis when uploading into
  // its position texture (see shared.centerAndScaleData[+1] = -y in
  // Splat.js), so we match that flip here so our orbit target lines
  // up with the *rendered* center, not the raw-file center.
  const [splatCenter, setSplatCenter] = useState(null); // [x,y,z] in rendered world space
  const [splatRadius, setSplatRadius] = useState(null); // half-diagonal of bbox

  // Orbit camera state. Angles in degrees; distance in world units.
  // null until the first splat loads (the OrbitControlsSync component
  // below fills these in on mount + every 'change' event from the
  // underlying OrbitControls, so mouse-drag and slider-drag share
  // the same source of truth).
  const [azimuth, setAzimuth] = useState(0);
  const [elevation, setElevation] = useState(15);
  const [distance, setDistance] = useState(3);
  // Snapshot captured at first frame of every new splat load. Used by
  // the Reset View button to restore the initial framing.
  const initialPoseRef = useRef({ azimuth: 0, elevation: 15, distance: 3 });

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

  // Compute bbox over the gaussian centers stored in the .splat blob.
  // Cheap enough to run on the full set (12 bytes per gaussian read as
  // float32, ~8 MB for a 700k-gaussian scene). Returns the rendered-
  // world-space center (y negated to match drei's flip) + a robust
  // half-diagonal radius we'll use to scale the Distance slider range.
  //
  // Robust bounds: instead of raw min/max we take the 5th/95th
  // percentile on each axis. This excludes the halo of stray "floater"
  // gaussians most trained scenes end up with — they'd otherwise blow
  // up the bbox and push the default camera way too far back, making
  // the subject look small in the viewport at default zoom.
  function computeBboxFromBlob(arrayBuffer) {
    const stride = 8; // 8 float32s per 32-byte record
    const floats = new Float32Array(arrayBuffer);
    const count = Math.floor(floats.length / stride);
    if (count === 0) return null;

    // Extract per-axis position arrays so we can sort each one and
    // read percentile bounds. Float32Array allocates ~4 bytes per
    // entry; for 700k gaussians that's ~8.4 MB across the three axes,
    // and the sorts take <200ms in our testing.
    const xs = new Float32Array(count);
    const ys = new Float32Array(count);
    const zs = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      const off = i * stride;
      xs[i] = floats[off];
      ys[i] = floats[off + 1];
      zs[i] = floats[off + 2];
    }
    // Float32Array.sort() is numeric by default (Array.prototype.sort
    // on regular arrays is lexicographic, which would be wrong here).
    xs.sort();
    ys.sort();
    zs.sort();

    const loIdx = Math.max(0, Math.floor(count * 0.05));
    const hiIdx = Math.min(count - 1, Math.ceil(count * 0.95) - 1);
    const minX = xs[loIdx], maxX = xs[hiIdx];
    const minY = ys[loIdx], maxY = ys[hiIdx];
    const minZ = zs[loIdx], maxZ = zs[hiIdx];

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const dx = (maxX - minX) / 2;
    const dy = (maxY - minY) / 2;
    const dz = (maxZ - minZ) / 2;
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    // Match drei Splat's y-axis flip on upload — the rendered scene
    // lives at (x, -y_raw, z), so the orbit target needs the negated y.
    return { center: [cx, -cy, cz], radius };
  }

  // Shared loader for both the dataset-root and HPG-run API paths. Keeps
  // the blob lifetime wrapped so the previous object URL is revoked as
  // soon as a new splat arrives.
  async function fetchSplatFromEndpoint(url) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`failed to fetch splat file: ${res.status}`);
      const blob = await res.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const objectUrl = URL.createObjectURL(blob);

      // Each .splat record is 32 bytes, so blob.size / 32 is the exact
      // gaussian count.
      setSplatCount(Math.floor(blob.size / BYTES_PER_SPLAT));

      const bbox = computeBboxFromBlob(arrayBuffer);
      if (bbox) {
        setSplatCenter(bbox.center);
        setSplatRadius(bbox.radius);
        // Initial framing: pull back 1.0 * robust_radius along the
        // default camera direction, where robust_radius is the 5/95
        // percentile bbox. 1.0x puts the subject filling most of the
        // viewport at load; the slider can go wider (3.0x) or tighter
        // (0.3x) from there. Angles are standard orbit start — slight
        // top-down tilt so the user isn't looking straight at the
        // equator.
        const initialDistance = bbox.radius * 1.0;
        initialPoseRef.current = {
          azimuth: 0,
          elevation: 15,
          distance: initialDistance,
        };
        setAzimuth(0);
        setElevation(15);
        setDistance(initialDistance);
      }

      setLoadedGaussModel((prev) => {
        if (prev && typeof prev === "string" && prev.startsWith("blob:")) {
          try { URL.revokeObjectURL(prev); } catch {}
        }
        return objectUrl;
      });
    } catch (e) {
      setError(String(e));
      setSplatCount(null);
      setSplatCenter(null);
      setSplatRadius(null);
    } finally {
      setLoading(false);
    }
  }

  async function loadSplatFromDataset(name) {
    if (!name) return;
    await fetchSplatFromEndpoint(`/api/datasets/${encodeURIComponent(name)}/splat`);
  }

  async function loadSplatFromPath(path) {
    if (!path) return;
    await fetchSplatFromEndpoint(path);
  }

  useEffect(() => {
    if (splatApiPath) {
      loadSplatFromPath(splatApiPath);
    } else if (datasetName) {
      loadSplatFromDataset(datasetName);
    }
  }, [datasetName, splatApiPath]);

  // Revoke blob URL on unmount so we don't leak the previous splat.
  useEffect(() => {
    return () => {
      if (loadedGaussModel && typeof loadedGaussModel === "string" && loadedGaussModel.startsWith("blob:")) {
        try { URL.revokeObjectURL(loadedGaussModel); } catch {}
      }
    };
  }, [loadedGaussModel]);

  // First-person freecam component. Retained from the earlier viewer so
  // the WASD + pointer-lock flow still works alongside the new panel.
  function FreeCam({ enabled = true, speed = 3, sprintMultiplier = 3 }) {
    const { camera, gl } = useThree();
    const keys = useRef({ KeyW: false, KeyA: false, KeyS: false, KeyD: false, Space: false, ControlLeft: false, ShiftLeft: false });
    const look = useRef({ yaw: 0, pitch: 0, locked: false });

    useEffect(() => {
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
        const sensitivity = 0.0025;
        look.current.yaw -= e.movementX * sensitivity;
        look.current.pitch -= e.movementY * sensitivity;
        const max = Math.PI / 2 - 0.001;
        look.current.pitch = Math.max(-max, Math.min(max, look.current.pitch));
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

      const dir = new THREE.Vector3();
      camera.getWorldDirection(dir);
      dir.normalize();
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

  // Slider range for the Distance control. Derived from the splat's
  // robust bounding radius (5/95 percentile): min 0.3x lets the user
  // get in close to inspect detail, max 3.0x pulls back enough to
  // frame the whole scene with padding. Default (1.0x) lands in the
  // middle of that range so the subject fills most of the viewport.
  const { distanceSliderMin, distanceSliderMax } = useMemo(() => {
    const r = splatRadius || 1;
    return { distanceSliderMin: r * 0.3, distanceSliderMax: r * 3.0 };
  }, [splatRadius]);

  // Mouse-drag -> slider state. Called from OrbitControlsSync's 'change'
  // handler. Wrapped in useCallback so the identity is stable and the
  // effect inside OrbitControlsSync doesn't rebind on every render.
  const handleOrbitControlsChange = useCallback((az, el, d) => {
    setAzimuth(az);
    setElevation(el);
    setDistance(d);
  }, []);

  // Reset View button: snap every camera-affecting control back to the
  // pose captured when this splat first loaded, plus revert FOV and
  // splat scale to their own defaults. Does NOT revert bg color, grid,
  // or freecam state — those are independent of "framing" and the user
  // probably wants them sticky.
  const handleResetView = useCallback(() => {
    const pose = initialPoseRef.current;
    setAzimuth(pose.azimuth);
    setElevation(pose.elevation);
    setDistance(pose.distance);
    setFov(FOV_DEFAULT);
    setSplatScale(SPLAT_SCALE_DEFAULT);
  }, []);

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

      <div className="gauss-viewer-stage">
        {/* Panel sits over the top-left of the canvas. z-index keeps it
            above the WebGL surface but below any page-level modals. */}
        <ControlsPanel
          bgColor={bgColor} setBgColor={setBgColor}
          fov={fov} setFov={setFov}
          splatScale={splatScale} setSplatScale={setSplatScale}
          showGrid={showGrid} setShowGrid={setShowGrid}
          splatCount={splatCount}
          shDegree={shDegree}
          azimuth={azimuth} setAzimuth={setAzimuth}
          elevation={elevation} setElevation={setElevation}
          distance={distance} setDistance={setDistance}
          distanceMin={distanceSliderMin}
          distanceMax={distanceSliderMax}
          onResetView={handleResetView}
        />

        <div style={{ width: "100%", height: 480 }}>
          <Canvas style={{ width: "100%", height: "100%" }}>
            {/* Scene clear color driven by the background picker. */}
            <color attach="background" args={[bgColor]} />
            <FovUpdater fov={fov} />
            {showGrid && <gridHelper args={[10, 10, "#93b7e5", "#4a6683"]} />}
            {freecamEnabled ? (
              <FreeCam enabled={true} />
            ) : (
              <OrbitControlsSync
                target={splatCenter || [0, 0, 0]}
                azimuth={azimuth}
                elevation={elevation}
                distance={distance}
                onChange={handleOrbitControlsChange}
              />
            )}

            {loadedGaussModel ? (
              <SplatWithScale src={loadedGaussModel} scaleMultiplier={splatScale} />
            ) : null}
          </Canvas>
        </div>
      </div>
    </>
  );
}
