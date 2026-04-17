import React from "react";
import { Link } from "react-router-dom";

export default function Documentation() {
  return (
    <main style={s.page}>
      <header style={s.header}>
        <h1 style={s.h1}>How the viewer renders splats</h1>
        <p style={s.lede}>
          A walkthrough of what actually happens when you open a scene on the
          Live Demos page. Written for anyone who hasn't worked with Gaussian
          splatting before.
        </p>
      </header>

      <section style={s.section}>
        <h2 style={s.h2}>What the viewer is looking at</h2>
        <p style={s.p}>
          The file being rendered is a list of 3D gaussians. Each gaussian is
          basically a tiny fuzzy ellipsoid floating in space. It has a
          position (x, y, z), a shape (covariance — how stretched and oriented
          the ellipsoid is), a color, and an opacity. A full scene is millions
          of these stacked up. When you look at enough of them blended
          together they form a photorealistic 3D image.
        </p>
        <p style={s.p}>
          There are two related file formats in play here:
        </p>
        <ul style={s.ul}>
          <li>
            <code style={s.code}>.ply</code> — the format the trainer writes
            out. Text or binary, one vertex per gaussian with all the
            parameters as properties.
          </li>
          <li>
            <code style={s.code}>.splat</code> — a compact binary repack of
            the same data. Smaller and faster to load in the browser. This is
            what the viewer actually consumes.
          </li>
        </ul>
        <p style={s.p}>
          Our backend pipeline (either OpenSplat or Faster-GS) writes a
          <code style={s.code}>.ply</code>, then a small converter script
          (<code style={s.code}>backend/scripts/converter.py</code>)
          repacks it into <code style={s.code}>.splat</code> before the file
          ever reaches the browser.
        </p>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>Libraries doing the rendering</h2>
        <p style={s.p}>
          The viewer component (<code style={s.code}>GaussViewer.jsx</code>)
          uses three nested libraries:
        </p>
        <ul style={s.ul}>
          <li>
            <strong>three.js</strong> (v0.183) — the underlying WebGL2 engine.
            Handles the GPU scene graph, shaders, and draw calls.
          </li>
          <li>
            <strong>@react-three/fiber</strong> (v9) — the React wrapper
            around three.js. Lets us describe the 3D scene with JSX.
          </li>
          <li>
            <strong>@react-three/drei</strong> (v10) — a grab-bag of R3F
            helper components. We use its <code style={s.code}>&lt;Splat&gt;</code>
            component to render the file. That component is the one actually
            parsing the <code style={s.code}>.splat</code> bytes and setting
            up the WebGL instanced-quad pipeline underneath.
          </li>
        </ul>
        <p style={s.p}>
          Separately, on the <Link to="/converter" style={s.link}>Converter</Link>
          page, we use a library called <strong>gsplat</strong> (v1.2) to turn
          a user-uploaded <code style={s.code}>.ply</code> into
          <code style={s.code}> .splat</code> client-side. It's pure
          JavaScript — no GPU involvement — and runs entirely on the main
          thread.
        </p>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>What runs on the CPU vs GPU</h2>
        <p style={s.p}>
          Rendering a splat scene is unusual compared to traditional 3D
          because two things happen each frame: the GPU rasterizes the
          gaussians as usual, but the CPU first has to sort them so the
          transparency stacks up correctly. Here's the breakdown.
        </p>

        <h3 style={s.h3}>On the CPU</h3>
        <p style={s.p}>
          <strong>Once, when the scene loads:</strong> the browser fetches
          the <code style={s.code}>.splat</code> file from the backend,
          parses the binary header and body into typed arrays (positions,
          covariances, colors, opacities), and uploads those arrays to the
          GPU as vertex buffers. This is O(N) in the number of gaussians,
          runs on the main thread, and typically takes under a second for
          a million gaussians.
        </p>
        <p style={s.p}>
          <strong>Every frame, while you're looking around:</strong> the CPU
          has to sort gaussians back-to-front relative to the camera so alpha
          compositing produces a stable image. Without this sort you get
          flickering edges as gaussians draw in the wrong order. drei's
          <code style={s.code}> &lt;Splat&gt;</code> pushes this sort into a
          Web Worker so it doesn't block the main thread — you can still
          scroll and interact while the sort is updating.
        </p>
        <p style={s.p}>
          The sort is usually the biggest CPU cost in the viewer. For
          million-gaussian scenes on a mid-range laptop, expect a few
          milliseconds per frame.
        </p>

        <h3 style={s.h3}>On the GPU</h3>
        <p style={s.p}>
          <strong>Vertex shader:</strong> projects each 3D gaussian into
          2D screen space. This means converting the 3D covariance matrix
          into a 2D covariance matrix that describes the ellipse the
          gaussian covers on screen. One instanced quad is issued per
          gaussian, positioned to cover that ellipse.
        </p>
        <p style={s.p}>
          <strong>Fragment shader:</strong> for each pixel inside the quad,
          evaluates the 2D gaussian density function to get a falloff
          weight, multiplies by the gaussian's color + opacity, and blends
          it into the framebuffer on top of everything behind it.
        </p>
        <p style={s.p}>
          The GPU is the workhorse. Modern integrated GPUs handle a few
          hundred thousand gaussians fine; for multi-million scenes you
          really want a dedicated GPU.
        </p>

        <h3 style={s.h3}>PLY → SPLAT conversion (converter page only)</h3>
        <p style={s.p}>
          The converter page is 100% CPU work. gsplat parses the PLY header,
          reads per-vertex properties, repacks into the .splat binary format,
          and triggers a download. No GPU, no worker, just typed-array
          shuffling on the main thread. Runtime scales linearly with the
          gaussian count.
        </p>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>Hardware and browser requirements</h2>
        <ul style={s.ul}>
          <li>
            <strong>WebGL2-capable browser.</strong> Chrome 56+, Firefox 51+,
            Safari 15+, Edge 79+. If WebGL2 is disabled or missing, the
            viewer silently renders nothing.
          </li>
          <li>
            <strong>A GPU.</strong> Anything dedicated from the last five
            years works well. Recent integrated GPUs (Apple M-series, Intel
            Iris Xe, AMD Radeon iGPUs) handle scenes up to around one
            million gaussians. Older integrated chips will still run but
            the framerate drops noticeably.
          </li>
          <li>
            <strong>VRAM.</strong> Roughly 4 GB comfortably handles a
            multi-million gaussian scene. Smaller demo scenes (under 500k
            gaussians) fit in about 1 GB.
          </li>
          <li>
            <strong>CPU.</strong> Any modern laptop CPU is fine. The
            per-frame sort is parallelizable in principle but drei's default
            implementation is single-threaded inside the Worker.
          </li>
          <li>
            <strong>Mobile.</strong> iOS Safari 15+ and Android Chrome both
            work. Expect thermal throttling on longer sessions with big
            scenes.
          </li>
        </ul>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>Known limits</h2>
        <ul style={s.ul}>
          <li>
            <strong>No WebGPU path yet.</strong> drei's
            <code style={s.code}> &lt;Splat&gt;</code> is WebGL2-only as of
            v10.7. A WebGPU version would likely shift the sort onto the GPU
            and reduce CPU load.
          </li>
          <li>
            <strong>SH band 0 only.</strong> The drei renderer reads only
            the flat-color spherical-harmonic coefficient (band 0). Higher
            bands (which add view-dependent color) are present in the
            <code style={s.code}> .splat</code> file but not used. Our
            backend only produces band 0 anyway, so this isn't a regression.
          </li>
          <li>
            <strong>Large files.</strong> A 20 MB+ splat has to be downloaded
            and parsed before anything renders. We don't stream gaussians
            progressively — you see a blank canvas until the whole file is
            ready.
          </li>
        </ul>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>Where to look in the code</h2>
        <ul style={s.ul}>
          <li>
            <code style={s.code}>frontend/src/components/GaussViewer.jsx</code>
            — the actual viewer component. Wraps drei's Splat inside a
            react-three-fiber Canvas. Also has a freecam mode (WASD + mouse
            look) that teammates added.
          </li>
          <li>
            <code style={s.code}>frontend/src/components/Converter.jsx</code>
            — the PLY→SPLAT conversion page.
          </li>
          <li>
            <code style={s.code}>backend/scripts/converter.py</code> — the
            server-side version of the same conversion, used by the
            pipeline before publishing the result.
          </li>
        </ul>
      </section>
    </main>
  );
}

const s = {
  page: {
    maxWidth: 840,
    margin: "0 auto",
    padding: "32px 24px 64px",
    color: "#0f172a",
    lineHeight: 1.6,
  },
  header: {
    marginBottom: 28,
    borderBottom: "1px solid #dbe3ef",
    paddingBottom: 20,
  },
  h1: { margin: "0 0 8px", fontSize: "1.9rem" },
  lede: { margin: "0 0 12px", color: "#475569" },
  section: { marginTop: 28 },
  h2: {
    margin: "0 0 10px",
    fontSize: "1.25rem",
    color: "#0048a4",
  },
  h3: {
    margin: "16px 0 6px",
    fontSize: "1.02rem",
    color: "#0f172a",
  },
  p: { margin: "0 0 10px" },
  ul: { margin: "0 0 10px", paddingLeft: 22 },
  code: {
    background: "#f1f5f9",
    padding: "1px 6px",
    borderRadius: 4,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "0.9em",
  },
  link: { color: "#0f6bd8" },
};
