import React from "react";

// External links on this page open in a new tab so the user doesn't lose
// their place in the app. The rel attrs are the standard safety pair that
// keeps the new tab from getting a back-reference to our window.
const ext = { target: "_blank", rel: "noopener noreferrer" };

export default function Documentation() {
  return (
    <main style={s.page}>
      <header style={s.header}>
        <h1 style={s.h1}>Documentation</h1>
        <p style={s.lede}>
          What this project is, how the end-to-end pipeline works, and what
          we built it on.
        </p>
      </header>

      <section style={s.section}>
        <h2 style={s.h2}>1. What this app does</h2>
        <p style={s.p}>
          You upload a phone video of something you want to capture
          (a banana on a table, a building, a stuffed animal), wait for the
          job to finish, and get a 3D scene you can fly around in directly
          in your browser. It is end-to-end: the same app handles the video
          upload, the reconstruction, the training, and the viewer. You
          don't have to install anything or run a separate tool to see the
          result.
        </p>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>2. How the pipeline works</h2>
        <p style={s.p}>
          The pipeline has five stages. Each one runs after the previous
          one finishes, and live logs stream back to the Live Demos page
          so you can watch it happen.
        </p>

        <h3 style={s.h3}>a. Video ingestion</h3>
        <p style={s.p}>
          The browser posts the video file to the FastAPI backend at
          <code style={s.code}> /api/upload</code>. The backend stores it
          under <code style={s.code}>backend/datasets/&lt;name&gt;/video/</code>,
          validates the content type, and waits. When you click Run, a
          second call to <code style={s.code}>/api/run</code> spawns the
          pipeline orchestrator as a subprocess and streams its stdout
          over a WebSocket back to the browser.
        </p>

        <h3 style={s.h3}>b. Preprocessing</h3>
        <p style={s.p}>
          OpenCV pulls frames out of the video at a target frame rate
          (12 FPS by default). Each frame is then run through two filters.
          The first drops blurry frames using the variance of the Laplacian
          (LAPV) focus measure from
          {" "}
          <a style={s.link} {...ext} href="https://www.sciencedirect.com/science/article/abs/pii/S0031320312004736">
            Pertuz et al. 2013
          </a>
          : low variance means few sharp edges, which usually means motion
          blur or focus hunt. The second drops near-identical consecutive
          frames using mean absolute difference between grayscale
          thumbnails. Survivors get resized if wider than 1280 pixels, a
          cap that matches the resolution band the reference 3D Gaussian
          Splatting code targets.
        </p>

        <h3 style={s.h3}>c. Structure-from-Motion</h3>
        <p style={s.p}>
          Structure-from-Motion (SfM) recovers where the camera was when
          each frame was taken and produces a sparse 3D point cloud of the
          scene. Without it, the Gaussian Splatting trainer would have no
          idea where to place anything. We support two SfM methods. COLMAP
          ({" "}
          <a style={s.link} {...ext} href="https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf">
            Schönberger and Frahm, 2016
          </a>
          ) is the default and the industry standard: slow, CPU-bound, but
          very accurate on natural scenes. VGGT is a feed-forward neural
          network that estimates camera poses in under a minute on a GPU;
          it also handles reflective and transparent surfaces that COLMAP's
          classical feature matching gives up on. A kitchen-blender scene we
          ran in early testing registered only 2 of 45 frames under COLMAP
          but all 45 under VGGT, so the choice matters in practice.
        </p>

        <h3 style={s.h3}>d. Gaussian Splatting training</h3>
        <p style={s.p}>
          The trainer starts by placing one 3D gaussian at each sparse SfM
          point. Each gaussian carries a position, a 3D covariance matrix
          (scale and rotation), an opacity, and a color stored as
          spherical-harmonic coefficients. Training is a differentiable
          rasterizer and gradient descent: the trainer renders the current
          gaussians from each known camera, compares the render against the
          real frame, backpropagates the pixel loss through all gaussian
          parameters, and periodically densifies regions that need more
          detail and prunes gaussians that don't contribute. The method is
          due to
          {" "}
          <a style={s.link} {...ext} href="https://doi.org/10.1145/3592433">
            Kerbl et al. 2023
          </a>
          . We support two training backends. Faster-GS is the default and
          runs on UF's HiPerGator cluster; OpenSplat is available as a
          local alternative. Experiment results comparing them live in the
          Testing Information section of our written report.
        </p>

        <h3 style={s.h3}>e. Rendering</h3>
        <p style={s.p}>
          The trainer writes a <code style={s.code}>.ply</code> file with
          one vertex per gaussian. A small converter repacks it into the
          compact <code style={s.code}>.splat</code> binary format (32
          bytes per gaussian) and the final file comes back down to the
          backend. The browser fetches it, uploads the data to the GPU,
          and lets you navigate the scene. No re-training, no server round
          trip for camera movement.
        </p>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>3. What runs on your computer vs the server</h2>
        <p style={s.p}>
          <strong>Server side (HiPerGator).</strong> Frame extraction,
          preprocessing, SfM, and Gaussian Splatting training all run
          remotely. The backend dispatches a SLURM job to HiPerGator when
          you hit Run. Training runs on NVIDIA B200 GPUs on the
          <code style={s.code}> hpg-b200</code> partition by default.
          When the job finishes, the final
          <code style={s.code}> .splat</code> file, metrics, and charts get
          rsynced back to our backend automatically.
        </p>
        <p style={s.p}>
          <strong>Client side (your browser).</strong> Viewing a trained
          splat happens entirely locally. The viewer uses the
          {" "}
          <code style={s.code}>&lt;Splat&gt;</code> component from
          {" "}
          <a style={s.link} {...ext} href="https://github.com/pmndrs/drei">
            @react-three/drei
          </a>
          , which renders through
          {" "}
          <a style={s.link} {...ext} href="https://threejs.org/">Three.js</a>
          {" "}and WebGL2. Sorting the gaussians front-to-back for correct
          alpha compositing happens on the CPU in JavaScript (drei pushes
          it into a Web Worker so the main thread stays responsive). The
          actual projection, rasterization, and compositing happen on the
          GPU via WebGL2 shaders. The PLY-to-SPLAT converter on the
          Converter page is also client-side, pure JavaScript, no server
          round trip.
        </p>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>4. Hardware requirements</h2>
        <p style={s.p}><strong>To view splats (client side):</strong></p>
        <ul style={s.ul}>
          <li>
            A modern browser with WebGL2 support. Recent Chrome, Firefox,
            Safari, and Edge all qualify.
          </li>
          <li>
            A discrete or integrated GPU from the last ~5 years. Intel UHD
            620 and newer handle small splats; dedicated NVIDIA, AMD, and
            Apple Silicon GPUs handle large splats smoothly.
          </li>
          <li>
            Around 2 GB of free RAM per loaded splat. Large scenes can
            have 4+ million gaussians and correspondingly bigger footprints.
          </li>
        </ul>
        <p style={s.p}><strong>To train splats (server side):</strong></p>
        <ul style={s.ul}>
          <li>
            Training runs on HiPerGator, not locally. If you wanted to run
            the pipeline on your own machine, you'd need an NVIDIA GPU
            with at least 8 GB of VRAM, CUDA 11.8 or newer, and either the
            Faster-GS or OpenSplat backend compiled for your architecture.
          </li>
        </ul>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>5. Papers and tools we built on</h2>
        <p style={s.p}><strong>Papers</strong> (same numbering as the written report):</p>
        <ul style={s.ul}>
          <li>
            Mildenhall et al., 2020 &mdash; NeRF: Representing Scenes as
            Neural Radiance Fields for View Synthesis.
            {" "}
            <a style={s.link} {...ext} href="https://arxiv.org/abs/2003.08934">arxiv.org/abs/2003.08934</a>
          </li>
          <li>
            Kerbl et al., 2023 &mdash; 3D Gaussian Splatting for Real-Time
            Radiance Field Rendering.
            {" "}
            <a style={s.link} {...ext} href="https://doi.org/10.1145/3592433">doi.org/10.1145/3592433</a>
          </li>
          <li>
            Schönberger and Frahm, 2016 &mdash; Structure-from-Motion
            Revisited.
            {" "}
            <a style={s.link} {...ext} href="https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf">CVPR 2016 paper</a>
          </li>
          <li>
            Hartley and Zisserman, 2004 &mdash; Multiple View Geometry in
            Computer Vision (book).
          </li>
          <li>
            Pertuz et al., 2013 &mdash; Analysis of focus measure operators
            for shape-from-focus. Pattern Recognition 46(5).
            {" "}
            <a style={s.link} {...ext} href="https://www.sciencedirect.com/science/article/abs/pii/S0031320312004736">sciencedirect.com</a>
          </li>
          <li>
            Hahlbohm et al., 2026 &mdash; Faster-GS: Analyzing and Improving
            Gaussian Splatting Optimization.
            {" "}
            <a style={s.link} {...ext} href="https://arxiv.org/abs/2602.09999">arxiv.org/abs/2602.09999</a>
          </li>
          <li>
            Hanson et al., 2025 &mdash; Speedy-Splat: Fast 3D Gaussian
            Splatting with Sparse Pixels and Sparse Primitives.
            {" "}
            <a style={s.link} {...ext} href="https://speedysplat.github.io/">speedysplat.github.io</a>
          </li>
        </ul>
        <p style={s.p}><strong>Tools and libraries</strong>:</p>
        <ul style={s.ul}>
          <li>
            COLMAP &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://github.com/colmap/colmap">github.com/colmap/colmap</a>
          </li>
          <li>
            OpenSplat (pierotofy) &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://github.com/pierotofy/OpenSplat">github.com/pierotofy/OpenSplat</a>
          </li>
          <li>
            Faster-GS (nerficg-project Inria fork) &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://github.com/nerficg-project/faster-gaussian-splatting">github.com/nerficg-project/faster-gaussian-splatting</a>
          </li>
          <li>
            Three.js &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://threejs.org/">threejs.org</a>
          </li>
          <li>
            React Three Fiber &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://r3f.docs.pmnd.rs/">r3f.docs.pmnd.rs</a>
          </li>
          <li>
            drei (<code style={s.code}>&lt;Splat&gt;</code> component) &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://github.com/pmndrs/drei">github.com/pmndrs/drei</a>
          </li>
          <li>
            FastAPI (backend) &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://fastapi.tiangolo.com/">fastapi.tiangolo.com</a>
          </li>
          <li>
            OpenCV (frame extraction and filtering) &mdash;
            {" "}
            <a style={s.link} {...ext} href="https://opencv.org/">opencv.org</a>
          </li>
        </ul>
      </section>

      <section style={s.section}>
        <h2 style={s.h2}>6. Known limitations</h2>
        <ul style={s.ul}>
          <li>
            COLMAP's feature matching breaks down on reflective or
            transparent surfaces. A kitchen-blender scene we ran in early
            testing (45 frames) only registered 2 frames under COLMAP but
            all 45 under VGGT; switching SfM methods is the right move on
            scenes like that.
          </li>
          <li>
            OpenSplat doesn't emit per-iteration evaluation renders, so
            SSIM and LPIPS come back null in our metrics pipeline; PSNR
            is only captured when OpenSplat prints it to stdout. See the
            Testing Information section of the written report for how we
            handle the cross-backend comparison.
          </li>
          <li>
            The Faster-GS custom rasterizer currently fails to launch on
            L4 (sm_89) and B200 (sm_100) GPUs due to a CUDA shared-memory
            configuration that assumes Ampere-era hardware. Runs on those
            architectures fall back to the stock Inria rasterizer; the
            fused-Adam optimizer from Faster-GS still works.
          </li>
          <li>
            The full UI-driven pipeline dispatches training to HiPerGator,
            so running end-to-end through our app requires an HPG
            allocation. Without HPG access a user can still upload a PLY
            to the Converter page, view pre-trained splats on the Gallery
            page, and run COLMAP locally if they install it themselves —
            but can't launch a GPU training job through the app.
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
