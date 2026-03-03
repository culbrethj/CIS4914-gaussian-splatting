import { useState } from 'react'
import { Canvas } from "@react-three/fiber"
import { Splat, OrbitControls } from "@react-three/drei"
import gaussModel from './assets/banana.splat'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  // .ply -> .splat converter: https://huggingface.co/spaces/dylanebert/ply-to-splat
  // .splat files only, .ply not supported

  // Click and drag to rotate
  // Ctrl + Click and drag to pan
  // Scroll to zoom in/out

  return (
    <>
      <Canvas>
        <OrbitControls />
        <Splat
          src={gaussModel}
          rotation={[0.75 * Math.PI, -.02 * Math.PI, .55 * Math.PI]}
          />
      </Canvas>
    </>
  )
}

export default App
