import GaussViewer from './components/GaussViewer'
import VideoUpload from './components/VideoUpload'
import './App.css'

function App() {
  return (
    <div className="app-container">
      <h1 className="app-title">Team Gauss Project</h1>
      <VideoUpload />
      <GaussViewer />
    </div>
  )
}

export default App