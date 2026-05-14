import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from '../training_dynamics.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)