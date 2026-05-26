// frontend/src/App.jsx

import { useState } from 'react'
import { useApi } from './hooks/useApi'
import Header from './components/Header'
import PredictMode from '../mode/PredictMode'
import CompareMode from '../mode/CompareMode'
import './App.css'

export default function App() {
  const [mode, setMode] = useState('predict') // 'predict' | 'compare'
  const api = useApi()

  return (
    <div className="app-shell">
      <Header online={api.online} mode={mode} setMode={setMode} />
      <main className="app-main">
        {mode === 'predict'
          ? <PredictMode api={api} />
          : <CompareMode api={api} />
        }
      </main>
    </div>
  )
}