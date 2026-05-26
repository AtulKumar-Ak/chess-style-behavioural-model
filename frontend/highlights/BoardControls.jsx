// frontend/highlights/BoardControls.jsx

import { useState } from 'react'
import styles from './BoardControls.module.css'

export default function BoardControls({
  moves,
  onReset,
  onUndo,
  onLoadUci,
  onLoadPgn,
  onPredict,
  loading,
  online,
}) {
  const [inputVal, setInputVal] = useState('')
  const [inputMode, setInputMode] = useState('uci') // 'uci' | 'pgn'

  const handleLoad = () => {
    if (!inputVal.trim()) return
    if (inputMode === 'pgn') {
      onLoadPgn(inputVal.trim())
    } else {
      const parts = inputVal.trim().split(/\s+/).filter(m => /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(m))
      if (parts.length) onLoadUci(parts)
    }
    setInputVal('')
  }

  return (
    <div className={styles.wrap}>

      {/* Input row */}
      <div className={styles.inputRow}>
        <div className={styles.modePills}>
          <button
            className={`${styles.pill} ${inputMode === 'uci' ? styles.active : ''}`}
            onClick={() => setInputMode('uci')}
          >UCI</button>
          <button
            className={`${styles.pill} ${inputMode === 'pgn' ? styles.active : ''}`}
            onClick={() => setInputMode('pgn')}
          >PGN</button>
        </div>
        <input
          className={styles.input}
          value={inputVal}
          onChange={e => setInputVal(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleLoad()}
          placeholder={
            inputMode === 'uci'
              ? 'Paste UCI moves: e2e4 e7e5 g1f3 …'
              : 'Paste PGN text, then press Enter'
          }
          spellCheck={false}
        />
        <button className={styles.btnLoad} onClick={handleLoad} disabled={!inputVal.trim()}>
          Load
        </button>
      </div>

      {/* Action row */}
      <div className={styles.actionRow}>
        <button className={styles.btn} onClick={onReset}>Reset</button>
        <button className={styles.btn} onClick={onUndo} disabled={moves.length === 0}>
          ← Undo
        </button>
        <button
          className={`${styles.btn} ${styles.btnPredict}`}
          onClick={onPredict}
          disabled={loading || !online}
        >
          {loading
            ? <span className={styles.spinner}><span/><span/><span/></span>
            : '▶ Predict'
          }
        </button>
        <span className={styles.moveCount}>
          {moves.length === 0 ? 'Opening' : `Move ${Math.ceil(moves.length / 2)}`}
        </span>
      </div>

      {/* Move history */}
      <div className={styles.history}>
        {moves.length === 0
          ? <span className={styles.historyEmpty}>No moves yet — click pieces to play</span>
          : moves.map((m, i) => (
              <span key={i} className={styles.historyMove}>
                {i % 2 === 0 && <span className={styles.moveNum}>{Math.floor(i/2)+1}.</span>}
                <span className={styles.moveUci}>{m}</span>
              </span>
            ))
        }
      </div>

    </div>
  )
}