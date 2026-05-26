// frontend/highlights/PlayerSelector.jsx

import { getPlayerCfg, PLAYER_CFG } from '../src/loading/constants'
import styles from './PlayerSelector.module.css'

export default function PlayerSelector({ players, selected, onSelect, label = 'Select Grandmaster' }) {
  // Use API players if available, otherwise fall back to static config keys
  const displayPlayers = players.length > 0 ? players : Object.keys(PLAYER_CFG)

  return (
    <div className={styles.wrap}>
      <div className={styles.label}>{label}</div>
      <div className={styles.grid}>
        {displayPlayers.map(p => {
          const cfg = getPlayerCfg(p)
          const isSel = Array.isArray(selected) ? selected.includes(p) : selected === p
          return (
            <button
              key={p}
              className={`${styles.card} ${isSel ? styles.sel : ''}`}
              style={isSel ? { '--pc': cfg.color } : {}}
              onClick={() => onSelect(p)}
            >
              <div className={styles.emoji}>{cfg.emoji}</div>
              <div className={styles.info}>
                <div className={styles.name}>{cfg.short}</div>
                <div className={styles.title}>{cfg.title}</div>
              </div>
              <div className={styles.dot} style={{ background: cfg.color }} />
            </button>
          )
        })}
      </div>
    </div>
  )
}