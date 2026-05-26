// frontend/highlights/PredictionPanel.jsx

import { getPlayerCfg, formatMove, tokenToPiece, PIECE_NAMES } from '../src/loading/constants'
import styles from './PredictionPanel.module.css'

export default function PredictionPanel({
  playerKey,
  predictions,
  loading,
  online,
  onPlayMove,
}) {
  const cfg = getPlayerCfg(playerKey)

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.playerInfo}>
          <span className={styles.flag}>{cfg.flag}</span>
          <div>
            <div className={styles.playerName} style={{ color: cfg.color }}>
              {cfg.display}
            </div>
            <div className={styles.playerStyle}>{cfg.style}</div>
          </div>
        </div>
        {loading && (
          <div className={styles.loadingDots}>
            <span/><span/><span/>
          </div>
        )}
      </div>

      <div className={styles.divider} style={{ background: cfg.color }}/>

      <div className={styles.body}>
        {!online ? (
          <Notice type="offline" />
        ) : loading ? (
          <div className={styles.loadingMsg}>Consulting {cfg.short}'s style…</div>
        ) : !predictions ? (
          <Notice type="idle" name={cfg.short} />
        ) : predictions.length === 0 ? (
          <Notice type="empty" />
        ) : (
          <div className={styles.moveList}>
            {predictions.map((pred, i) => (
              <MoveCard
                key={pred.move}
                pred={pred}
                rank={i}
                color={cfg.color}
                onClick={() => onPlayMove(pred.move)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function MoveCard({ pred, rank, color, onClick }) {
  const label = formatMove(pred)
  const piece = tokenToPiece(pred.token)
  const pName = PIECE_NAMES[piece] || piece
  const pct   = Math.round(pred.probability * 100)
  const isTop = rank === 0

  return (
    <button
      className={`${styles.card} ${isTop ? styles.cardTop : ''}`}
      onClick={onClick}
      style={isTop ? { '--player-color': color } : {}}
    >
      <div className={styles.cardRank} style={{ color: isTop ? color : undefined }}>
        #{rank + 1}
      </div>
      <div className={styles.cardInfo}>
        <div className={styles.cardMove}>{label}</div>
        <div className={styles.cardDesc}>{pName} → {pred.move.slice(2,4)}</div>
      </div>
      <div className={styles.cardProb}>
        <div className={styles.cardPct}>{pct}%</div>
        <div className={styles.barWrap}>
          <div
            className={styles.bar}
            style={{
              width: `${pred.probability * 100}%`,
              background: isTop ? color : 'var(--line2)',
            }}
          />
        </div>
      </div>
    </button>
  )
}

function Notice({ type, name }) {
  const content = {
    offline: {
      icon: '⚠',
      title: 'API Offline',
      body: 'Start the FastAPI server:\n\nuvicorn backend.main:app --reload',
    },
    idle: {
      icon: '♟',
      title: 'Ready',
      body: `Make a move or click Predict\nto see how ${name || 'this player'} would respond`,
    },
    empty: {
      icon: '◇',
      title: 'No predictions',
      body: 'No legal moves found in vocabulary.\nTry a different position.',
    },
  }[type]

  return (
    <div className={styles.notice}>
      <div className={styles.noticeIcon}>{content.icon}</div>
      <div className={styles.noticeTitle}>{content.title}</div>
      <pre className={styles.noticeBody}>{content.body}</pre>
    </div>
  )
}