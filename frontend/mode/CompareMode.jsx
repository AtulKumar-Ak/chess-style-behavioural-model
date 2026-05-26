// frontend/mode/CompareMode.jsx

import { useState, useCallback, useRef } from 'react'
import { Chess } from 'chess.js'
import ChessBoard from '../src/components/ChessBoard'
import BoardControls from '../highlights/BoardControls'
import PlayerSelector from '../highlights/PlayerSelector'
import { getPlayerCfg, formatMove, tokenToPiece, PIECE_NAMES, PLAYER_CFG } from '../src/loading/constants'
import styles from './CompareMode.module.css'

export default function CompareMode({ api }) {
  const [game, setGame]           = useState(() => new Chess())
  const [moves, setMoves]         = useState([])
  const [lastMove, setLastMove]   = useState(null)
  const [cmpPlayers, setCmpPlayers] = useState(['MagnusCarlsen', 'Hikaru'])
  const [results, setResults]     = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  const gameRef = useRef(game)
  gameRef.current = game

  // Collect all predicted squares for board overlay
  const allPredictions = results
    ? results.flatMap(r => r.predictions.slice(0, 1))
    : []

  const handleMove = useCallback(({ from, to, promotion = 'q' }) => {
    const g = new Chess(gameRef.current.fen())
    const result = g.move({ from, to, promotion })
    if (!result) return false
    const newMoves = [...moves, result.lan || `${from}${to}`]
    setGame(g); setMoves(newMoves)
    setLastMove({ from, to }); setResults(null)
    return true
  }, [moves])

  const handleLoadUci = useCallback((parts) => {
    const g = new Chess()
    const valid = []
    for (const uci of parts) {
      const r = g.move({ from: uci.slice(0,2), to: uci.slice(2,4), promotion: uci[4]||'q' })
      if (!r) break
      valid.push(uci)
    }
    setGame(g); setMoves(valid)
    setLastMove(valid.length > 0 ? { from: valid[valid.length-1].slice(0,2), to: valid[valid.length-1].slice(2,4) } : null)
    setResults(null)
  }, [])

  const handleLoadPgn = useCallback(async (pgn) => {
    try {
      const data = await api.parsePgn(pgn)
      handleLoadUci(data.moves)
    } catch (e) {
      setError(e.message)
      setTimeout(() => setError(null), 3000)
    }
  }, [api, handleLoadUci])

  const handleReset = useCallback(() => {
    setGame(new Chess()); setMoves([]); setLastMove(null); setResults(null); setError(null)
  }, [])

  const handleUndo = useCallback(() => {
    const g = new Chess(game.fen())
    g.undo()
    const newMoves = moves.slice(0,-1)
    setGame(g); setMoves(newMoves)
    setLastMove(newMoves.length > 0 ? { from: newMoves[newMoves.length-1].slice(0,2), to: newMoves[newMoves.length-1].slice(2,4) } : null)
    setResults(null)
  }, [game, moves])

  const handleCompare = useCallback(async () => {
    if (!api.online || cmpPlayers.length === 0) return
    setLoading(true); setError(null)
    try {
      const data = await api.compare({ moves, playerNames: cmpPlayers, topK: 4 })
      setResults(data)
    } catch (e) {
      setError(e.message)
      setTimeout(() => setError(null), 4000)
    }
    setLoading(false)
  }, [api, moves, cmpPlayers])

  const togglePlayer = useCallback((p) => {
    setCmpPlayers(prev =>
      prev.includes(p) ? prev.filter(x => x !== p) : [...prev, p]
    )
    setResults(null)
  }, [])

  return (
    <div className={styles.layout}>

      {/* Left col */}
      <div className={styles.boardCol}>
        <ChessBoard
          game={game}
          onMove={handleMove}
          predictions={allPredictions}
          lastMove={lastMove}
          size={480}
        />
        <BoardControls
          moves={moves}
          onReset={handleReset}
          onUndo={handleUndo}
          onLoadUci={handleLoadUci}
          onLoadPgn={handleLoadPgn}
          onPredict={handleCompare}
          loading={loading}
          online={api.online}
        />
        {error && <div className={styles.errorBanner}>{error}</div>}
      </div>

      {/* Right col */}
      <div className={styles.sideCol}>

        {/* Player multi-select */}
        <PlayerSelector
          players={api.players}
          selected={cmpPlayers}
          onSelect={togglePlayer}
          label="Compare Players (select multiple)"
        />

        {/* Compare button */}
        <button
          className={styles.cmpBtn}
          onClick={handleCompare}
          disabled={loading || !api.online || cmpPlayers.length < 1}
        >
          {loading
            ? <><span className={styles.spinDots}><span/><span/><span/></span> Comparing…</>
            : `⚖ Compare ${cmpPlayers.length} Player${cmpPlayers.length !== 1 ? 's' : ''}`
          }
        </button>

        {/* Results */}
        {!api.online ? (
          <OfflineNotice />
        ) : !results ? (
          <IdleNotice loading={loading} />
        ) : (
          <CompareResults results={results} cmpPlayers={cmpPlayers} />
        )}

      </div>
    </div>
  )
}


function CompareResults({ results, cmpPlayers }) {
  // Find moves that multiple players agree on
  const allTopMoves = results.map(r => r.predictions[0]?.move).filter(Boolean)
  const agreedMove = allTopMoves.length > 1 && allTopMoves.every(m => m === allTopMoves[0])
    ? allTopMoves[0] : null

  return (
    <div className={styles.results}>
      {agreedMove && (
        <div className={styles.agreement}>
          <span className={styles.agreementIcon}>◈</span>
          All players agree: <strong>{agreedMove.slice(2,4)}</strong>
        </div>
      )}
      {results.map((r) => {
        const cfg = getPlayerCfg(r.player_name)
        return (
          <PlayerBlock
            key={r.player_name}
            playerKey={r.player_name}
            cfg={cfg}
            predictions={r.predictions}
          />
        )
      })}
    </div>
  )
}

function PlayerBlock({ playerKey, cfg, predictions }) {
  if (!predictions?.length) return null

  return (
    <div className={styles.playerBlock}>
      <div className={styles.pbHeader}>
        <span className={styles.pbDot} style={{ background: cfg.color }} />
        <span className={styles.pbFlag}>{cfg.flag}</span>
        <span className={styles.pbName} style={{ color: cfg.color }}>{cfg.display}</span>
      </div>
      <div className={styles.moveRows}>
        {predictions.map((pred, i) => {
          const label = formatMove(pred)
          const piece = tokenToPiece(pred.token)
          const pName = PIECE_NAMES[piece] || piece
          const pct   = Math.round(pred.probability * 100)
          return (
            <div key={pred.move} className={`${styles.moveRow} ${i === 0 ? styles.topRow : ''}`}>
              <span className={styles.mrUci}>{label}</span>
              <div className={styles.mrBarWrap}>
                <div
                  className={styles.mrBar}
                  style={{
                    width: `${pred.probability * 100}%`,
                    background: i === 0 ? cfg.color : 'var(--line2)',
                    animationDelay: `${i * 0.06}s`,
                  }}
                />
              </div>
              <span className={styles.mrPct}>{pct}%</span>
              <span className={styles.mrPiece}>{pName}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function OfflineNotice() {
  return (
    <div className={styles.notice}>
      <div className={styles.noticeIcon}>⚠</div>
      <pre className={styles.noticeText}>
        {`Start the FastAPI server:\n\nuvicorn backend.main:app --reload`}
      </pre>
    </div>
  )
}

function IdleNotice({ loading }) {
  return (
    <div className={styles.notice}>
      <div className={styles.noticeIcon}>⚖</div>
      <div className={styles.noticeText}>
        {loading ? 'Comparing styles…' : 'Select players above and click Compare\nto see how their move choices differ on this position'}
      </div>
    </div>
  )
}