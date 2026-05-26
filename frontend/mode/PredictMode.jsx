// frontend/mode/PredictMode.jsx

import { useState, useCallback, useRef } from 'react'
import { Chess } from 'chess.js'
import ChessBoard from '../src/components/ChessBoard'
import BoardControls from '../highlights/BoardControls'
import PlayerSelector from '../highlights/PlayerSelector'
import PredictionPanel from '../highlights/PredictionPanel'
import styles from './PredictMode.module.css'

export default function PredictMode({ api }) {
  const [game, setGame]             = useState(() => new Chess())
  const [moves, setMoves]           = useState([])
  const [lastMove, setLastMove]     = useState(null)
  const [selectedPlayer, setPlayer] = useState('MagnusCarlsen')
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)

  // Keep a ref to current game for stable callbacks
  const gameRef = useRef(game)
  gameRef.current = game

  const handleMove = useCallback(({ from, to, promotion = 'q' }) => {
    const g = new Chess(gameRef.current.fen())
    const result = g.move({ from, to, promotion })
    if (!result) return false

    const newMoves = [...moves, result.lan || `${from}${to}`]
    setGame(g)
    setMoves(newMoves)
    setLastMove({ from, to })
    setPredictions(null)
    return true
  }, [moves])

  const handleLoadUci = useCallback((parts) => {
    const g = new Chess()
    const valid = []
    for (const uci of parts) {
      const from = uci.slice(0, 2)
      const to   = uci.slice(2, 4)
      const promo = uci[4] || 'q'
      const r = g.move({ from, to, promotion: promo })
      if (!r) break
      valid.push(uci)
    }
    setGame(g)
    setMoves(valid)
    setLastMove(valid.length > 0 ? {
      from: valid[valid.length-1].slice(0,2),
      to:   valid[valid.length-1].slice(2,4),
    } : null)
    setPredictions(null)
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
    setGame(new Chess())
    setMoves([])
    setLastMove(null)
    setPredictions(null)
    setError(null)
  }, [])

  const handleUndo = useCallback(() => {
    const g = new Chess(game.fen())
    g.undo()
    const newMoves = moves.slice(0, -1)
    setGame(g)
    setMoves(newMoves)
    setLastMove(newMoves.length > 0 ? {
      from: newMoves[newMoves.length-1].slice(0,2),
      to:   newMoves[newMoves.length-1].slice(2,4),
    } : null)
    setPredictions(null)
  }, [game, moves])

  const handlePredict = useCallback(async () => {
    if (!api.online) return
    setLoading(true)
    setError(null)
    try {
      const data = await api.predict({
        moves: moves,
        playerName: selectedPlayer,
        topK: 5,
      })
      setPredictions(data.predictions)
    } catch (e) {
      setError(e.message)
      setTimeout(() => setError(null), 4000)
    }
    setLoading(false)
  }, [api, moves, selectedPlayer])

  const handlePlayPrediction = useCallback((uciMove) => {
    const moved = handleMove({
      from: uciMove.slice(0, 2),
      to:   uciMove.slice(2, 4),
      promotion: uciMove[4] || 'q',
    })
    if (moved) setPredictions(null)
  }, [handleMove])

  return (
    <div className={styles.layout}>

      {/* Left: board + controls */}
      <div className={styles.boardCol}>
        <ChessBoard
          game={game}
          onMove={handleMove}
          predictions={predictions || []}
          lastMove={lastMove}
          size={480}
        />
        <BoardControls
          moves={moves}
          onReset={handleReset}
          onUndo={handleUndo}
          onLoadUci={handleLoadUci}
          onLoadPgn={handleLoadPgn}
          onPredict={handlePredict}
          loading={loading}
          online={api.online}
        />
        {error && <div className={styles.errorBanner}>{error}</div>}
      </div>

      {/* Right: player selector + prediction panel */}
      <div className={styles.sideCol}>
        <PlayerSelector
          players={api.players}
          selected={selectedPlayer}
          onSelect={(p) => {
            setPlayer(p)
            setPredictions(null)
          }}
        />
        <PredictionPanel
          playerKey={selectedPlayer}
          predictions={predictions}
          loading={loading}
          online={api.online}
          onPlayMove={handlePlayPrediction}
        />
      </div>

    </div>
  )
}