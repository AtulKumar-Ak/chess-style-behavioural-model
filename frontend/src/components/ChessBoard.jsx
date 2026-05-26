// frontend/src/components/ChessBoard.jsx

import { useState, useCallback } from 'react'
import { Chessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import styles from './ChessBoard.module.css'

export default function ChessBoard({
  game,
  onMove,
  predictions = [],
  lastMove = null,
  size = 480,
}) {
  const [selectedSq, setSelectedSq] = useState(null)

  // Build custom square styles from predictions and last move
  const customSquareStyles = {}

  // Highlight last move
  if (lastMove) {
    const dimColor = 'rgba(107,174,214,0.22)'
    customSquareStyles[lastMove.from] = { background: dimColor }
    customSquareStyles[lastMove.to]   = { background: dimColor }
  }

  // Highlight predicted target squares (top 3 get decreasing intensity)
  predictions.slice(0, 5).forEach((pred, i) => {
    const to    = pred.move.slice(2, 4)
    const alpha = i === 0 ? 0.75 : i === 1 ? 0.45 : i === 2 ? 0.28 : 0.16
    customSquareStyles[to] = {
      background: `rgba(212,168,83,${alpha})`,
      borderRadius: i === 0 ? '0' : '50%',
    }
  })

  // Selected square highlight
  if (selectedSq) {
    customSquareStyles[selectedSq] = {
      background: 'rgba(212,168,83,0.6)',
    }
    // Show legal moves for selected piece
    const legalMoves = game.moves({ square: selectedSq, verbose: true })
    legalMoves.forEach(mv => {
      if (!customSquareStyles[mv.to]) {
        customSquareStyles[mv.to] = {
          background: 'radial-gradient(circle, rgba(0,0,0,0.25) 25%, transparent 25%)',
        }
      }
    })
  }

  const onSquareClick = useCallback((square) => {
    if (selectedSq === null) {
      const piece = game.get(square)
      if (piece && piece.color === (game.turn() === 'w' ? 'w' : 'b')) {
        setSelectedSq(square)
      }
    } else {
      if (square === selectedSq) {
        setSelectedSq(null)
        return
      }
      // try move
      const moved = onMove({ from: selectedSq, to: square, promotion: 'q' })
      setSelectedSq(null)
      if (!moved) {
        // maybe clicking a different own piece
        const piece = game.get(square)
        if (piece && piece.color === game.turn()) {
          setSelectedSq(square)
        }
      }
    }
  }, [selectedSq, game, onMove])

  const onPieceDrop = useCallback((from, to) => {
    setSelectedSq(null)
    return onMove({ from, to, promotion: 'q' })
  }, [onMove])

  return (
    <div className={styles.wrap} style={{ width: size, height: size }}>
      <Chessboard
        position={game.fen()}
        onSquareClick={onSquareClick}
        onPieceDrop={onPieceDrop}
        customSquareStyles={customSquareStyles}
        boardWidth={size}
        customDarkSquareStyle={{ backgroundColor: 'var(--sq-d)' }}
        customLightSquareStyle={{ backgroundColor: 'var(--sq-l)' }}
        customBoardStyle={{
          borderRadius: '4px',
          boxShadow: '0 12px 48px rgba(0,0,0,0.7)',
          border: '1px solid var(--line)',
        }}
        animationDuration={150}
      />
    </div>
  )
}