// frontend/src/hooks/useApi.js

import { useState, useEffect, useCallback } from 'react'

const BASE = 'http://127.0.0.1:8000'

export function useApi() {
  const [online, setOnline]   = useState(false)
  const [players, setPlayers] = useState([])

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(2500) })
      if (r.ok) {
        setOnline(true)
        return true
      }
    } catch {}
    setOnline(false)
    return false
  }, [])

  const loadPlayers = useCallback(async () => {
    try {
      const r = await fetch(`${BASE}/players`)
      if (r.ok) {
        const d = await r.json()
        setPlayers(d.players || [])
      }
    } catch {}
  }, [])

  useEffect(() => {
    checkHealth().then(ok => { if (ok) loadPlayers() })
    const id = setInterval(async () => {
      const ok = await checkHealth()
      if (ok && players.length === 0) loadPlayers()
    }, 8000)
    return () => clearInterval(id)
  }, [])  // eslint-disable-line

  const predict = useCallback(async ({ moves, playerName, topK = 5 }) => {
    const r = await fetch(`${BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ moves, player_name: playerName, top_k: topK }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }))
      throw new Error(err.detail || 'Prediction failed')
    }
    return r.json()  // { predictions, player_name, move_count }
  }, [])

  const compare = useCallback(async ({ moves, playerNames, topK = 3 }) => {
    const results = await Promise.all(
      playerNames.map(name => predict({ moves, playerName: name, topK }))
    )
    return results  // array of { predictions, player_name, move_count }
  }, [predict])

  const parsePgn = useCallback(async (pgn) => {
    const r = await fetch(`${BASE}/parse-pgn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pgn }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }))
      throw new Error(err.detail || 'PGN parse failed')
    }
    return r.json()  // { moves }
  }, [])

  return { online, players, checkHealth, predict, compare, parsePgn }
}