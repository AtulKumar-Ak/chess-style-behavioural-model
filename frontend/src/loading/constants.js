// frontend/src/loading/constants.js

// Player display config — add more as you expand the model
export const PLAYER_CFG = {
  MagnusCarlsen: {
    display: 'Magnus Carlsen',
    short:   'Magnus',
    title:   'World Champion',
    style:   'Positional maestro, endgame wizard',
    color:   '#d4a853',
    emoji:   '♟',
    flag:    '🇳🇴',
  },
  Hikaru: {
    display: 'Hikaru Nakamura',
    short:   'Hikaru',
    title:   'Speed Chess Legend',
    style:   'Tactical fireworks, blitz god',
    color:   '#6baed6',
    emoji:   '⚡',
    flag:    '🇺🇸',
  },
  alireza2003: {
    display: 'Alireza Firouzja',
    short:   'Alireza',
    title:   'Young Prodigy',
    style:   'Aggressive, sharp & unpredictable',
    color:   '#c97a9a',
    emoji:   '🔥',
    flag:    '🇫🇷',
  },
  lachesisQ: {
    display: 'Ian Nepomniachtchi',
    short:   'Nepo',
    title:   'Challenger',
    style:   'Dynamic attacker, opening theorist',
    color:   '#7ac99a',
    emoji:   '♞',
    flag:    '🇷🇺',
  },
}

// Fallback config for players not in the map above
export function getPlayerCfg(key) {
  return PLAYER_CFG[key] ?? {
    display: key,
    short:   key,
    title:   'Grandmaster',
    style:   'Elite player',
    color:   '#8888aa',
    emoji:   '♙',
    flag:    '🏁',
  }
}

export const PIECE_NAMES = {
  P: 'Pawn', N: 'Knight', B: 'Bishop', R: 'Rook', Q: 'Queen', K: 'King',
}

// Extract piece letter from piece-aware token like "P_e2e4" → "P"
export function tokenToPiece(token) {
  return token?.split('_')[0] ?? '?'
}

// Extract UCI from token like "P_e2e4" → "e2e4"
export function tokenToUci(token) {
  return token?.split('_')[1] ?? token
}

// Format move for display: "Qd5", "e4", etc.
export function formatMove(prediction) {
  if (!prediction) return ''
  const uci   = prediction.move
  const piece = tokenToPiece(prediction.token)
  const to    = uci.slice(2, 4)
  return piece === 'P' ? to : piece + to
}