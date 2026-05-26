// frontend/src/components/Header.jsx

import styles from './Header.module.css'

export default function Header({ online, mode, setMode }) {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>

        {/* Logo */}
        <div className={styles.logo}>
          <span className={styles.logoGlyph}>♜</span>
          <div>
            <div className={styles.logoTitle}>
              Grandmaster <em>Style</em> Simulator
            </div>
            <div className={styles.logoSub}>
              Behavioral Chess AI · Player-Conditioned Move Prediction
            </div>
          </div>
        </div>

        {/* Tabs */}
        <nav className={styles.tabs}>
          <button
            className={`${styles.tab} ${mode === 'predict' ? styles.active : ''}`}
            onClick={() => setMode('predict')}
          >
            <span className={styles.tabIcon}>⬡</span>
            Predict Moves
          </button>
          <button
            className={`${styles.tab} ${mode === 'compare' ? styles.active : ''}`}
            onClick={() => setMode('compare')}
          >
            <span className={styles.tabIcon}>⬡</span>
            Style Comparison
          </button>
        </nav>

        {/* Status */}
        <div className={styles.status}>
          <span className={`${styles.dot} ${online ? styles.online : styles.offline}`} />
          <span className={styles.statusText}>
            {online ? 'Model Online' : 'API Offline'}
          </span>
        </div>

      </div>
    </header>
  )
}