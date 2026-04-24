import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { sampleReplay, type ReplayData, type ReplayTick } from './sampleTrace'

const CELL_SIZE = 110
const STATION_ART: Record<string, string> = { onion: '🧅', dish: '🍽️', pot: '🍲', serve: '✨' }
const HELD_ART: Record<string, string> = { onion: '🧅', dish: '🍽️', soup: '🍲', 'soup in plate': '🍛' }
const ORIENTATION_CLASS: Record<string, string> = {
  '0,-1': 'north',
  '0,1': 'south',
  '1,0': 'east',
  '-1,0': 'west',
  '0,0': 'south',
}

function isStoryBeat(tick: ReplayTick) {
  return tick.scoreDelta > 0 || tick.events.length > 0 || tick.goal_changes.length > 0
}

function formatTickLabel(tick: number) {
  return `Tick ${tick.toString().padStart(3, '0')}`
}

async function loadReplay(): Promise<{ replay: ReplayData; source: string }> {
  try {
    const response = await fetch(`/traces/latest.json?ts=${Date.now()}`)
    if (!response.ok) throw new Error(`Trace not found (${response.status})`)
    return { replay: (await response.json()) as ReplayData, source: 'live trace' }
  } catch {
    return { replay: sampleReplay, source: 'mock trace' }
  }
}

function App() {
  const [replay, setReplay] = useState<ReplayData>(sampleReplay)
  const [sourceLabel, setSourceLabel] = useState('mock trace')
  const [tickIndex, setTickIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [storyOnly, setStoryOnly] = useState(false)

  useEffect(() => {
    let cancelled = false
    loadReplay().then(({ replay: nextReplay, source }) => {
      if (cancelled) return
      setReplay(nextReplay)
      setSourceLabel(source)
      setTickIndex(0)
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!isPlaying) return
    const handle = window.setInterval(() => {
      setTickIndex((current) => (current >= replay.frames.length - 1 ? 0 : current + 1))
    }, 900)
    return () => window.clearInterval(handle)
  }, [isPlaying, replay.frames.length])

  const currentFrame = replay.frames[tickIndex] ?? replay.frames[0]
  const currentTickEvent = replay.tick_events[Math.max(0, tickIndex - 1)]
  const storyBeats = useMemo(() => replay.tick_events.filter((tick) => isStoryBeat(tick)), [replay.tick_events])
  const shownEvents = useMemo(() => {
    const base = storyOnly ? storyBeats : replay.tick_events
    if (base.length <= 8) return base

    const anchorIndex = Math.max(
      0,
      base.findIndex((entry) => entry.tick >= currentFrame.tick),
    )
    const start = Math.max(0, anchorIndex - 2)
    return base.slice(start, start + 8)
  }, [currentFrame.tick, replay.tick_events, storyBeats, storyOnly])

  return (
    <main className="page-shell">
      <section className="playground">
        <div className="board-card">
          <div className="board-header">
            <div>
              <p className="board-title">Kitchen Trail</p>
              <h2>{currentTickEvent?.headline ?? 'The chefs step into the kitchen.'}</h2>
            </div>
            <button
              type="button"
              className="secondary-button"
              onClick={async () => {
                const { replay: nextReplay, source } = await loadReplay()
                setReplay(nextReplay)
                setSourceLabel(source)
                setTickIndex(0)
              }}
            >
              Reload trace
            </button>
          </div>

          <div className="board-scene">
            <div
              className="board-grid"
              style={{ width: `${replay.layout.width * CELL_SIZE}px`, height: `${replay.layout.height * CELL_SIZE}px` }}
            >
              {replay.layout.terrain.flat().map((tile) => (
                <div
                  key={`${tile.position[0]}-${tile.position[1]}`}
                  className={`tile tile-${tile.theme}`}
                  style={{ left: `${tile.position[0] * CELL_SIZE}px`, top: `${tile.position[1] * CELL_SIZE}px` }}
                >
                  {tile.terrain !== ' ' && <span className="tile-glyph">{STATION_ART[tile.theme] ?? ''}</span>}
                </div>
              ))}

              {currentFrame.pots.map((pot) => (
                <div
                  key={`pot-${pot.position.join('-')}`}
                  className={`pot-status stage-${pot.stage}`}
                  style={{ left: `${pot.position[0] * CELL_SIZE + 12}px`, top: `${pot.position[1] * CELL_SIZE + 12}px` }}
                >
                  <span>{pot.ingredient_count}/3</span>
                  <strong>{pot.stage}</strong>
                </div>
              ))}

              {currentFrame.counter_objects.map((object, index) => (
                <div
                  key={`${object.name}-${index}`}
                  className="counter-item"
                  style={{ left: `${object.position[0] * CELL_SIZE + 18}px`, top: `${object.position[1] * CELL_SIZE + 18}px` }}
                >
                  {HELD_ART[object.name] ?? '📦'}
                </div>
              ))}

              {currentFrame.players.map((player) => (
                <div
                  key={player.id}
                  className={`chef-token chef-${player.id} ${ORIENTATION_CLASS[player.orientation.join(',')]}`}
                  style={{ left: `${player.position[0] * CELL_SIZE + 10}px`, top: `${player.position[1] * CELL_SIZE + 10}px` }}
                >
                  <span className="chef-face">{player.id === 0 ? '🌼' : '🌿'}</span>
                  {player.held_object && <span className="held-bubble">{HELD_ART[player.held_object.name] ?? '📦'}</span>}
                  <span className="chef-name">{player.name}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="control-bar">
            <div className="transport">
              <button type="button" onClick={() => setTickIndex((tick) => Math.max(0, tick - 1))}>◀</button>
              <button type="button" className="play-button" onClick={() => setIsPlaying((value) => !value)}>
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button type="button" onClick={() => setTickIndex((tick) => Math.min(replay.frames.length - 1, tick + 1))}>▶</button>
            </div>

            <label className="timeline">
              <span>{formatTickLabel(currentFrame.tick)}</span>
              <input type="range" min={0} max={Math.max(0, replay.frames.length - 1)} value={tickIndex} onChange={(event) => setTickIndex(Number(event.target.value))} />
            </label>

            <label className="story-toggle">
              <input type="checkbox" checked={storyOnly} onChange={(event) => setStoryOnly(event.target.checked)} />
              Story beats only
            </label>
          </div>
        </div>

        <aside className="side-panel">
          <section className="companion-card">
            <p className="panel-label">Party Notes</p>
            <div className="party-grid">
              {currentFrame.players.map((player) => (
                <article key={player.id} className={`party-member tone-${player.id}`}>
                  <div className="member-header">
                    <span className="portrait">{player.id === 0 ? '🌼' : '🌿'}</span>
                    <div>
                      <h3>{player.name}</h3>
                      <p>{player.goal_label}</p>
                    </div>
                  </div>
                  <p className="member-fact">
                    Holding {player.held_object ? player.held_object.name : 'nothing'} and facing {ORIENTATION_CLASS[player.orientation.join(',')]}.
                  </p>
                </article>
              ))}
            </div>
          </section>

          <section className="companion-card">
            <div className="log-header">
              <p className="panel-label">Adventure Log</p>
              <span>{storyOnly ? `${storyBeats.length} beats` : `${replay.tick_events.length} ticks`}</span>
            </div>
            <div className="log-list">
              {shownEvents.map((tick) => (
                <button key={tick.tick} type="button" className={`log-item ${tick.tick === currentFrame.tick ? 'active' : ''}`} onClick={() => setTickIndex(tick.tick)}>
                  <span className="log-tick">{formatTickLabel(tick.tick)}</span>
                  <strong>{tick.headline}</strong>
                  <span className="log-meta">{tick.scoreDelta > 0 ? `+${tick.scoreDelta} points` : `${tick.events.length} event(s)`}</span>
                </button>
              ))}
            </div>
          </section>
        </aside>
      </section>
    </main>
  )
}

export default App
