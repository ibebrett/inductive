import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d'
import { interpolateViridis } from 'd3-scale-chromatic'
import { MoleculeSvg } from './components/MoleculeSvg'

type Compound = { id: string; name: string; smiles: string; logS: number }
type Pair = {
  source: string
  target: string
  from_smiles: string
  to_smiles: string
  size: number
  rule_count: number
  delta_logS: number
}
type Graph = { nodes: Compound[]; links: Pair[] }

// force-graph replaces link endpoints with node objects once the simulation starts.
const endpointId = (end: string | { id?: string | number }) => (typeof end === 'object' ? String(end.id) : end)

// ESOL logS spans roughly -11.6 (insoluble) to +1.6 (very soluble).
const LOGS_MIN = -12
const LOGS_MAX = 2
const logSColor = (logS: number) => interpolateViridis((logS - LOGS_MIN) / (LOGS_MAX - LOGS_MIN))

const fragment = (smiles: string) => smiles.replaceAll('[*:1]', '*')

function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    if (!ref.current) return
    const obs = new ResizeObserver(([entry]) => setSize(entry.contentRect))
    obs.observe(ref.current)
    return () => obs.disconnect()
  }, [])
  return [ref, size] as const
}

export function EsolGraph() {
  const [maxSize, setMaxSize] = useState(2)
  const [graph, setGraph] = useState<Graph>()
  const [error, setError] = useState<string>()
  const [selectedId, setSelectedId] = useState<string>()
  const [containerRef, { width, height }] = useElementSize<HTMLDivElement>()
  const fg = useRef<ForceGraphMethods<Compound, Pair>>(undefined)

  useEffect(() => {
    fetch(`/api/esol/graph?max_size=${maxSize}`)
      .then(async (r) => (r.ok ? r.json() : Promise.reject((await r.json()).detail ?? r.statusText)))
      .then((g: Graph) => {
        setGraph(g)
        setError(undefined)
      })
      .catch((e) => setError(String(e)))
  }, [maxSize])

  const byId = useMemo(() => new Map(graph?.nodes.map((n) => [n.id, n])), [graph])
  const selected = selectedId ? byId.get(selectedId) : undefined

  // Neighbours of the selected compound, each pair oriented away from it.
  const neighbors = useMemo(() => {
    if (!graph || !selectedId) return []
    return graph.links
      .map((l) => ({ ...l, source: endpointId(l.source), target: endpointId(l.target) }))
      .filter((l) => l.source === selectedId || l.target === selectedId)
      .map((l) =>
        l.source === selectedId
          ? l
          : { ...l, source: l.target, target: l.source, from_smiles: l.to_smiles, to_smiles: l.from_smiles, delta_logS: -l.delta_logS },
      )
      .sort((a, b) => b.delta_logS - a.delta_logS)
  }, [graph, selectedId])
  const neighborIds = useMemo(() => new Set(neighbors.map((n) => n.target)), [neighbors])

  const select = (id: string) => {
    setSelectedId(id)
    const node = byId.get(id) as (Compound & { x?: number; y?: number }) | undefined
    if (node?.x !== undefined && node.y !== undefined) {
      fg.current?.centerAt(node.x, node.y, 500)
    }
  }

  return (
    <div className="viewer">
      <div className="graph" ref={containerRef}>
        <div className="toolbar">
          <label>
            Max transform size: {maxSize} heavy atom{maxSize > 1 ? 's' : ''}
            <input type="range" min={1} max={6} value={maxSize} onChange={(e) => setMaxSize(Number(e.target.value))} />
          </label>
          {graph && (
            <span>
              {graph.nodes.length} compounds · {graph.links.length} pairs
            </span>
          )}
          {error && <span className="error">{error}</span>}
        </div>
        {graph && width > 0 && (
          <ForceGraph2D<Compound, Pair>
            ref={fg}
            graphData={graph}
            width={width}
            height={height}
            nodeId="id"
            nodeLabel={(n) => `${n.name} (logS ${n.logS.toFixed(2)})`}
            nodeColor={(n) => logSColor(n.logS)}
            nodeRelSize={6}
            nodeVal={(n) => (n.id === selectedId ? 6 : neighborIds.has(n.id) ? 3 : 1)}
            // Generous hit area: at least ~10 screen px radius regardless of zoom.
            nodePointerAreaPaint={(n, color, ctx, scale) => {
              const node = n as Compound & { x: number; y: number }
              ctx.fillStyle = color
              ctx.beginPath()
              ctx.arc(node.x, node.y, Math.max(10 / scale, 8), 0, 2 * Math.PI)
              ctx.fill()
            }}
            linkColor={(l) =>
              selectedId && (endpointId(l.source) === selectedId || endpointId(l.target) === selectedId)
                ? '#e4572e'
                : 'rgba(0,0,0,0.15)'
            }
            linkLabel={(l) => `${fragment(l.from_smiles)} >> ${fragment(l.to_smiles)}`}
            onNodeClick={(n) => select(n.id)}
            onBackgroundClick={() => setSelectedId(undefined)}
            cooldownTime={5000}
          />
        )}
      </div>

      <aside className="panel">
        {selected ? (
          <>
            <MoleculeSvg smiles={selected.smiles} />
            <h2>{selected.name}</h2>
            <code>{selected.smiles}</code>
            <p>
              logS <strong>{selected.logS.toFixed(2)}</strong>
            </p>
            <h3>{neighbors.length} matched pairs</h3>
            <ul className="neighbors">
              {neighbors.map((p) => {
                const other = byId.get(p.target)!
                return (
                  <li key={p.target} onClick={() => select(p.target)}>
                    <code>
                      {fragment(p.from_smiles)} → {fragment(p.to_smiles)}
                    </code>
                    <span className={p.delta_logS >= 0 ? 'up' : 'down'}>
                      ΔlogS {p.delta_logS >= 0 ? '+' : ''}
                      {p.delta_logS.toFixed(2)}
                    </span>
                    <div className="muted">
                      {other.name} · transform seen in {p.rule_count} pairs
                    </div>
                  </li>
                )
              })}
            </ul>
          </>
        ) : (
          <div className="muted">
            <h2>ESOL matched-pair graph</h2>
            <p>
              Each node is a compound coloured by measured solubility (logS: yellow = soluble, purple = insoluble).
              Edges join compounds that differ by a single fragment swap.
            </p>
            <p>Click a compound to see its structure and neighbours.</p>
          </div>
        )}
      </aside>
    </div>
  )
}
