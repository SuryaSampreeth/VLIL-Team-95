import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { AUC_RANKS, DETECTION_RESULTS, DISEASES, ENHANCEMENT_LEADERBOARD } from '../lib/content'
import { useChartColors } from '../lib/hooks'
import { cx } from '../lib/utils'
import { Tabs } from './ui'

function ChartTooltip({ active, payload, label, unit = '', digits = 3 }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg bg-elev px-3 py-2 text-xs shadow-soft ring-1 ring-line/20">
      <p className="mb-1 font-semibold text-ink">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} className="flex items-center gap-2 text-muted">
          <span className="h-2 w-2 rounded-sm" style={{ background: p.color || p.payload.fill }} />
          {p.name}: <b className="tabular-nums text-ink">{Number(p.value).toFixed(digits)}{unit}</b>
        </p>
      ))}
    </div>
  )
}

const axisProps = (c) => ({ stroke: c.axis, tick: { fill: c.axis, fontSize: 11 }, tickLine: false, axisLine: { stroke: c.grid } })

/** Enhancement leaderboard: one metric at a time, our method highlighted. */
const ENH_METRICS = [
  { key: 'brisque', label: 'BRISQUE ↓', d: 2 },
  { key: 'entropy', label: 'Entropy ↑', d: 2, domain: [6, 8.2] },
  { key: 'sharpness', label: 'Sharpness ↑', d: 1 },
  { key: 'cii', label: 'CII ↑', d: 2 },
  { key: 'rank', label: 'Avg rank ↓', d: 1 },
]

export function LeaderboardChart() {
  const c = useChartColors()
  const [metric, setMetric] = useState('rank')
  const [view, setView] = useState('chart')
  const m = ENH_METRICS.find((x) => x.key === metric)
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <Tabs size="sm" value={metric} onChange={setMetric} tabs={ENH_METRICS.map((x) => ({ value: x.key, label: x.label }))} />
        <Tabs size="sm" value={view} onChange={setView} tabs={[{ value: 'chart', label: 'Chart' }, { value: 'table', label: 'Table' }]} />
      </div>
      {view === 'chart' ? (
        <div className="h-72" role="img" aria-label={`${m.label} for 8 enhancement methods; Attention U-Net + CLAHE highlighted`}>
          <ResponsiveContainer>
            <BarChart data={ENHANCEMENT_LEADERBOARD} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }} barCategoryGap={6}>
              <CartesianGrid horizontal={false} stroke={c.grid} />
              <XAxis type="number" {...axisProps(c)} domain={m.domain || ['auto', 'auto']} />
              <YAxis type="category" dataKey="method" width={130} {...axisProps(c)} />
              <Tooltip cursor={{ fill: c.grid, opacity: 0.4 }} content={<ChartTooltip digits={m.d} />} />
              <Bar dataKey={metric} name={m.label} radius={[0, 4, 4, 0]} isAnimationActive>
                {ENHANCEMENT_LEADERBOARD.map((r) => (
                  <Cell key={r.method} fill={r.ours ? c.series[0] : c.muted} />
                ))}
                <LabelList dataKey={metric} position="right" fill={c.axis} fontSize={11} formatter={(v) => Number(v).toFixed(m.d)} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl ring-1 ring-line/15">
          <table className="w-full min-w-[520px] text-sm">
            <thead className="bg-ink/[0.04] text-left text-[11px] uppercase tracking-wider text-muted">
              <tr>
                {['Method', 'BRISQUE ↓', 'Entropy ↑', 'Sharpness ↑', 'CII ↑', 'Avg rank'].map((h) => (
                  <th key={h} className="px-3 py-2 font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line/10 tabular-nums">
              {ENHANCEMENT_LEADERBOARD.map((r) => (
                <tr key={r.method} className={cx(r.ours && 'bg-primary/[0.08] font-semibold')}>
                  <td className="px-3 py-2">{r.method}</td>
                  <td className="px-3 py-2">{r.brisque}</td>
                  <td className="px-3 py-2">{r.entropy}</td>
                  <td className="px-3 py-2">{r.sharpness}</td>
                  <td className="px-3 py-2">{r.cii}</td>
                  <td className="px-3 py-2">{r.rank}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-2 text-xs text-muted">Mean over 23 low-quality CheXpert images (TEAM_95_PR2, slide 9).</p>
    </div>
  )
}

/** Test AUC or F1 per disease, grouped by the 4 detection techniques (6k per disease). */
export function DetectionChart() {
  const c = useChartColors()
  const [metric, setMetric] = useState('auc')
  const data = DISEASES.map((d, i) => Object.fromEntries([['disease', d], ...DETECTION_RESULTS.map((r) => [r.model, r[metric][i]])]))
  return (
    <div>
      <Tabs size="sm" className="mb-4" value={metric} onChange={setMetric} tabs={[{ value: 'auc', label: 'ROC-AUC' }, { value: 'f1', label: 'F1' }]} />
      <div className="h-72" role="img" aria-label={`Test ${metric.toUpperCase()} per disease for four detection models`}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ left: -8, right: 8, top: 8 }} barGap={2} barCategoryGap="22%">
            <CartesianGrid vertical={false} stroke={c.grid} />
            <XAxis dataKey="disease" {...axisProps(c)} />
            <YAxis domain={metric === 'auc' ? [0.7, 0.86] : [0.6, 0.8]} {...axisProps(c)} />
            <Tooltip cursor={{ fill: c.grid, opacity: 0.4 }} content={<ChartTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: c.axis }} iconType="square" iconSize={10} />
            {DETECTION_RESULTS.map((r, i) => (
              <Bar key={r.model} dataKey={r.model} fill={c.series[i]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-muted">Held-out test set, 6k images per disease; thresholds tuned on validation only.</p>
    </div>
  )
}

/** AUC rank per disease x data scale, as an HTML grid (sequential: darker = better). */
export function RankGrid() {
  const shade = { 1: 'bg-primary text-white', 2: 'bg-primary/60 text-white', 3: 'bg-primary/25 text-ink', 4: 'bg-primary/[0.08] text-muted' }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[620px] border-separate border-spacing-1 text-center text-sm">
        <caption className="mb-2 text-left text-xs text-muted">AUC rank per disease × data scale (1 = best) — TEAM_95_PR2, slide 17</caption>
        <thead>
          <tr>
            <th />
            {AUC_RANKS.columns.map((col) => (
              <th key={col} scope="col" className="px-1 pb-1 text-[11px] font-medium text-muted">{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {AUC_RANKS.rows.map((r) => (
            <tr key={r.model}>
              <th scope="row" className="whitespace-nowrap pr-2 text-left text-xs font-semibold">{r.model}</th>
              {r.ranks.map((v, i) => (
                <td key={i} className={cx('rounded-md py-2 font-display font-semibold tabular-nums', shade[v])} title={`${r.model} · ${AUC_RANKS.columns[i]}: rank ${v}`}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Generic two-series grouped bar (e.g. GPT-2 vs Qwen, proposed vs previous). */
export function PairChart({ data, a, b, unit = '', digits = 3, domain, label }) {
  const c = useChartColors()
  return (
    <div className="h-64" role="img" aria-label={label}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: -8, right: 8, top: 16 }} barGap={2}>
          <CartesianGrid vertical={false} stroke={c.grid} />
          <XAxis dataKey="metric" {...axisProps(c)} tick={{ fill: c.axis, fontSize: 10 }} interval={0} />
          <YAxis domain={domain} {...axisProps(c)} />
          <Tooltip cursor={{ fill: c.grid, opacity: 0.4 }} content={<ChartTooltip unit={unit} digits={digits} />} />
          <Legend wrapperStyle={{ fontSize: 12, color: c.axis }} iconType="square" iconSize={10} />
          <Bar dataKey={a.key} name={a.name} fill={c.series[0]} radius={[4, 4, 0, 0]}>
            <LabelList dataKey={a.key} position="top" fill={c.axis} fontSize={10} formatter={(v) => Number(v).toFixed(unit === '%' ? 1 : 2)} />
          </Bar>
          <Bar dataKey={b.key} name={b.name} fill={c.series[1]} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Animation Variants ────────────────────────────────────────
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
  transition: { duration: 0.4, ease: 'easeOut' }
};

const staggerContainer = {
  animate: {
    transition: { staggerChildren: 0.08, delayChildren: 0.15 }
  }
};

const scaleOnHover = {
  whileHover: { scale: 1.03 },
  whileTap: { scale: 0.98 },
  transition: { type: 'spring', stiffness: 400 }
};
