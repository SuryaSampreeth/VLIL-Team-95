import { motion } from 'framer-motion'
import {
  CheckCircle2,
  Clock,
  Columns2,
  Copy,
  Cpu,
  Download,
  FileDown,
  ImageUp,
  Layers,
  Loader2,
  Play,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  XCircle,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { DISEASES } from '../lib/content'
import { useBackend, useElapsed } from '../lib/hooks'
import { ACCEPT, MAX_MB, cx, download, fmt, pct, sampleFile, slug, validateFile } from '../lib/utils'
import { Badge, Card, EmptyState, Gauge, ModeBadge, Notice, Skeleton, Tabs } from './ui'

export const SAMPLES = [1, 2, 3, 4].map((i) => ({ url: `/samples/sample_${i}.jpg`, name: `sample_${i}.jpg` }))

export const DISEASE_TONE = {
  Cardiomegaly: { text: 'text-dz-cardio', bg: 'bg-dz-cardio', ring: 'ring-dz-cardio/40', var: '--d-cardio' },
  Edema: { text: 'text-dz-edema', bg: 'bg-dz-edema', ring: 'ring-dz-edema/40', var: '--d-edema' },
  'Pleural Effusion': { text: 'text-dz-effusion', bg: 'bg-dz-effusion', ring: 'ring-dz-effusion/40', var: '--d-effusion' },
}

// ── upload ───────────────────────────────────────────────────────────────────
export function Uploader({ file, onFile, disabled, title = 'Chest X-ray', scanning }) {
  const [drag, setDrag] = useState(false)
  const [preview, setPreview] = useState(null)
  const input = useRef(null)

  useEffect(() => {
    if (!file) return setPreview(null)
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const accept = (f) => {
    const err = validateFile(f)
    if (err) return toast.error('File rejected', { description: err })
    onFile(f)
    toast.success('X-ray loaded', { description: `${f.name} · ${(f.size / 1024).toFixed(0)} KB` })
  }
  const pickSample = async (s) => {
    try {
      onFile(await sampleFile(s.url, s.name))
      toast.success('Sample loaded', { description: s.name })
    } catch {
      toast.error('Could not load the sample image')
    }
  }

  return (
    <Card className="p-4 sm:p-5">
      <div className="mb-3 flex items-center justify-between">
        <p className="label">{title}</p>
        {file && !disabled && (
          <button type="button" onClick={() => onFile(null)} className="inline-flex items-center gap-1 text-xs text-muted hover:text-bad">
            <Trash2 size={13} /> Remove
          </button>
        )}
      </div>

      {preview && file ? (
        <div className="relative overflow-hidden rounded-xl bg-black ring-1 ring-line/15">
          <img src={preview} alt={`Uploaded chest X-ray ${file?.name}`} className="mx-auto aspect-square w-full object-contain" />
          {scanning && (
            <div className="pointer-events-none absolute inset-0" aria-hidden>
              <div className="absolute inset-0 bg-primary/10" />
              <div className="absolute inset-x-0 h-16 -translate-y-1/2 animate-scan bg-gradient-to-b from-transparent via-primary/40 to-transparent" />
              <div className="absolute inset-x-0 h-px animate-scan bg-primary shadow-[0_0_12px_rgb(var(--primary))]" />
            </div>
          )}
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-2 bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-8 text-xs text-white/90">
            <span className="truncate">{file.name}</span>
            <span className="shrink-0 tabular-nums">{(file.size / 1024).toFixed(0)} KB</span>
          </div>
        </div>
      ) : (
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload a chest X-ray: click or drop a file"
          onClick={() => !disabled && input.current?.click()}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), input.current?.click())}
          onDragOver={(e) => (e.preventDefault(), setDrag(true))}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDrag(false)
            if (!disabled && e.dataTransfer.files[0]) accept(e.dataTransfer.files[0])
          }}
          className={cx(
            'group flex aspect-square w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 text-center transition',
            drag ? 'border-primary bg-primary/[0.08]' : 'hairline hover:border-primary/60 hover:bg-primary/[0.04]',
          )}
        >
          <motion.div animate={drag ? { scale: 1.12, y: -4 } : { scale: 1, y: 0 }} className="mb-4 grid h-16 w-16 place-items-center rounded-2xl bg-primary/10 text-primary">
            <ImageUp size={30} />
          </motion.div>
          <p className="font-display font-semibold">{drag ? 'Drop to upload' : 'Drag & drop an X-ray'}</p>
          <p className="mt-1 text-sm text-muted">
            or <span className="text-primary underline-offset-2 group-hover:underline">browse files</span>
          </p>
          <p className="mt-3 text-xs text-muted">
            {ACCEPT.join(' ')} · max {MAX_MB} MB
          </p>
        </div>
      )}
      <input ref={input} type="file" accept={ACCEPT.join(',')} className="hidden" onChange={(e) => (e.target.files[0] && accept(e.target.files[0]), (e.target.value = ''))} />

      <div className="mt-4">
        <p className="mb-2 text-xs text-muted">Or try a sample X-ray</p>
        <div className="grid grid-cols-4 gap-2">
          {SAMPLES.map((s, i) => (
            <button
              key={s.url}
              type="button"
              disabled={disabled}
              onClick={() => pickSample(s)}
              className={cx(
                'overflow-hidden rounded-lg bg-black ring-1 transition hover:ring-primary disabled:opacity-50',
                file?.name === s.name ? 'ring-2 ring-primary' : 'ring-line/20',
              )}
              aria-label={`Use sample X-ray ${i + 1}`}
            >
              <img src={s.url} alt="" className="aspect-square w-full object-cover opacity-90 transition hover:opacity-100" />
            </button>
          ))}
        </div>
      </div>
    </Card>
  )
}

// ── model selection ──────────────────────────────────────────────────────────
export function ModelPicker({ stage, state, value, onChange, disabled, compact }) {
  const { models, loading, error } = state
  if (loading)
    return (
      <div className="grid gap-2">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-16" />
        ))}
      </div>
    )
  if (error) return <Notice tone="bad">Model list unavailable — the backend is offline.</Notice>
  return (
    <div role="radiogroup" aria-label={`Stage ${stage} model`} className={cx('grid gap-2', !compact && 'sm:grid-cols-2')}>
      {models.map((m) => {
        const active = value === m.id
        return (
          <label
            key={m.id}
            className={cx(
              'relative flex cursor-pointer flex-col gap-1 rounded-xl p-3 ring-1 transition',
              active ? 'bg-primary/[0.08] ring-2 ring-primary' : 'bg-ink/[0.02] ring-line/15 hover:ring-primary/50',
              disabled && 'pointer-events-none opacity-60',
            )}
          >
            <input type="radio" name={`model-${stage}`} value={m.id} checked={active} onChange={() => onChange(m.id)} className="sr-only" disabled={disabled} />
            <span className="flex items-start justify-between gap-2">
              <span className="font-display text-sm font-semibold leading-snug">{m.name}</span>
              <span className={cx('mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full ring-2', active ? 'ring-primary' : 'ring-line/40')}>
                {active && <span className="h-2 w-2 rounded-full bg-primary" />}
              </span>
            </span>
            <span className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">{m.family}</span>
              {m.proposed && <Badge tone="primary">Ours</Badge>}
              {!m.real_available && <Badge tone="warn">mock only</Badge>}
            </span>
            {stage === 1 && m.benchmark && !compact && (
              <span className="mt-1 grid grid-cols-3 gap-1 text-[11px] text-muted">
                <span>
                  BRISQUE <b className="text-ink">{m.benchmark.brisque}</b>
                </span>
                <span>
                  CII <b className="text-ink">{m.benchmark.cii}</b>
                </span>
                <span>
                  Rank <b className="text-ink">{m.benchmark.avg_rank}</b>
                </span>
              </span>
            )}
            {stage === 2 && m.metrics && !compact && (
              <span className="mt-1 grid grid-cols-3 gap-1 text-[11px] text-muted">
                {m.metrics.map((x) => (
                  <span key={x.disease} title={`${x.disease}: AUC ${x.auc}, F1 ${x.f1}`}>
                    {x.disease.split(' ').pop().slice(0, 5)}. <b className="text-ink">{x.auc.toFixed(3)}</b>
                  </span>
                ))}
              </span>
            )}
          </label>
        )
      })}
    </div>
  )
}

export function ModelInfo({ model }) {
  if (!model) return null
  return (
    <p className="text-xs leading-relaxed text-muted">
      {model.description}
      {model.weights_file && (
        <>
          {' '}
          <code className="rounded bg-ink/5 px-1 py-0.5 font-mono text-[11px]">{model.weights_file}</code>
        </>
      )}
    </p>
  )
}

// ── run button with elapsed timer ────────────────────────────────────────────
export function RunButton({ onRun, onCancel, running, disabled, label = 'Run', hint }) {
  const ms = useElapsed(running)
  const { status } = useBackend()
  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <button type="button" className="btn-primary flex-1 py-3 text-base" onClick={onRun} disabled={disabled || running || status === 'offline'}>
          {running ? (
            <>
              <Loader2 size={18} className="animate-spin" /> Processing… <span className="tabular-nums opacity-80">{(ms / 1000).toFixed(1)}s</span>
            </>
          ) : (
            <>
              <Play size={18} /> {label}
            </>
          )}
        </button>
        {running && onCancel && (
          <button type="button" className="btn-ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
      {status === 'offline' ? (
        <p className="text-xs text-bad">Backend offline — start it to run the models (see README).</p>
      ) : (
        hint && <p className="flex items-center gap-1.5 text-xs text-muted"><Clock size={12} /> {hint}</p>
      )}
    </div>
  )
}

export function ResultHeader({ title, result, children }) {
  return (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-display text-xl font-semibold">{title}</h2>
        <ModeBadge mode={result?.mode} />
        {result?.timing_ms != null && (
          <Badge icon={Cpu}>{result.timing_ms >= 1000 ? `${(result.timing_ms / 1000).toFixed(1)} s` : `${result.timing_ms} ms`}</Badge>
        )}
      </div>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  )
}

// ── before / after slider ────────────────────────────────────────────────────
export function CompareSlider({ before, after, beforeLabel = 'Original', afterLabel = 'Enhanced' }) {
  const [pos, setPos] = useState(50)
  const box = useRef(null)
  const dragging = useRef(false)
  const move = (clientX) => {
    const r = box.current.getBoundingClientRect()
    setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)))
  }
  useEffect(() => {
    const up = () => (dragging.current = false)
    const mv = (e) => dragging.current && move(e.touches ? e.touches[0].clientX : e.clientX)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointermove', mv)
    window.addEventListener('touchmove', mv, { passive: true })
    return () => {
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointermove', mv)
      window.removeEventListener('touchmove', mv)
    }
  }, [])
  return (
    <div
      ref={box}
      className="relative aspect-square w-full touch-none select-none overflow-hidden rounded-xl bg-black ring-1 ring-line/15"
      onPointerDown={(e) => {
        dragging.current = true
        move(e.clientX)
      }}
    >
      <img src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      </div>
      <span className="absolute left-3 top-3 rounded-md bg-black/60 px-2 py-1 text-xs font-medium text-white">{beforeLabel}</span>
      <span className="absolute right-3 top-3 rounded-md bg-primary/80 px-2 py-1 text-xs font-medium text-white">{afterLabel}</span>
      <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -ml-px w-0.5 bg-white shadow-[0_0_10px_rgba(0,0,0,0.6)]" />
        <div className="absolute top-1/2 -ml-5 -mt-5 grid h-10 w-10 place-items-center rounded-full bg-white text-slate-900 shadow-lg">
          <Columns2 size={18} />
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Comparison slider: drag to compare original and enhanced"
        className="absolute inset-x-0 bottom-0 h-full w-full cursor-ew-resize opacity-0"
      />
    </div>
  )
}

// ── per-disease probability bars ─────────────────────────────────────────────
export function ProbabilityBars({ diseases, selected, onSelect }) {
  return (
    <div className="space-y-3" role="list">
      {diseases.map((d, i) => {
        const tone = DISEASE_TONE[d.name] || DISEASE_TONE.Cardiomegaly
        const active = selected === d.name
        return (
          <button
            key={d.name}
            type="button"
            role="listitem"
            onClick={() => onSelect?.(d.name)}
            aria-pressed={active}
            className={cx('w-full rounded-xl p-3 text-left ring-1 transition', active ? `bg-ink/[0.05] ring-2 ${tone.ring}` : 'ring-line/10 hover:bg-ink/[0.03]')}
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="flex items-center gap-2 text-sm font-semibold">
                <span className={cx('h-2.5 w-2.5 rounded-full', tone.bg)} /> {d.name}
              </span>
              <span className="flex items-center gap-2">
                {d.positive ? <Badge tone="bad">Positive</Badge> : <Badge tone="ok">Negative</Badge>}
                <span className="w-14 text-right font-display font-semibold tabular-nums">{pct(d.probability)}</span>
              </span>
            </div>
            <div className="relative h-2.5 rounded-full bg-ink/[0.07]">
              <motion.div
                className={cx('h-full rounded-full', tone.bg)}
                initial={{ width: 0 }}
                animate={{ width: `${d.probability * 100}%` }}
                transition={{ duration: 0.9, delay: i * 0.12, ease: [0.16, 1, 0.3, 1] }}
              />
              <div className="absolute -top-1 w-0.5 rounded bg-ink/70" style={{ left: `${d.threshold * 100}%`, height: 18 }} title={`Decision threshold ${d.threshold.toFixed(3)}`} />
            </div>
            <div className="mt-1.5 flex justify-between text-[11px] text-muted">
              <span>threshold {d.threshold.toFixed(3)}</span>
              <span>
                test AUC {d.auc.toFixed(3)} · F1 {d.f1.toFixed(3)}
              </span>
            </div>
          </button>
        )
      })}
    </div>
  )
}

// ── heatmap viewer ───────────────────────────────────────────────────────────
// Defined at module level (not inside HeatmapViewer) so parent re-renders, e.g. a ticking timer, never remount it.
function HeatmapLayer({ result, name, opacity, small }) {
  const dd = result.diseases.find((x) => x.name === name)
  return (
    <div className="relative aspect-square overflow-hidden rounded-xl bg-black ring-1 ring-line/15">
      <img src={result.base_image} alt={`X-ray under ${name} heatmap`} className="absolute inset-0 h-full w-full object-contain" />
      {/* key = disease: the fade-in plays only when the disease changes; the slider then just updates opacity */}
      <img
        key={name}
        src={dd.heatmap}
        alt={`Score-CAM heatmap for ${name}`}
        className="heat-in absolute inset-0 h-full w-full object-contain transition-opacity duration-150"
        style={{ opacity }}
      />
      <span className={cx('absolute left-2 top-2 rounded-md bg-black/65 px-2 py-0.5 font-medium text-white', small ? 'text-[10px]' : 'text-xs')}>
        {name} · {pct(dd.probability)}
      </span>
    </div>
  )
}

export function HeatmapViewer({ result, selected, onSelect }) {
  const [opacity, setOpacity] = useState(0.5)
  const [view, setView] = useState('overlay')
  const d = result.diseases.find((x) => x.name === selected) || result.diseases[0]
  const tone = DISEASE_TONE[d.name]

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Disease heatmap">
          {result.diseases.map((x) => {
            const t = DISEASE_TONE[x.name]
            const active = x.name === d.name
            return (
              <button
                key={x.name}
                type="button"
                onClick={() => onSelect(x.name)}
                aria-pressed={active}
                className={cx('chip px-3 py-1.5 ring-1 transition', active ? `bg-ink/[0.08] ${t.text} ring-2 ${t.ring}` : 'text-muted ring-line/20 hover:text-ink')}
              >
                <span className={cx('h-2 w-2 rounded-full', t.bg)} /> {x.name}
              </button>
            )
          })}
        </div>
        <Tabs
          size="sm"
          value={view}
          onChange={setView}
          tabs={[
            { value: 'overlay', label: 'Overlay', icon: Layers },
            { value: 'side', label: 'Side by side', icon: Columns2 },
            { value: 'grid', label: 'All diseases' },
          ]}
        />
      </div>

      {view === 'overlay' && <HeatmapLayer result={result} name={d.name} opacity={opacity} />}
      {view === 'side' && (
        <div className="grid grid-cols-2 gap-3">
          <div className="relative aspect-square overflow-hidden rounded-xl bg-black ring-1 ring-line/15">
            <img src={result.base_image} alt="Input X-ray" className="h-full w-full object-contain" />
            <span className="absolute left-2 top-2 rounded-md bg-black/65 px-2 py-0.5 text-xs text-white">Input</span>
          </div>
          <HeatmapLayer result={result} name={d.name} opacity={opacity} />
        </div>
      )}
      {view === 'grid' && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {result.diseases.map((x) => (
            <button key={x.name} type="button" onClick={() => (onSelect(x.name), setView('overlay'))} className="text-left" aria-label={`Open ${x.name} heatmap`}>
              <HeatmapLayer result={result} name={x.name} opacity={opacity} small />
            </button>
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-4 rounded-xl bg-ink/[0.03] px-4 py-3 ring-1 ring-line/10">
        <label htmlFor="opacity" className="text-sm font-medium">
          Heatmap opacity
        </label>
        <input id="opacity" type="range" min={0} max={1} step={0.01} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="min-w-[140px] flex-1" />
        <span className="w-10 text-right text-sm tabular-nums text-muted">{Math.round(opacity * 100)}%</span>
        <div className="flex w-full items-center gap-2 text-[11px] text-muted sm:w-auto">
          low
          <span className="h-2 w-28 rounded-full" style={{ background: 'linear-gradient(90deg,#00008f,#0000ff,#00ffff,#ffff00,#ff0000,#800000)' }} />
          high activation
        </div>
      </div>
      <p className={cx('text-xs', tone.text)}>
        {result.mode === 'mock' ? (
          <>Mock heatmap for <b>{d.name}</b> (anatomical prior). Run with real weights to get Score-CAM.</>
        ) : (
          <>
            Score-CAM for <b>{d.name}</b> uses this disease&apos;s own class score, so this map is independent of the other diseases.
          </>
        )}
      </p>
    </div>
  )
}

// ── report ───────────────────────────────────────────────────────────────────
const SENT_STYLE = {
  verified: 'decoration-ok/70',
  unsupported: 'bg-warn/15 decoration-warn',
  contradicted: 'bg-bad/15 decoration-bad',
  unchecked: 'decoration-transparent',
}

export function ReportView({ result, onPdf }) {
  const [highlight, setHighlight] = useState(true)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(result.report.text)
      toast.success('Report copied to clipboard')
    } catch {
      toast.error('Clipboard is not available in this browser')
    }
  }
  const part = (section) => {
    const sents = result.sentences.filter((s) => s.section === section)
    if (!sents.length) return <span className="text-muted">—</span>
    return sents.map((s, i) => (
      <span
        key={i}
        title={s.claims.length ? `${s.status}: ${s.claims.join(', ')}` : 'No extracted claim'}
        className={cx('rounded px-0.5 underline decoration-2 underline-offset-4 transition', highlight ? SENT_STYLE[s.status] : 'decoration-transparent')}
      >
        {s.text}{' '}
      </span>
    ))
  }
  return (
    <div className="overflow-hidden rounded-2xl bg-elev ring-1 ring-line/15">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b hairline px-5 py-3">
        <div>
          <p className="font-display font-semibold">Radiology Report</p>
          <p className="text-xs text-muted">Generated by {result.model.name}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className="chip cursor-pointer bg-ink/5 text-muted ring-1 ring-line/15">
            <input type="checkbox" checked={highlight} onChange={(e) => setHighlight(e.target.checked)} className="accent-[rgb(var(--primary))]" />
            Highlight claims
          </label>
          <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={copy}>
            <Copy size={14} /> Copy
          </button>
          {onPdf && (
            <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={onPdf}>
              <FileDown size={14} /> PDF
            </button>
          )}
        </div>
      </div>
      <div className="space-y-5 px-5 py-5 text-[15px] leading-8">
        <div>
          <p className="label mb-1 text-primary">Findings</p>
          <p>{part('findings')}</p>
        </div>
        <div>
          <p className="label mb-1 text-primary">Impression</p>
          <p className="font-medium">{part('impression')}</p>
        </div>
      </div>
      {highlight && (
        <div className="flex flex-wrap gap-4 border-t hairline px-5 py-3 text-[11px] text-muted">
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-ok" /> verified claim</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-4 rounded-sm bg-warn/40" /> unsupported / hallucination risk</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-4 rounded-sm bg-bad/40" /> contradicted</span>
          <span>no underline: no claim extracted</span>
        </div>
      )}
    </div>
  )
}

// ── evaluation ───────────────────────────────────────────────────────────────
export function EvaluationPanel({ evaluation, consistency }) {
  const [tab, setTab] = useState('claims')
  if (!evaluation?.available)
    return (
      <Notice tone="warn">
        <b>Verification unavailable.</b> {evaluation?.reason || 'RadCheck did not run.'} The report is shown unverified. Set{' '}
        <code className="font-mono">GOOGLE_API_KEY</code> in the project&apos;s <code className="font-mono">.env</code> to enable RadCheck RAG.
      </Notice>
    )
  const c = evaluation.counts
  const halluc = evaluation.hallucination_detected
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-[auto_1fr]">
        <div className="flex flex-col items-center justify-center rounded-2xl bg-ink/[0.03] px-6 py-4 ring-1 ring-line/10">
          <Gauge value={evaluation.overall_confidence} label="Overall confidence" tone={evaluation.overall_confidence >= 0.8 ? 'ok' : evaluation.overall_confidence >= 0.5 ? 'warn' : 'bad'} />
        </div>
        <div className="space-y-3">
          <div className={cx('flex items-center gap-3 rounded-2xl px-4 py-3 ring-1', halluc ? 'bg-bad/10 ring-bad/30' : 'bg-ok/10 ring-ok/30')}>
            {halluc ? <ShieldAlert className="text-bad" size={28} /> : <ShieldCheck className="text-ok" size={28} />}
            <div>
              <p className={cx('font-display text-lg font-semibold', halluc ? 'text-bad' : 'text-ok')}>{halluc ? 'Hallucination detected' : 'Not hallucinated'}</p>
              <p className="text-xs text-muted">
                {evaluation.verifier} verdict · {evaluation.fully_verified ? 'all claims verified' : 'some claims need review'}
              </p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
            {[
              ['Claims', c.claims],
              ['Verified', c.verified, 'ok'],
              ['Flagged', c.flagged, c.flagged ? 'bad' : null],
              ['Typical', c.typical],
              ['Atypical', c.atypical, c.atypical ? 'warn' : null],
              ['PubMed', c.pubmed_grounded],
            ].map(([k, v, t]) => (
              <div key={k} className="rounded-xl bg-ink/[0.03] px-2 py-2 text-center ring-1 ring-line/10">
                <p className={cx('font-display text-xl font-bold tabular-nums', t && `text-${t}`)}>{v}</p>
                <p className="text-[10px] uppercase tracking-wider text-muted">{k}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <Tabs
        size="sm"
        value={tab}
        onChange={setTab}
        tabs={[
          { value: 'claims', label: `Claims (${c.claims})` },
          ...(consistency ? [{ value: 'consistency', label: 'Detector consistency' }] : []),
          { value: 'contradictions', label: `Contradictions (${evaluation.contradictions.length})` },
        ]}
      />

      {tab === 'claims' && (
        <div className="overflow-x-auto rounded-xl ring-1 ring-line/15">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="bg-ink/[0.04] text-[11px] uppercase tracking-wider text-muted">
              <tr>
                <th className="px-3 py-2 font-semibold">Claim</th>
                <th className="px-3 py-2 font-semibold">Target</th>
                <th className="px-3 py-2 font-semibold">Status</th>
                <th className="px-3 py-2 font-semibold">LLM baseline</th>
                <th className="px-3 py-2 font-semibold">PubMed</th>
                <th className="px-3 py-2 font-semibold">Grounding</th>
                <th className="px-3 py-2 font-semibold">Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/10">
              {evaluation.claims.map((cl, i) => {
                const bad = cl.hallucination_risk || cl.is_contradicted
                return (
                  <tr key={i} className="align-top hover:bg-ink/[0.02]" title={cl.baseline_reasoning}>
                    <td className="px-3 py-2 font-medium">{cl.finding}</td>
                    <td className="px-3 py-2 text-muted">{cl.target}</td>
                    <td className="px-3 py-2 text-muted">{cl.status}</td>
                    <td className="px-3 py-2">
                      <Badge tone={cl.baseline_classification === 'typical' ? 'muted' : 'warn'}>{cl.baseline_classification}</Badge>
                    </td>
                    <td className="px-3 py-2 text-muted">{cl.pubmed_grounded ? 'grounded' : cl.baseline_classification === 'atypical' ? 'not found' : 'skipped'}</td>
                    <td className="px-3 py-2 tabular-nums">{fmt(cl.grounding_score, 2)}</td>
                    <td className="px-3 py-2">
                      {bad ? (
                        <span className="inline-flex items-center gap-1 text-bad"><XCircle size={14} /> {cl.is_contradicted ? 'contradicted' : 'risk'}</span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-ok"><CheckCircle2 size={14} /> verified</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'consistency' &&
        (consistency ? (
          <div className="space-y-3">
            <p className="text-sm text-muted">
              Report statements compared with <b className="text-ink">{consistency.detector.source}</b> predictions
              {consistency.agreement_rate != null && (
                <>
                  {' '}— agreement on mentioned diseases: <b className="text-ink">{pct(consistency.agreement_rate, 0)}</b>
                </>
              )}
              .
            </p>
            <div className="grid gap-2 sm:grid-cols-3">
              {DISEASES.map((d) => {
                const v = consistency.per_disease[d]
                if (!v) return null
                const tone = v.agreement === 'agree' ? 'ok' : v.agreement === 'disagree' ? 'bad' : 'muted'
                return (
                  <div key={d} className="rounded-xl bg-ink/[0.03] p-3 ring-1 ring-line/10">
                    <div className="mb-2 flex items-center justify-between">
                      <span className={cx('text-sm font-semibold', DISEASE_TONE[d].text)}>{d}</span>
                      <Badge tone={tone}>{v.agreement}</Badge>
                    </div>
                    <p className="text-xs text-muted">
                      Report: <b className="text-ink">{v.report}</b>
                    </p>
                    <p className="text-xs text-muted">
                      Detector: <b className="text-ink">{v.detector}</b>
                      {consistency.detector.probabilities?.[d] != null && ` (${pct(consistency.detector.probabilities[d])})`}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>
        ) : (
          <EmptyState title="No detector findings">Pick a Stage 2 model to cross-check against, or run the complete pipeline.</EmptyState>
        ))}

      {tab === 'contradictions' &&
        (evaluation.contradictions.length ? (
          <ul className="space-y-2">
            {evaluation.contradictions.map((x, i) => (
              <li key={i} className="rounded-xl bg-bad/10 px-4 py-3 text-sm ring-1 ring-bad/20">
                {typeof x === 'string' ? x : x.description || JSON.stringify(x)}
              </li>
            ))}
          </ul>
        ) : (
          <Notice tone="ok">The MKG contradiction engine found no conflicting claims ({evaluation.graph_stats?.num_entities ?? '–'} entities, {evaluation.graph_stats?.num_relations ?? '–'} relations in the graph).</Notice>
        ))}
    </div>
  )
}

// ── quality metric comparison (Stage 1) ──────────────────────────────────────
const METRIC_INFO = [
  { key: 'brisque', label: 'BRISQUE', better: 'lower', d: 2 },
  { key: 'entropy', label: 'Entropy', better: 'higher', d: 2 },
  { key: 'sharpness', label: 'Sharpness', better: 'higher', d: 0 },
  { key: 'cii', label: 'CII', better: 'higher', d: 2 },
  { key: 'eme', label: 'EME', better: 'higher', d: 1 },
]

export function QualityMetrics({ before, after }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
      {METRIC_INFO.map((m) => {
        const b = before[m.key]
        const a = after[m.key]
        const known = a != null && b != null
        const improved = known && (m.better === 'lower' ? a < b : a > b)
        const delta = known ? a - b : null
        return (
          <div key={m.key} className="rounded-xl bg-ink/[0.03] p-3 ring-1 ring-line/10">
            <p className="label flex items-center justify-between">
              {m.label} <span className="normal-case tracking-normal">{m.better === 'lower' ? '↓' : '↑'}</span>
            </p>
            <p className="mt-1 font-display text-xl font-semibold tabular-nums">{a == null ? 'n/a' : fmt(a, m.d)}</p>
            <p className="text-[11px] text-muted tabular-nums">
              from {b == null ? 'n/a' : fmt(b, m.d)}
              {known && m.key !== 'cii' && (
                <span className={cx('ml-1 font-semibold', improved ? 'text-ok' : 'text-bad')}>
                  {delta > 0 ? '+' : ''}
                  {fmt(delta, m.d)}
                </span>
              )}
            </p>
          </div>
        )
      })}
    </div>
  )
}

export const downloadDataUrl = (url, name) => download(url, name)
export const fileStem = (file) => slug((file?.name || 'xray').replace(/\.[^.]+$/, ''))
