import { CheckCircle2, Columns2, Download, FileJson, Gauge as GaugeIcon, ImageOff, Sparkles, SplitSquareHorizontal } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { CompareSlider, ModelInfo, ModelPicker, QualityMetrics, ResultHeader, RunButton, Uploader, fileStem } from '../components/demo'
import { Badge, Card, EmptyState, Figure, Notice, PageHeader, Skeleton, Tabs } from '../components/ui'
import { api } from '../lib/api'
import { FIGURES } from '../lib/content'
import { useModels } from '../lib/hooks'
import { cx, download, downloadJson, stripImages } from '../lib/utils'

export function GateToggle({ value, onChange, disabled }) {
  return (
    <div role="radiogroup" aria-label="Enhancement mode" className="grid grid-cols-2 gap-2">
      {[
        { v: 'always', t: 'Always enhance', d: 'Run the chosen model' },
        { v: 'auto', t: 'Quality gate', d: 'Enhance only if needed' },
      ].map((o) => (
        <label key={o.v} className={cx('cursor-pointer rounded-xl p-3 ring-1 transition', value === o.v ? 'bg-primary/[0.08] ring-2 ring-primary' : 'ring-line/15 hover:ring-primary/50', disabled && 'pointer-events-none opacity-60')}>
          <input type="radio" className="sr-only" name="gate" checked={value === o.v} onChange={() => onChange(o.v)} />
          <span className="block text-sm font-semibold">{o.t}</span>
          <span className="block text-xs text-muted">{o.d}</span>
        </label>
      ))}
    </div>
  )
}

export function GateVerdict({ gate, wasEnhanced }) {
  return (
    <div className={cx('flex flex-wrap items-center gap-3 rounded-xl px-4 py-3 text-sm ring-1', gate.needs_enhancement ? 'bg-warn/10 ring-warn/25' : 'bg-ok/10 ring-ok/25')}>
      <GaugeIcon size={18} className={gate.needs_enhancement ? 'text-warn' : 'text-ok'} />
      <span className="font-semibold">Quality gate: {gate.needs_enhancement ? 'low quality — enhancement needed' : 'passes — image is already clear'}</span>
      <span className="text-muted">({gate.reason})</span>
      <span className="ml-auto">
        {wasEnhanced ? <Badge tone="primary">Enhanced</Badge> : <Badge tone="ok" icon={CheckCircle2}>Skipped (unchanged)</Badge>}
      </span>
    </div>
  )
}

export default function Enhance() {
  const models = useModels(1)
  const [file, setFile] = useState(null)
  const [model, setModel] = useState(null)
  const [gateMode, setGateMode] = useState('always')
  const [result, setResult] = useState(null)
  const [running, setRunning] = useState(false)
  const [view, setView] = useState('slider')
  const ctl = useRef(null)

  useEffect(() => {
    if (!model && models.default) setModel(models.default)
  }, [models.default, model])
  useEffect(() => () => ctl.current?.abort(), [])

  const spec = models.models.find((m) => m.id === model)

  const run = async () => {
    if (!file) return toast.error('Upload an X-ray first')
    ctl.current = new AbortController()
    setRunning(true)
    setResult(null)
    try {
      const r = await api.enhance(file, model, gateMode, ctl.current.signal)
      setResult(r)
      if (r.notice) toast.warning('Fell back to mock output', { description: r.notice })
      else toast.success(r.was_enhanced ? 'Enhancement complete' : 'Image passed the quality gate', { description: `${r.model.name} · ${(r.timing_ms / 1000).toFixed(1)} s` })
    } catch (e) {
      if (e.name !== 'AbortError') toast.error('Enhancement failed', { description: e.message })
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <PageHeader
        icon={Sparkles}
        eyebrow="Stage 1 · Image enhancement"
        title="Conditional Image Enhancement"
        intro="Upload a chest X-ray, pick an enhancement model and compare the result. The quality gate (BRISQUE > 5.0 or entropy < 7.95) decides whether an image actually needs enhancing."
      />
      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-10 sm:px-6 lg:grid-cols-[380px_1fr] lg:px-8">
        <div className="space-y-4">
          <Uploader file={file} onFile={(f) => (setFile(f), setResult(null))} disabled={running} scanning={running} />
          <Card className="space-y-4 p-4 sm:p-5">
            <div>
              <p className="label mb-3">Enhancement model</p>
              <ModelPicker stage={1} state={models} value={model} onChange={setModel} disabled={running} compact />
            </div>
            <ModelInfo model={spec} />
            <div>
              <p className="label mb-2">Mode</p>
              <GateToggle value={gateMode} onChange={setGateMode} disabled={running} />
            </div>
            <RunButton onRun={run} onCancel={() => ctl.current?.abort()} running={running} disabled={!file || !model} label="Run enhancement" hint="Real models: ~1–5 s on CPU" />
          </Card>
        </div>

        <div className="min-w-0 space-y-6">
          <Card>
            {running ? (
              <div className="space-y-4">
                <Skeleton className="h-8 w-60" />
                <Skeleton className="aspect-square w-full max-w-xl" />
                <div className="grid grid-cols-5 gap-2">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <Skeleton key={i} className="h-20" />
                  ))}
                </div>
              </div>
            ) : !result ? (
              <EmptyState icon={ImageOff} title="No enhanced image yet">
                Upload an X-ray (or pick a sample), choose a model and press <b>Run enhancement</b>.
              </EmptyState>
            ) : (
              <div className="space-y-5">
                <ResultHeader title={result.model.name} result={result}>
                  <Tabs
                    size="sm"
                    value={view}
                    onChange={setView}
                    tabs={[
                      { value: 'slider', label: 'Slider', icon: SplitSquareHorizontal },
                      { value: 'side', label: 'Side by side', icon: Columns2 },
                    ]}
                  />
                  <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={() => download(result.enhanced, `${fileStem(file)}_${result.model.id}.png`)}>
                    <Download size={14} /> Enhanced PNG
                  </button>
                  <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={() => downloadJson(stripImages(result), `${fileStem(file)}_stage1.json`)}>
                    <FileJson size={14} /> JSON
                  </button>
                </ResultHeader>
                {result.notice && <Notice>{result.notice}</Notice>}
                <GateVerdict gate={result.gate} wasEnhanced={result.was_enhanced} />
                {view === 'slider' ? (
                  <div className="mx-auto max-w-xl">
                    <CompareSlider before={result.original} after={result.enhanced} afterLabel={result.was_enhanced ? 'Enhanced' : 'Unchanged'} />
                  </div>
                ) : (
                  <div className="grid gap-3 sm:grid-cols-2">
                    {[
                      ['Original (256×256)', result.original],
                      [result.was_enhanced ? `Enhanced · ${result.model.name}` : 'Unchanged', result.enhanced],
                    ].map(([l, src]) => (
                      <figure key={l}>
                        <img src={src} alt={l} className="aspect-square w-full rounded-xl bg-black object-contain ring-1 ring-line/15" />
                        <figcaption className="mt-1.5 text-xs text-muted">{l}</figcaption>
                      </figure>
                    ))}
                  </div>
                )}
                <div>
                  <p className="label mb-2">Image quality on this X-ray (before → after)</p>
                  <QualityMetrics before={result.metrics.before} after={result.metrics.after} />
                  {result.metrics.after.brisque == null && <p className="mt-2 text-xs text-muted">BRISQUE needs the real engine (piq + torch); it is n/a in mock mode.</p>}
                </div>
              </div>
            )}
          </Card>

          {spec?.benchmark && (
            <Card>
              <p className="label mb-3">Benchmark for {spec.name} · 23 low-quality CheXpert images (TEAM_95_PR2)</p>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                {[
                  ['BRISQUE ↓', spec.benchmark.brisque],
                  ['Entropy ↑', spec.benchmark.entropy],
                  ['Sharpness ↑', spec.benchmark.sharpness],
                  ['CII ↑', spec.benchmark.cii],
                  ['Avg rank ↓', spec.benchmark.avg_rank],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-xl bg-ink/[0.03] p-3 ring-1 ring-line/10">
                    <p className="label">{k}</p>
                    <p className="mt-1 font-display text-xl font-semibold tabular-nums">{v}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          <div className="grid gap-6 md:grid-cols-2">
            <Figure {...FIGURES.attentionUnet} />
            <Figure {...FIGURES.featureMaps} />
          </div>
        </div>
      </div>
    </>
  )
}
