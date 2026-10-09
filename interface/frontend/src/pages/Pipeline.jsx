import { AnimatePresence, motion } from 'framer-motion'
import { Archive, Check, FileText, Loader2, RotateCcw, ScanLine, ShieldAlert, ShieldCheck, Sparkles, Workflow, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { CompareSlider, EvaluationPanel, HeatmapViewer, ModelInfo, ModelPicker, ProbabilityBars, ReportView, ResultHeader, RunButton, Uploader, fileStem } from '../components/demo'
import { Badge, Card, Notice, PageHeader, Skeleton, Stat } from '../components/ui'
import { api } from '../lib/api'
import { useElapsed, useModels } from '../lib/hooks'
import { cx, dataUrlToBlob, downloadBlob, pct, reportPdf, secs, stripImages } from '../lib/utils'
import { DetectionSummary, zipDetection } from './Detect'
import { GateToggle, GateVerdict } from './Enhance'
import { downloadReportPdf } from './Report'

const STEPS = [
  { key: 's1', label: 'Enhancement', sub: 'Quality gate + Stage 1', icon: Sparkles },
  { key: 's2', label: 'Detection', sub: 'Probabilities + Score-CAM', icon: ScanLine },
  { key: 's3', label: 'Report', sub: 'MediVLM + RadCheck', icon: FileText },
]

function Stepper({ status, current }) {
  // the 100 ms timer lives here so only the stepper re-renders while a stage runs, not the result panels below
  const elapsed = useElapsed(current ? Object.values(status).join() : false)
  return (
    <ol className="grid gap-3 sm:grid-cols-3" aria-label="Pipeline progress">
      {STEPS.map((s, i) => {
        const st = status[s.key]
        const active = st === 'running'
        return (
          <li key={s.key} className="relative" aria-current={active ? 'step' : undefined}>
            {i > 0 && (
              <div className="absolute -left-3 top-7 hidden h-0.5 w-3 sm:block" aria-hidden>
                <div className={cx('h-full transition-colors duration-500', status[STEPS[i - 1].key] === 'done' ? 'bg-primary' : 'bg-line/20')} />
              </div>
            )}
            <div
              className={cx(
                'flex items-center gap-3 rounded-2xl p-3 ring-1 transition duration-500',
                active ? 'bg-primary/10 shadow-glow ring-primary' : st === 'done' ? 'bg-ok/[0.06] ring-ok/30' : st === 'error' ? 'bg-bad/10 ring-bad/40' : 'ring-line/15',
              )}
            >
              <span
                className={cx(
                  'relative grid h-11 w-11 shrink-0 place-items-center rounded-xl transition',
                  active ? 'bg-primary text-white' : st === 'done' ? 'bg-ok text-white' : st === 'error' ? 'bg-bad text-white' : 'bg-ink/[0.06] text-muted',
                )}
              >
                {active && <span className="absolute inset-0 animate-ping rounded-xl bg-primary/40" />}
                {active ? <Loader2 size={20} className="relative animate-spin" /> : st === 'done' ? <Check size={20} /> : st === 'error' ? <X size={20} /> : <s.icon size={20} />}
              </span>
              <span className="min-w-0">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-muted">Stage {i + 1}</span>
                <span className="block font-display font-semibold">{s.label}</span>
                <span className="block truncate text-xs text-muted">{active ? `running… ${(elapsed / 1000).toFixed(0)}s` : st === 'skipped' ? 'not run' : s.sub}</span>
              </span>
            </div>
            {active && current && <span className="sr-only" aria-live="polite">{`Running ${s.label}`}</span>}
          </li>
        )
      })}
    </ol>
  )
}

function StepPanel({ n, title, icon: Icon, result, children }) {
  return (
    <motion.section initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}>
      <Card>
        <ResultHeader
          title={
            <span className="flex items-center gap-2">
              <span className="grid h-7 w-7 place-items-center rounded-lg bg-primary/15 text-sm text-primary">{n}</span>
              <Icon size={18} className="text-muted" /> {title}
            </span>
          }
          result={result}
        />
        {result?.notice && <Notice className="mb-4">{result.notice}</Notice>}
        {children}
      </Card>
    </motion.section>
  )
}

export default function Pipeline() {
  const m1 = useModels(1)
  const m2 = useModels(2)
  const [file, setFile] = useState(null)
  const [s1Model, setS1Model] = useState(null)
  const [s2Model, setS2Model] = useState(null)
  const [gateMode, setGateMode] = useState('auto')
  const [status, setStatus] = useState({ s1: 'idle', s2: 'idle', s3: 'idle' })
  const [out, setOut] = useState({})
  const [running, setRunning] = useState(false)
  const [selected, setSelected] = useState('Cardiomegaly')
  const [totalMs, setTotalMs] = useState(null)
  const ctl = useRef(null)
  const summaryRef = useRef(null)

  useEffect(() => {
    if (!s1Model && m1.default) setS1Model(m1.default)
  }, [m1.default, s1Model])
  useEffect(() => {
    if (!s2Model && m2.default) setS2Model(m2.default)
  }, [m2.default, s2Model])
  useEffect(() => () => ctl.current?.abort(), [])

  const reset = () => {
    setOut({})
    setTotalMs(null)
    setStatus({ s1: 'idle', s2: 'idle', s3: 'idle' })
  }

  const run = async () => {
    if (!file) return toast.error('Upload an X-ray first')
    reset()
    ctl.current = new AbortController()
    const signal = ctl.current.signal
    setRunning(true)
    const t0 = performance.now()
    let stage = 's1'
    try {
      setStatus({ s1: 'running', s2: 'idle', s3: 'idle' })
      const s1 = await api.enhance(file, s1Model, gateMode, signal)
      setOut({ s1 })
      setStatus((s) => ({ ...s, s1: 'done', s2: 'running' }))
      toast.success(s1.was_enhanced ? 'Stage 1 · image enhanced' : 'Stage 1 · quality gate passed, enhancement skipped')

      stage = 's2'
      const enhanced = dataUrlToBlob(s1.enhanced, 'enhanced.png')
      const s2 = await api.detect(enhanced, s2Model, signal)
      setOut((o) => ({ ...o, s2 }))
      setSelected(s2.top_disease)
      setStatus((s) => ({ ...s, s2: 'done', s3: 'running' }))
      toast.success('Stage 2 · detection complete', { description: s2.predicted_class })

      stage = 's3'
      const findings = { model: s2.model.name, positive: s2.positive, probabilities: Object.fromEntries(s2.diseases.map((d) => [d.name, d.probability])) }
      const s3 = await api.report(enhanced, { findings }, signal)
      setOut((o) => ({ ...o, s3 }))
      setStatus((s) => ({ ...s, s3: 'done' }))
      setTotalMs(performance.now() - t0)
      toast.success('Pipeline complete', { description: `Total ${secs(Math.round(performance.now() - t0))}` })
      setTimeout(() => summaryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 300)
    } catch (e) {
      if (e.name === 'AbortError') {
        setStatus((s) => Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v === 'running' ? 'idle' : v])))
        toast.info('Pipeline cancelled')
      } else {
        setStatus((s) => ({ ...s, [stage]: 'error' }))
        toast.error(`Stage ${stage[1]} failed`, { description: e.message })
      }
    } finally {
      setRunning(false)
    }
  }

  const downloadAll = async () => {
    const { s1, s2, s3 } = out
    const stem = fileStem(file)
    const JSZip = (await import('jszip')).default
    const z = new JSZip()
    z.file(`input_${file.name}`, file)
    const f1 = z.folder('stage1_enhancement')
    f1.file('original_256.png', dataUrlToBlob(s1.original))
    f1.file(`enhanced_${s1.model.id}.png`, dataUrlToBlob(s1.enhanced))
    f1.file('enhancement.json', JSON.stringify(stripImages(s1), null, 2))
    await zipDetection(s2, stem, z)
    const f3 = z.folder('stage3_report')
    f3.file('report.txt', s3.report.text)
    f3.file('report.json', JSON.stringify(stripImages(s3), null, 2))
    const pdf = await reportPdf({
      report: s3.report,
      evaluation: s3.evaluation,
      consistency: s3.consistency,
      image: s1.enhanced,
      title: 'ReXTrust Pipeline Report',
      extra: [
        ['Stage 1 · Enhancement', `${s1.model.name}: ${s1.was_enhanced ? 'enhanced' : 'skipped by quality gate'} (${s1.gate.reason}).`],
        ['Stage 2 · Detection', `${s2.model.name}: ${s2.diseases.map((d) => `${d.name} ${pct(d.probability)}${d.positive ? ' (positive)' : ''}`).join(', ')}.`],
      ],
    })
    f3.file('report.pdf', pdf)
    z.file('summary.json', JSON.stringify({ input: file.name, modes: { stage1: s1.mode, stage2: s2.mode, stage3: s3.mode }, total_ms: Math.round(totalMs), stage1: stripImages(s1), stage2: stripImages(s2), stage3: stripImages(s3) }, null, 2))
    downloadBlob(await z.generateAsync({ type: 'blob' }), `${stem}_rextrust_results.zip`)
    toast.success('All results downloaded')
  }

  const done = status.s3 === 'done'
  const { s1, s2, s3 } = out
  const anyMock = [s1, s2, s3].some((x) => x?.mode === 'mock')

  return (
    <>
      <PageHeader
        icon={Workflow}
        eyebrow="Complete pipeline"
        title="From raw X-ray to verified report"
        intro="One upload runs Stage 1 → Stage 2 → Stage 3 in sequence. The enhanced image feeds detection, and the detection findings feed the report check."
      />
      <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6 lg:px-8">
        <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
          <Uploader file={file} onFile={(f) => (setFile(f), reset())} disabled={running} scanning={running} title="1 · Upload chest X-ray" />
          <Card className="space-y-5">
            <div className="grid gap-5 md:grid-cols-2">
              <div className="space-y-3">
                <p className="label flex items-center gap-2">
                  <Sparkles size={13} /> 2 · Stage 1 model
                </p>
                <ModelPicker stage={1} state={m1} value={s1Model} onChange={setS1Model} disabled={running} compact />
                <GateToggle value={gateMode} onChange={setGateMode} disabled={running} />
              </div>
              <div className="space-y-3">
                <p className="label flex items-center gap-2">
                  <ScanLine size={13} /> 3 · Stage 2 model
                </p>
                <ModelPicker stage={2} state={m2} value={s2Model} onChange={setS2Model} disabled={running} compact />
                <ModelInfo model={m2.models.find((m) => m.id === s2Model)} />
                <p className="text-xs text-muted">Stage 3 uses MediVLM + RadCheck RAG and cross-checks the report against the Stage 2 findings.</p>
              </div>
            </div>
            <RunButton onRun={run} onCancel={() => ctl.current?.abort()} running={running} disabled={!file || !s1Model || !s2Model} label="Run complete pipeline" hint="Real models on CPU: about 1.5–2.5 minutes end to end" />
          </Card>
        </div>

        <Card>
          <Stepper status={status} current={running} />
          {!running && !s1 && <p className="mt-4 text-center text-sm text-muted">Choose an X-ray and models, then run the pipeline. Each stage&apos;s output appears below as soon as it finishes.</p>}
          {Object.values(status).includes('error') && (
            <div className="mt-4 flex justify-center">
              <button type="button" className="btn-ghost" onClick={run}>
                <RotateCcw size={16} /> Retry pipeline
              </button>
            </div>
          )}
        </Card>

        <AnimatePresence>
          {done && (
            <motion.section ref={summaryRef} initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="scroll-mt-24" aria-label="Summary dashboard">
              <Card className="relative overflow-hidden">
                <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-primary/15 blur-3xl" aria-hidden />
                <div className="relative">
                  <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="label text-primary">Summary dashboard</p>
                      <h2 className="font-display text-2xl font-bold">Diagnostic summary · {file?.name}</h2>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {anyMock && <Badge tone="warn">contains mock output</Badge>}
                      <button type="button" className="btn-ghost px-3 py-2 text-sm" onClick={() => downloadReportPdf(s3, fileStem(file))}>
                        <FileText size={16} /> Report PDF
                      </button>
                      <button type="button" className="btn-primary px-4 py-2 text-sm" onClick={downloadAll}>
                        <Archive size={16} /> Download all (ZIP)
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
                    <Stat label="Quality gate" value={s1.was_enhanced ? 'Enhanced' : 'Skipped'} sub={s1.gate.needs_enhancement ? 'low quality input' : 'input already clear'} tone={s1.was_enhanced ? 'primary' : 'ok'} />
                    <Stat label="Entropy" value={`${s1.metrics.before.entropy.toFixed(2)} → ${s1.metrics.after.entropy.toFixed(2)}`} sub={s1.model.name} />
                    <Stat label="Prediction" value={s2.positive.length ? `${s2.positive.length} positive` : 'No finding'} sub={s2.positive.join(', ') || 'of the 3 target diseases'} tone={s2.positive.length ? 'bad' : 'ok'} />
                    <Stat label="Top probability" value={pct(s2.top_probability)} sub={s2.top_disease} />
                    <Stat
                      label="Report verdict"
                      value={s3.evaluation.available ? (s3.evaluation.hallucination_detected ? 'Hallucinated' : 'Verified') : 'Unverified'}
                      sub={s3.evaluation.available ? `confidence ${pct(s3.evaluation.overall_confidence, 0)}` : 'RadCheck unavailable'}
                      tone={s3.evaluation.available ? (s3.evaluation.hallucination_detected ? 'bad' : 'ok') : 'warn'}
                    />
                    <Stat label="Total time" value={secs(Math.round(totalMs))} sub={`stages: ${[s1, s2, s3].map((x) => x.mode).join(' · ')}`} />
                  </div>
                  <div className="mt-5 grid gap-4 md:grid-cols-[repeat(4,minmax(0,1fr))_1.6fr]">
                    {[
                      ['Original', s1.original],
                      [s1.was_enhanced ? 'Enhanced' : 'Unchanged', s1.enhanced],
                      ...s2.diseases.slice(0, 2).map((d) => [`${d.name} map`, d.overlay]),
                    ].map(([l, src]) => (
                      <figure key={l}>
                        <img src={src} alt={l} className="aspect-square w-full rounded-xl bg-black object-contain ring-1 ring-line/15" />
                        <figcaption className="mt-1 text-xs text-muted">{l}</figcaption>
                      </figure>
                    ))}
                    <div className="rounded-xl bg-ink/[0.03] p-4 text-sm leading-relaxed ring-1 ring-line/10">
                      <p className="label mb-1 flex items-center gap-1.5">
                        {s3.evaluation.hallucination_detected ? <ShieldAlert size={13} className="text-bad" /> : <ShieldCheck size={13} className="text-ok" />} Impression
                      </p>
                      <p className="font-medium">{s3.report.impression || s3.report.findings}</p>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.section>
          )}
        </AnimatePresence>

        {status.s1 === 'running' && <Skeleton className="h-96 w-full" />}
        {s1 && (
          <StepPanel n={1} title={`Enhancement · ${s1.model.name}`} icon={Sparkles} result={s1}>
            <div className="grid gap-5 lg:grid-cols-[1fr_1fr]">
              <CompareSlider before={s1.original} after={s1.enhanced} afterLabel={s1.was_enhanced ? 'Enhanced' : 'Unchanged'} />
              <div className="space-y-4">
                <GateVerdict gate={s1.gate} wasEnhanced={s1.was_enhanced} />
                <div className="grid grid-cols-2 gap-2">
                  {['brisque', 'entropy', 'sharpness', 'cii'].map((k) => (
                    <Stat
                      key={k}
                      label={k === 'cii' ? 'CII' : k[0].toUpperCase() + k.slice(1)}
                      value={s1.metrics.after[k] == null ? 'n/a' : Number(s1.metrics.after[k]).toFixed(k === 'sharpness' ? 0 : 2)}
                      sub={`before ${s1.metrics.before[k] == null ? 'n/a' : Number(s1.metrics.before[k]).toFixed(k === 'sharpness' ? 0 : 2)}`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </StepPanel>
        )}

        {status.s2 === 'running' && <Skeleton className="h-96 w-full" />}
        {s2 && (
          <StepPanel n={2} title={`Detection · ${s2.model.name}`} icon={ScanLine} result={s2}>
            <DetectionSummary result={s2} />
            <div className="mt-5 grid gap-6 xl:grid-cols-[1fr_1.25fr]">
              <ProbabilityBars diseases={s2.diseases} selected={selected} onSelect={setSelected} />
              <HeatmapViewer result={s2} selected={selected} onSelect={setSelected} />
            </div>
          </StepPanel>
        )}

        {status.s3 === 'running' && <Skeleton className="h-96 w-full" />}
        {s3 && (
          <StepPanel n={3} title="Report + RadCheck verification" icon={FileText} result={s3}>
            <div className="space-y-6">
              <ReportView result={s3} onPdf={() => downloadReportPdf(s3, fileStem(file))} />
              <EvaluationPanel evaluation={s3.evaluation} consistency={s3.consistency} />
            </div>
          </StepPanel>
        )}

        {!s1 && !running && (
          <div className="grid gap-4 sm:grid-cols-3">
            {STEPS.map((s, i) => (
              <div key={s.key} className="flex items-center gap-3 rounded-2xl border border-dashed hairline p-4 text-sm text-muted">
                <s.icon size={18} /> Stage {i + 1} output will appear here
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}
