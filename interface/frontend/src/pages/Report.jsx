import { FileJson, FileText, ImageOff, ShieldCheck } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { PairChart } from '../components/charts'
import { EvaluationPanel, ReportView, ResultHeader, RunButton, Uploader, fileStem } from '../components/demo'
import { Card, EmptyState, Notice, PageHeader, Skeleton } from '../components/ui'
import { api } from '../lib/api'
import { RADCHECK_COMPARISON, REPORT_METRICS } from '../lib/content'
import { useBackend } from '../lib/hooks'
import { downloadBlob, downloadJson, reportPdf, stripImages } from '../lib/utils'

export async function downloadReportPdf(result, stem, extra) {
  try {
    const blob = await reportPdf({ report: result.report, evaluation: result.evaluation, consistency: result.consistency, image: result.input_image, extra })
    downloadBlob(blob, `${stem}_report.pdf`)
    toast.success('PDF report downloaded')
  } catch (e) {
    toast.error('Could not build the PDF', { description: e.message })
  }
}

export default function Report() {
  const { health } = useBackend()
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [running, setRunning] = useState(false)
  const ctl = useRef(null)

  useEffect(() => () => ctl.current?.abort(), [])

  const run = async () => {
    if (!file) return toast.error('Upload an X-ray first')
    ctl.current = new AbortController()
    setRunning(true)
    setResult(null)
    try {
      const r = await api.report(file, {}, ctl.current.signal)
      setResult(r)
      if (r.notice) toast.warning('Fell back to mock output', { description: r.notice })
      else if (!r.evaluation.available) toast.warning('Report generated without verification', { description: r.evaluation.reason })
      else toast.success('Report generated and verified', { description: `${(r.timing_ms / 1000).toFixed(1)} s` })
    } catch (e) {
      if (e.name !== 'AbortError') toast.error('Report generation failed', { description: e.message })
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <PageHeader
        icon={FileText}
        eyebrow="Stage 3 · Report generation + evaluation"
        title="Grounded Radiology Reporting"
        intro="MediVLM writes a Findings / Impression report from the X-ray. RadCheck RAG then extracts every clinical claim, checks it against an LLM baseline, grounds atypical claims in PubMed and runs a knowledge-graph contradiction engine."
      />
      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-10 sm:px-6 lg:grid-cols-[380px_1fr] lg:px-8">
        <div className="space-y-4">
          <Uploader file={file} onFile={(f) => (setFile(f), setResult(null))} disabled={running} scanning={running} />
          <Card className="space-y-4 p-4 sm:p-5">
            <div className="rounded-xl bg-ink/[0.03] p-3 text-xs leading-relaxed text-muted ring-1 ring-line/10">
              <p className="mb-1 font-semibold text-ink">What happens when you press Generate</p>
              MediVLM writes the Findings / Impression report, then RadCheck RAG extracts and verifies every clinical claim.
            </div>
            {health && !health.radcheck_key_configured && health.mode !== 'mock' && (
              <Notice tone="warn">GOOGLE_API_KEY is not configured, so RadCheck will be skipped.</Notice>
            )}
            <RunButton onRun={run} onCancel={() => ctl.current?.abort()} running={running} disabled={!file} label="Generate report" hint="Real models: ~60–90 s (MediVLM loads ~3 GB once, RadCheck calls the LLM)" />
          </Card>
        </div>

        <div className="min-w-0 space-y-6">
          <Card>
            {running ? (
              <div className="space-y-4">
                <Skeleton className="h-8 w-56" />
                <Skeleton className="h-44 w-full" />
                <div className="grid gap-4 sm:grid-cols-[200px_1fr]">
                  <Skeleton className="h-40" />
                  <Skeleton className="h-40" />
                </div>
              </div>
            ) : !result ? (
              <EmptyState icon={ImageOff} title="No report yet">
                Upload an X-ray and press <b>Generate report</b>. Unsupported or contradicted statements will be highlighted.
              </EmptyState>
            ) : (
              <div className="space-y-6">
                <ResultHeader title="Generated report" result={result}>
                  <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={() => downloadJson(stripImages(result), `${fileStem(file)}_stage3.json`)}>
                    <FileJson size={14} /> JSON
                  </button>
                </ResultHeader>
                {result.notice && <Notice>{result.notice}</Notice>}
                <div className="grid gap-5 xl:grid-cols-[200px_1fr]">
                  <figure className="hidden xl:block">
                    <img src={result.input_image} alt="Input X-ray" className="aspect-square w-full rounded-xl bg-black object-contain ring-1 ring-line/15" />
                    <figcaption className="mt-1.5 text-xs text-muted">Model input (256×256)</figcaption>
                  </figure>
                  <ReportView result={result} onPdf={() => downloadReportPdf(result, fileStem(file))} />
                </div>
                <div>
                  <p className="label mb-3 flex items-center gap-2">
                    <ShieldCheck size={14} /> Evaluation · RadCheck RAG
                  </p>
                  <EvaluationPanel evaluation={result.evaluation} consistency={result.consistency} />
                </div>
              </div>
            )}
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <p className="mb-1 font-display font-semibold">MediVLM benchmark (MIMIC-CXR)</p>
              <p className="mb-4 text-xs text-muted">GPT-2 is the deployed decoder.</p>
              <PairChart data={REPORT_METRICS} a={{ key: 'gpt2', name: 'GPT-2' }} b={{ key: 'qwen', name: 'Qwen' }} domain={[0, 0.4]} label="MediVLM NLG metrics, GPT-2 vs Qwen" />
            </Card>
            <Card>
              <p className="mb-1 font-display font-semibold">RadCheck: proposed vs previous</p>
              <p className="mb-4 text-xs text-muted">Manually labelled clinical statements.</p>
              <PairChart data={RADCHECK_COMPARISON} a={{ key: 'proposed', name: 'Proposed (A)' }} b={{ key: 'previous', name: 'Previous (B)' }} unit="%" digits={2} domain={[0, 100]} label="RadCheck proposed vs previous architecture" />
            </Card>
          </div>
        </div>
      </div>
    </>
  )
}
