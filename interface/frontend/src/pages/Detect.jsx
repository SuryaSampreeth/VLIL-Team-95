import { Archive, FileJson, ImageOff, ScanLine } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { DetectionChart } from '../components/charts'
import { HeatmapViewer, ModelInfo, ModelPicker, ProbabilityBars, ResultHeader, RunButton, Uploader, fileStem } from '../components/demo'
import { Badge, Card, EmptyState, Figure, Notice, PageHeader, Skeleton } from '../components/ui'
import { api } from '../lib/api'
import { FIGURES } from '../lib/content'
import { useModels } from '../lib/hooks'
import { dataUrlToBlob, downloadBlob, downloadJson, slug, stripImages } from '../lib/utils'

export async function zipDetection(result, stem, zip) {
  const JSZip = zip ? null : (await import('jszip')).default
  const z = zip || new JSZip()
  const folder = zip ? z.folder('stage2_detection') : z
  folder.file('input.jpg', dataUrlToBlob(result.base_image))
  result.diseases.forEach((d) => {
    if (d.heatmap) folder.file(`heatmap_${slug(d.name)}.jpg`, dataUrlToBlob(d.heatmap))
    if (d.overlay) folder.file(`overlay_${slug(d.name)}.jpg`, dataUrlToBlob(d.overlay))
  })
  folder.file('detection.json', JSON.stringify(stripImages(result), null, 2))
  if (!zip) downloadBlob(await z.generateAsync({ type: 'blob' }), `${stem}_stage2_${result.model.id}.zip`)
}

export function DetectionSummary({ result }) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl bg-ink/[0.03] px-4 py-3 text-sm ring-1 ring-line/10">
      <span className="text-muted">Prediction:</span>
      <span className="font-semibold">{result.predicted_class}</span>
      {result.positive.map((p) => (
        <Badge key={p} tone="bad">
          {p}
        </Badge>
      ))}
      <span className="ml-auto text-xs text-muted">{result.explainer}</span>
    </div>
  )
}

export default function Detect() {
  const models = useModels(2)
  const [file, setFile] = useState(null)
  const [model, setModel] = useState(null)
  const [result, setResult] = useState(null)
  const [selected, setSelected] = useState('Cardiomegaly')
  const [running, setRunning] = useState(false)
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
      const r = await api.detect(file, model, ctl.current.signal)
      setResult(r)
      setSelected(r.top_disease)
      if (r.notice) toast.warning('Fell back to mock output', { description: r.notice })
      else toast.success('Detection complete', { description: `${r.predicted_class} · ${(r.timing_ms / 1000).toFixed(1)} s` })
    } catch (e) {
      if (e.name !== 'AbortError') toast.error('Detection failed', { description: e.message })
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <PageHeader
        icon={ScanLine}
        eyebrow="Stage 2 · Disease detection + Score-CAM"
        title="Multi-Disease Detection with Per-Disease Heatmaps"
        intro="Upload any X-ray — Stage 1 is not required. Choose one of the four architectures; you get a probability for Cardiomegaly, Edema and Pleural Effusion, plus an independent Score-CAM heatmap for each disease."
      />
      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-10 sm:px-6 lg:grid-cols-[380px_1fr] lg:px-8">
        <div className="space-y-4">
          <Uploader file={file} onFile={(f) => (setFile(f), setResult(null))} disabled={running} scanning={running} />
          <Card className="space-y-4 p-4 sm:p-5">
            <div>
              <p className="label mb-3">Detection model</p>
              <ModelPicker stage={2} state={models} value={model} onChange={setModel} disabled={running} compact />
            </div>
            <ModelInfo model={spec} />
            <RunButton onRun={run} onCancel={() => ctl.current?.abort()} running={running} disabled={!file || !model} label="Detect diseases" hint="Real models: ~30–45 s on CPU (Score-CAM runs 256 masked forward passes)" />
          </Card>
        </div>

        <div className="min-w-0 space-y-6">
          <Card>
            {running ? (
              <div className="grid gap-6 xl:grid-cols-[1fr_1.2fr]">
                <div className="space-y-3">
                  {[0, 1, 2].map((i) => (
                    <Skeleton key={i} className="h-20" />
                  ))}
                </div>
                <Skeleton className="aspect-square w-full" />
              </div>
            ) : !result ? (
              <EmptyState icon={ImageOff} title="No detection results yet">
                Upload an X-ray, select a model and press <b>Detect diseases</b>. Click a disease to see its own heatmap.
              </EmptyState>
            ) : (
              <div className="space-y-5">
                <ResultHeader title={result.model.name} result={result}>
                  <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={() => zipDetection(result, fileStem(file))}>
                    <Archive size={14} /> Images (ZIP)
                  </button>
                  <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={() => downloadJson(stripImages(result), `${fileStem(file)}_stage2.json`)}>
                    <FileJson size={14} /> JSON
                  </button>
                </ResultHeader>
                {result.notice && <Notice>{result.notice}</Notice>}
                <DetectionSummary result={result} />
                <div className="grid gap-6 xl:grid-cols-[1fr_1.25fr]">
                  <div>
                    <p className="label mb-3">Probabilities (click to view heatmap)</p>
                    <ProbabilityBars diseases={result.diseases} selected={selected} onSelect={setSelected} />
                    <p className="mt-3 text-xs text-muted">The vertical tick marks the per-disease decision threshold, tuned on the validation split. Test AUC / F1 are from the 6k-per-disease experiment.</p>
                  </div>
                  <HeatmapViewer result={result} selected={selected} onSelect={setSelected} />
                </div>
              </div>
            )}
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <p className="label mb-3">How the four models compare (test set)</p>
              <DetectionChart />
            </Card>
            <div className="space-y-4">
              <Figure {...FIGURES.scorecam} />
              <Figure {...FIGURES.aucScales} />
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
