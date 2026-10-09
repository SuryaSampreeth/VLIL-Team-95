import { motion } from 'framer-motion'
import {
  ArrowRight,
  BookOpen,
  Code2,
  Database,
  Download,
  ExternalLink,
  FileText,
  FileWarning,
  HelpCircle,
  Lightbulb,
  Github,
  GraduationCap,
  Layers,
  PlayCircle,
  ScanEye,
  ScanLine,
  Sparkles,
  Target,
  Terminal,
  Workflow,
} from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { DetectionChart, LeaderboardChart, PairChart, RankGrid } from '../components/charts'
import { HowToSteps, SectionNav, useUi } from '../components/extras'
import { HeroBackground, HeroVisual, PipelineDiagram, StageLinkCard, TeamCard, Timeline } from '../components/portal'
import { Accordion, Badge, Card, Counter, Figure, Notice, Reveal, Section, Tabs } from '../components/ui'
import { API_URL } from '../lib/api'
import {
  DATASETS,
  DETECTION_TAKEAWAYS,
  DOCUMENTS,
  ENHANCEMENT_HIGHLIGHTS,
  FIGURES,
  GAPS,
  LINKS,
  NOVELTY,
  OBJECTIVES,
  PROBLEMS,
  PROJECT,
  RADCHECK_COMPARISON,
  REFERENCES,
  REPORT_METRICS,
  TEAM,
  TIMELINE,
  isTodo,
} from '../lib/content'

const PROBLEM_ICONS = { ScanEye, Layers, FileWarning }

function scrollTo(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
}

function Hero() {
  const github = () =>
    isTodo(LINKS.github) ? (scrollTo('github'), toast.info('GitHub URL not set yet', { description: 'Set LINKS.github in src/lib/content.js' })) : window.open(LINKS.github, '_blank', 'noopener')
  return (
    <header id="overview" className="relative scroll-mt-16 overflow-hidden pb-16 pt-28 sm:pb-24 sm:pt-36">
      <HeroBackground />
      <div className="relative mx-auto grid max-w-7xl items-center gap-14 px-4 sm:px-6 lg:grid-cols-[1.15fr_1fr] lg:px-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}>
          <div className="mb-5 flex flex-wrap gap-2">
            <Badge tone="primary">{PROJECT.team}</Badge>
            <Badge>{PROJECT.type}</Badge>
            <Badge tone="accent" icon={GraduationCap}>
              {PROJECT.department}
            </Badge>
          </div>
          <h1 className="text-4xl font-bold leading-[1.05] sm:text-6xl">
            <span className="text-gradient">ReXTrust</span>
            <span className="mt-3 block text-2xl font-semibold leading-tight text-ink sm:text-[2rem]">
              XAI-Driven Multi-Disease Diagnosis in Chest X-rays via Conditional Enhancement and Grounded Reporting
            </span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted">{PROJECT.tagline} Enhance only when needed, show proof for every disease separately, and fact-check the report before trusting it.</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/pipeline" className="btn-primary px-5 py-3 text-base">
              <Sparkles size={18} /> Try Demo
            </Link>
            <button type="button" onClick={github} className="btn-ghost px-5 py-3 text-base">
              <Github size={18} /> GitHub
            </button>
            <button type="button" onClick={() => scrollTo('video')} className="btn-ghost px-5 py-3 text-base">
              <PlayCircle size={18} /> Watch Video
            </button>
          </div>
          <dl className="mt-12 grid max-w-xl grid-cols-2 gap-6 sm:grid-cols-4">
            {[
              { v: 8, l: 'enhancement methods compared' },
              { v: 4, l: 'detection architectures' },
              { v: 3, l: 'diseases, one heatmap each' },
              { v: 86.96, d: 2, s: '%', l: 'RadCheck accuracy' },
            ].map((x) => (
              <div key={x.l}>
                <dt className="sr-only">{x.l}</dt>
                <dd className="font-display text-3xl font-bold">
                  <Counter value={x.v} decimals={x.d || 0} suffix={x.s || ''} />
                </dd>
                <p className="mt-1 text-xs leading-snug text-muted">{x.l}</p>
              </div>
            ))}
          </dl>
        </motion.div>
        <motion.div initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8, delay: 0.2 }}>
          <HeroVisual />
        </motion.div>
      </div>
    </header>
  )
}

function Results() {
  const [tab, setTab] = useState('enh')
  return (
    <div id="results" className="mt-16 scroll-mt-32">
      <Reveal className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="label mb-2 text-primary">Results by stage</p>
          <h3 className="font-display text-2xl font-bold">What the experiments showed</h3>
        </div>
        <Tabs
          value={tab}
          onChange={setTab}
          tabs={[
            { value: 'enh', label: 'Enhancement', icon: Sparkles },
            { value: 'det', label: 'Detection', icon: ScanLine },
            { value: 'rep', label: 'Report', icon: FileText },
          ]}
        />
      </Reveal>

      {tab === 'enh' && (
        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <Card>
            <LeaderboardChart />
          </Card>
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3">
              {ENHANCEMENT_HIGHLIGHTS.map((h) => (
                <Card key={h.label} className="p-4">
                  <p className="font-display text-3xl font-bold text-primary">
                    <Counter value={h.value} decimals={h.decimals} />
                  </p>
                  <p className="mt-1 text-sm font-semibold">{h.label}</p>
                  <p className="mt-1 text-xs text-muted">{h.sub}</p>
                </Card>
              ))}
            </div>
            <Figure {...FIGURES.detail} />
            <Figure {...FIGURES.qualitative} />
          </div>
        </div>
      )}

      {tab === 'det' && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <DetectionChart />
          </Card>
          <Card>
            <RankGrid />
            <ul className="mt-5 space-y-2">
              {DETECTION_TAKEAWAYS.map((t) => (
                <li key={t} className="flex gap-2 text-sm text-muted">
                  <ArrowRight size={14} className="mt-1 shrink-0 text-primary" /> {t}
                </li>
              ))}
            </ul>
          </Card>
          <Figure {...FIGURES.scorecam} className="lg:col-span-2" />
        </div>
      )}

      {tab === 'rep' && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <p className="mb-1 font-display font-semibold">MediVLM report generation</p>
            <p className="mb-4 text-sm text-muted">GPT-2 (original decoder) vs Qwen (experimental). Qwen additionally reports BERTScore = 0.7864.</p>
            <PairChart data={REPORT_METRICS} a={{ key: 'gpt2', name: 'GPT-2' }} b={{ key: 'qwen', name: 'Qwen' }} domain={[0, 0.4]} label="BLEU, METEOR and ROUGE-L for GPT-2 and Qwen decoders" />
          </Card>
          <Card>
            <p className="mb-1 font-display font-semibold">RadCheck RAG hallucination detection</p>
            <p className="mb-4 text-sm text-muted">Proposed architecture vs previous architecture, on manually labelled clinical statements.</p>
            <PairChart
              data={RADCHECK_COMPARISON}
              a={{ key: 'proposed', name: 'Proposed (A)' }}
              b={{ key: 'previous', name: 'Previous (B)' }}
              unit="%"
              digits={2}
              domain={[0, 100]}
              label="Accuracy, precision, recall, F1 and specificity of the proposed and previous RadCheck architectures"
            />
          </Card>
          <Notice tone="muted" className="lg:col-span-2">
            RadCheck was evaluated on a curated set of manually labelled clinical statements, since MIMIC-CXR / IU-Xray reports have no ground-truth contradiction or hallucination labels.
          </Notice>
        </div>
      )}
    </div>
  )
}

const DOC_SECTIONS = [
  {
    title: 'Setup',
    icon: Terminal,
    content: (
      <div className="space-y-3">
        <p>The interface lives in <code className="font-mono text-ink">interface/</code> next to the project&apos;s <code className="font-mono text-ink">pipeline_core.py</code> and <code className="font-mono text-ink">Models/</code>.</p>
        <pre className="overflow-x-auto rounded-xl bg-black/80 p-4 font-mono text-xs leading-relaxed text-slate-200">{`# backend
cd interface/backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# frontend
cd interface/frontend
npm install
npm run dev        # http://localhost:5173`}</pre>
      </div>
    ),
  },
  {
    title: 'Usage',
    icon: BookOpen,
    content: (
      <ol className="list-decimal space-y-1.5 pl-5">
        <li>Open a stage from the navbar, or the Complete Pipeline.</li>
        <li>Drop a chest X-ray (PNG / JPEG / DICOM, ≤ 10 MB) or click a sample.</li>
        <li>Choose the model(s) and press Run. Real models on CPU take ~5 s (enhancement), ~30–45 s (detection + Score-CAM) and ~60–90 s (MediVLM + RadCheck).</li>
        <li>Inspect the outputs, then download images, JSON or the PDF report.</li>
      </ol>
    ),
  },
  {
    title: 'API reference',
    icon: Code2,
    content: (
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-xs">
          <tbody className="divide-y divide-line/10 font-mono">
            {[
              ['GET', '/health', 'mode, device, weight availability'],
              ['GET', '/models/{stage}', 'model registry for stage 1 or 2'],
              ['POST', '/stage1/enhance', 'image, model, gate_mode → enhanced image + quality metrics'],
              ['POST', '/stage2/detect', 'image, model → probabilities + per-disease Score-CAM'],
              ['POST', '/stage3/report', 'image, findings?, crosscheck_model? → report + RadCheck'],
              ['POST', '/pipeline/run', 'image, stage1_model, stage2_model, gate_mode → all outputs'],
            ].map(([m, p, d]) => (
              <tr key={p}>
                <td className="py-2 pr-3 text-primary">{m}</td>
                <td className="py-2 pr-3 text-ink">{p}</td>
                <td className="py-2 font-sans text-muted">{d}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1 text-primary hover:underline">
          Interactive OpenAPI docs <ExternalLink size={12} />
        </a>
      </div>
    ),
  },
  {
    title: 'Adding models & mock mode',
    icon: Database,
    content: (
      <div className="space-y-2">
        <p>
          Every model is an entry in <code className="font-mono text-ink">backend/config.yaml</code> (id, name, architecture, weights path, metrics). Add an entry and it appears in the selectors automatically.
        </p>
        <p>
          <code className="font-mono text-ink">mode: auto</code> uses real weights and falls back to mock output per stage; <code className="font-mono text-ink">mode: mock</code> runs the whole UI without torch; <code className="font-mono text-ink">mode: real</code> never falls back. Mock outputs are always labelled <Badge tone="warn">MOCK OUTPUT</Badge>.
        </p>
      </div>
    ),
  },
]

function Documentation() {
  const [doc, setDoc] = useState(0)
  const d = DOCUMENTS[doc]
  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <Card className="p-3 sm:p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 px-1">
          <Tabs size="sm" value={doc} onChange={setDoc} tabs={DOCUMENTS.map((x, i) => ({ value: i, label: x.title.split(' — ')[0] + (i ? ' (lit.)' : '') }))} />
          <div className="flex gap-2">
            <a href={d.file} target="_blank" rel="noreferrer" className="btn-ghost px-3 py-1.5 text-xs">
              <ExternalLink size={14} /> Open
            </a>
            <a href={d.file} download className="btn-primary px-3 py-1.5 text-xs">
              <Download size={14} /> Download
            </a>
          </div>
        </div>
        <object data={`${d.file}#view=FitH`} type="application/pdf" className="h-[520px] w-full rounded-xl bg-white" aria-label={d.title}>
          <div className="grid h-[520px] place-items-center rounded-xl bg-ink/5 p-6 text-center text-sm text-muted">
            <p>
              Your browser cannot display PDFs inline.{' '}
              <a href={d.file} className="text-primary underline">
                Download {d.title}
              </a>
              .
            </p>
          </div>
        </object>
        <p className="mt-2 px-1 text-xs text-muted">
          {d.title} · {d.pages} slides · {d.desc}
        </p>
      </Card>
      <div className="space-y-4">
        <Accordion items={DOC_SECTIONS} />
        <Card className="p-5">
          <p className="label mb-3">References</p>
          <ol className="list-decimal space-y-1.5 pl-4 text-xs leading-relaxed text-muted">
            {REFERENCES.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ol>
        </Card>
      </div>
    </div>
  )
}

function VideoEmbed() {
  if (isTodo(LINKS.video))
    return (
      <div className="glass relative grid aspect-video place-items-center overflow-hidden">
        <div className="grid-bg absolute inset-0 opacity-60" aria-hidden />
        <div className="relative text-center">
          <motion.div animate={{ scale: [1, 1.08, 1] }} transition={{ duration: 2.4, repeat: Infinity }} className="mx-auto mb-4 grid h-20 w-20 place-items-center rounded-full bg-gradient-to-br from-primary to-accent text-white shadow-glow">
            <PlayCircle size={40} />
          </motion.div>
          <p className="font-display text-lg font-semibold">Demonstration video coming soon</p>
          <p className="mt-1 text-sm text-muted">
            Placeholder: set <code className="font-mono">LINKS.video</code> (TODO_DEMO_VIDEO_URL) in <code className="font-mono">src/lib/content.js</code>
          </p>
          <Link to="/pipeline" className="btn-ghost mt-5">
            Try the live demo instead <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    )
  const isFile = /\.(mp4|webm)$/i.test(LINKS.video)
  return (
    <div className="glass overflow-hidden p-2">
      {isFile ? (
        <video src={LINKS.video} controls className="aspect-video w-full rounded-xl" />
      ) : (
        <iframe src={LINKS.video} title="ReXTrust demonstration video" className="aspect-video w-full rounded-xl" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
      )}
    </div>
  )
}

function HowToUse() {
  const { openHelp } = useUi()
  return (
    <Section id="how-to-use" eyebrow="How to use" title="Get a result in four steps" intro="No setup needed in the browser — pick a stage, drop an X-ray (or use a sample) and press Run." className="bg-elev/40">
      <Reveal>
        <HowToSteps />
      </Reveal>
      <Reveal className="mt-6 flex flex-wrap items-center gap-3">
        <Link to="/pipeline" className="btn-primary">
          <Workflow size={16} /> Start with the complete pipeline
        </Link>
        <Link to="/enhance" className="btn-ghost">
          <Sparkles size={16} /> Stage 1
        </Link>
        <Link to="/detect" className="btn-ghost">
          <ScanLine size={16} /> Stage 2
        </Link>
        <Link to="/report" className="btn-ghost">
          <FileText size={16} /> Stage 3
        </Link>
        <button type="button" onClick={openHelp} className="btn-ghost">
          <HelpCircle size={16} /> Page-by-page tips
        </button>
      </Reveal>
    </Section>
  )
}

export default function Home() {
  return (
    <>
      <SectionNav />
      <Hero />
      <HowToUse />

      <Section id="objectives" eyebrow="Project objectives" title="Four goals, one trustworthy pipeline" intro="To build one pipeline that takes a raw chest X-ray and produces a diagnosis you can actually trust.">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {OBJECTIVES.map((o, i) => (
            <Reveal key={o.title} delay={i * 0.08}>
              <Card hover className="h-full">
                <div className="mb-4 flex items-center justify-between">
                  <span className="grid h-10 w-10 place-items-center rounded-xl bg-primary/12 text-primary">
                    <Target size={20} />
                  </span>
                  <span className="font-display text-3xl font-bold text-ink/10">0{i + 1}</span>
                </div>
                <h3 className="font-display text-lg font-semibold">{o.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{o.body}</p>
              </Card>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section id="problem" eyebrow="Problem statement" title="Chest X-ray AI has three unresolved challenges" className="bg-elev/40">
        <div className="grid gap-4 md:grid-cols-3">
          {PROBLEMS.map((p, i) => {
            const Icon = PROBLEM_ICONS[p.icon]
            return (
              <Reveal key={p.title} delay={i * 0.1}>
                <Card hover className="h-full border-t-2 border-t-bad/60">
                  <Icon className="mb-4 text-bad" size={28} />
                  <h3 className="font-display text-lg font-semibold">{p.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted">{p.body}</p>
                </Card>
              </Reveal>
            )
          })}
        </div>
        <Reveal className="mt-10">
          <p className="label mb-4">Research gaps we address</p>
          <div className="grid gap-4 md:grid-cols-3">
            {GAPS.map((g) => (
              <div
                key={g.title}
                className="group relative overflow-hidden rounded-2xl bg-ok/[0.04] p-5 pl-6 ring-1 ring-ok/20 transition duration-300 hover:-translate-y-1 hover:ring-ok/50 hover:shadow-[0_8px_32px_-10px_rgb(var(--ok)/0.45)]"
              >
                <span className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-ok via-primary to-ok/30" aria-hidden />
                <span className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-ok/70 via-primary/40 to-transparent" aria-hidden />
                <p className="flex items-center gap-2 font-semibold">
                  <Lightbulb size={16} className="shrink-0 text-ok" /> {g.title}
                </p>
                <p className="mt-2 text-sm leading-relaxed text-muted">{g.body}</p>
              </div>
            ))}
          </div>
        </Reveal>
      </Section>

      <Section id="methodology" eyebrow="Methodology" title="A three-stage, explainable pipeline" intro="Click a stage to see how it works. Each stage can also be run on its own from the navbar.">
        <Reveal>
          <PipelineDiagram />
        </Reveal>
        <div className="mt-10 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Reveal>
            <Figure {...FIGURES.architecture} />
          </Reveal>
          <div className="space-y-4">
            {DATASETS.map((d, i) => (
              <Reveal key={d.name} delay={i * 0.1}>
                <Card>
                  <div className="mb-3 flex items-center justify-between gap-2">
                    <p className="font-display text-lg font-semibold">{d.name}</p>
                    <Badge tone="accent">{d.use}</Badge>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {d.stats.map((s) => (
                      <div key={s.label}>
                        <p className="font-display text-xl font-bold text-primary">
                          <Counter value={s.value} decimals={s.decimals || 0} suffix={s.suffix || ''} />
                        </p>
                        <p className="text-xs text-muted">{s.label}</p>
                      </div>
                    ))}
                  </div>
                  <p className="mt-3 text-xs leading-relaxed text-muted">{d.body}</p>
                </Card>
              </Reveal>
            ))}
          </div>
        </div>
        <Results />
      </Section>

      <Section id="novelty" eyebrow="Novelty & key contributions" title="What ReXTrust adds" className="bg-elev/40">
        <div className="grid gap-5 lg:grid-cols-3">
          {NOVELTY.map((n, i) => (
            <Reveal key={n.n} delay={i * 0.1}>
              <Card hover className="relative h-full overflow-hidden">
                <span className="absolute -right-3 -top-6 font-display text-[7rem] font-bold leading-none text-primary/[0.08]" aria-hidden>
                  {n.n}
                </span>
                <p className="label text-primary">Novelty {n.n}</p>
                <h3 className="mt-2 font-display text-xl font-semibold">{n.title}</h3>
                <ul className="mt-4 space-y-2.5">
                  {n.points.map((p) => (
                    <li key={p} className="flex gap-2.5 text-sm text-muted">
                      <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" /> {p}
                    </li>
                  ))}
                </ul>
              </Card>
            </Reveal>
          ))}
        </div>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <StageLinkCard to="/enhance" icon={Sparkles} title="Stage 1 · Enhancement" body="Compare 6 runnable enhancers with a before/after slider." />
          <StageLinkCard to="/detect" icon={ScanLine} tone="accent" title="Stage 2 · Detection" body="Per-disease probabilities and independent Score-CAM maps." />
          <StageLinkCard to="/report" icon={FileText} tone="cardio" title="Stage 3 · Report" body="MediVLM report verified claim by claim with RadCheck." />
        </div>
      </Section>

      <Section id="team" eyebrow="Team details" title="Team 95">
        <Reveal className="mb-8">
          <Card className="flex flex-col items-start gap-4 sm:flex-row sm:items-center">
            <span className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-primary to-accent text-white">
              <GraduationCap size={26} />
            </span>
            <div>
              <p className="label">Project guide</p>
              <p className="font-display text-xl font-semibold">{PROJECT.guide.name}</p>
              <p className="text-sm text-muted">{PROJECT.guide.role}</p>
            </div>
          </Card>
        </Reveal>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-5">
          {TEAM.map((m, i) => (
            <TeamCard key={m.reg} m={m} i={i} />
          ))}
        </div>
      </Section>

      <Section id="progress" eyebrow="Project progress" title="Where the project stands" intro="Milestones as presented across the reviews. The current focus is end-to-end integration — which this interface demonstrates." className="bg-elev/40">
        <Timeline items={TIMELINE} />
      </Section>

      <Section id="docs" eyebrow="Documentation" title="Review decks, setup and API">
        <Documentation />
      </Section>

      <Section id="github" eyebrow="Source code" title="GitHub repository" className="bg-elev/40">
        <Reveal>
          <Card className="relative overflow-hidden p-8 sm:p-10">
            <div className="absolute -right-10 -top-10 h-56 w-56 rounded-full bg-accent/20 blur-3xl" aria-hidden />
            <div className="relative flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
              <div className="flex items-start gap-5">
                <span className="grid h-16 w-16 shrink-0 place-items-center rounded-2xl bg-ink text-bg">
                  <Github size={32} />
                </span>
                <div>
                  <p className="font-display text-2xl font-bold">rextrust</p>
                  <p className="mt-1 max-w-lg text-sm text-muted">Pipeline core, trained-model registry, FastAPI backend and this React interface.</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {['Python', 'PyTorch', 'FastAPI', 'React', 'Tailwind'].map((t) => (
                      <Badge key={t}>{t}</Badge>
                    ))}
                  </div>
                </div>
              </div>
              {isTodo(LINKS.github) ? (
                <div className="text-sm">
                  <button type="button" disabled className="btn-primary px-6 py-3">
                    <Github size={18} /> View on GitHub
                  </button>
                  <p className="mt-2 text-xs text-warn">TODO_GITHUB_URL — set LINKS.github in src/lib/content.js</p>
                </div>
              ) : (
                <a href={LINKS.github} target="_blank" rel="noreferrer" className="btn-primary px-6 py-3">
                  <Github size={18} /> View on GitHub <ExternalLink size={14} />
                </a>
              )}
            </div>
          </Card>
        </Reveal>
      </Section>

      <Section id="video" eyebrow="Demonstration" title="Watch ReXTrust in action">
        <Reveal>
          <VideoEmbed />
        </Reveal>
        <Reveal className="mt-10 text-center">
          <Link to="/pipeline" className="btn-primary px-6 py-3 text-base">
            <Workflow size={18} /> Run the complete pipeline
          </Link>
        </Reveal>
      </Section>
    </>
  )
}
