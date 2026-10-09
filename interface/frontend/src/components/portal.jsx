import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, Check, CircleDot, Github, Linkedin, Loader2, ScanLine, ShieldCheck, Sparkles } from 'lucide-react'
import { Fragment, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { STAGES, isTodo } from '../lib/content'
import { cx } from '../lib/utils'
import { Badge } from './ui'

// ── hero visual: real pipeline output from results/1 cycling through the stages ────────────────────
const HERO_FRAMES = [
  { src: '/hero/original.jpg', label: 'Input X-ray', tone: 'muted' },
  { src: '/hero/enhanced.jpg', label: 'Attention U-Net + CLAHE', tone: 'primary' },
  { src: '/hero/cardiomegaly.jpg', label: 'Score-CAM · Cardiomegaly', tone: 'd-cardio' },
  { src: '/hero/edema.jpg', label: 'Score-CAM · Edema', tone: 'd-edema' },
  { src: '/hero/pleural_effusion.jpg', label: 'Score-CAM · Pleural Effusion', tone: 'd-effusion' },
]

export function HeroVisual() {
  const [i, setI] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setI((x) => (x + 1) % HERO_FRAMES.length), 2600)
    return () => clearInterval(id)
  }, [])
  const f = HERO_FRAMES[i]
  return (
    <div className="relative mx-auto w-full max-w-md animate-float">
      <div className="absolute -inset-6 rounded-[2.5rem] bg-gradient-to-br from-primary/30 via-accent/20 to-transparent blur-2xl" aria-hidden />
      <div className="glass relative overflow-hidden p-3">
        <div className="relative aspect-square overflow-hidden rounded-xl bg-black">
          <AnimatePresence mode="popLayout">
            <motion.img
              key={f.src}
              src={f.src}
              alt={f.label}
              className="absolute inset-0 h-full w-full object-cover"
              initial={{ opacity: 0, scale: 1.04 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.8 }}
            />
          </AnimatePresence>
          <div className="pointer-events-none absolute inset-x-0 h-px animate-scan bg-primary shadow-[0_0_16px_4px_rgb(var(--primary)/0.6)]" aria-hidden />
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-lg bg-black/60 px-2.5 py-1.5 text-xs font-medium text-white backdrop-blur">
            <span className="h-2 w-2 rounded-full" style={{ background: `rgb(var(--${f.tone === 'muted' ? 'muted' : f.tone}))` }} />
            {f.label}
          </div>
        </div>
        <div className="mt-3 flex gap-1.5" aria-hidden>
          {HERO_FRAMES.map((_, k) => (
            <span key={k} className={cx('h-1 flex-1 rounded-full transition', k <= i ? 'bg-primary' : 'bg-ink/10')} />
          ))}
        </div>
      </div>
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.8 }}
        className="glass absolute -right-3 top-10 hidden items-center gap-2 px-3 py-2 text-xs sm:flex"
      >
        <ShieldCheck size={16} className="text-ok" /> RadCheck verified
      </motion.div>
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 1.1 }}
        className="glass absolute -left-4 bottom-16 hidden items-center gap-2 px-3 py-2 text-xs sm:flex"
      >
        <ScanLine size={16} className="text-accent" /> 3 independent heatmaps
      </motion.div>
    </div>
  )
}

export function HeroBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      <div className="grid-bg absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_75%)]" />
      <motion.div
        className="absolute -left-32 top-10 h-[28rem] w-[28rem] rounded-full bg-primary/25 blur-[110px]"
        animate={{ x: [0, 60, 0], y: [0, 40, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute -right-24 top-40 h-[26rem] w-[26rem] rounded-full bg-accent/25 blur-[110px]"
        animate={{ x: [0, -50, 0], y: [0, -30, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-dz-cardio/15 blur-[100px]"
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
      />
    </div>
  )
}

// ── interactive pipeline diagram ─────────────────────────────────────────────
const NODE_COLOR = {
  muted: 'from-slate-500/30 to-slate-500/5 text-ink',
  primary: 'from-primary/35 to-primary/5 text-primary',
  accent: 'from-accent/35 to-accent/5 text-accent',
  'd-cardio': 'from-dz-cardio/35 to-dz-cardio/5 text-dz-cardio',
  ok: 'from-ok/35 to-ok/5 text-ok',
}

function Connector({ vertical }) {
  return (
    <div className={cx('relative flex items-center justify-center', vertical ? 'h-8 w-full' : 'w-8 shrink-0 self-center')} aria-hidden>
      <svg className={vertical ? 'h-8 w-4' : 'h-4 w-8'} viewBox={vertical ? '0 0 16 32' : '0 0 32 16'}>
        <line
          x1={vertical ? 8 : 0}
          y1={vertical ? 0 : 8}
          x2={vertical ? 8 : 32}
          y2={vertical ? 32 : 8}
          stroke="rgb(var(--primary))"
          strokeWidth="2"
          strokeDasharray="4 4"
        >
          <animate attributeName="stroke-dashoffset" from="16" to="0" dur="0.8s" repeatCount="indefinite" />
        </line>
      </svg>
    </div>
  )
}

export function PipelineDiagram() {
  const [active, setActive] = useState('stage1')
  const stage = STAGES.find((s) => s.id === active)
  return (
    <div>
      <div className="flex flex-col items-stretch lg:flex-row" role="tablist" aria-label="Pipeline stages">
        {STAGES.map((s, i) => (
          <Fragment key={s.id}>
            {i > 0 && (
              <>
                <div className="hidden lg:flex">
                  <Connector />
                </div>
                <div className="lg:hidden">
                  <Connector vertical />
                </div>
              </>
            )}
            <motion.button
              type="button"
              role="tab"
              aria-selected={active === s.id}
              onClick={() => setActive(s.id)}
              whileHover={{ y: -3 }}
              whileTap={{ scale: 0.98 }}
              className={cx(
                'relative flex-1 overflow-hidden rounded-2xl bg-gradient-to-b p-4 text-left ring-1 transition',
                NODE_COLOR[s.color],
                active === s.id ? 'shadow-glow ring-2 ring-current' : 'ring-line/15 hover:ring-line/40',
              )}
            >
              <span className="label block text-current opacity-80">{i === 0 ? 'Input' : i === STAGES.length - 1 ? 'Output' : `Stage ${i}`}</span>
              <span className="mt-1 block font-display text-[15px] font-semibold leading-snug text-ink">{s.label.replace(/^Stage \d · /, '')}</span>
              <span className="mt-1 block text-xs text-muted">{s.sub}</span>
              {s.id === 'stage1' && (
                <span className="mt-2 inline-flex items-center gap-1 rounded-full bg-ok/15 px-2 py-0.5 text-[10px] font-medium text-ok">
                  <CircleDot size={10} /> good quality → skip
                </span>
              )}
            </motion.button>
          </Fragment>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={stage.id}
          role="tabpanel"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.3 }}
          className="glass mt-6 grid gap-6 p-6 md:grid-cols-[1.3fr_1fr]"
        >
          <div>
            <h3 className="font-display text-xl font-semibold">{stage.label}</h3>
            <p className="mt-2 leading-relaxed text-muted">{stage.details.what}</p>
            {stage.route && (
              <Link to={stage.route} className="btn-ghost mt-5">
                Try it live <ArrowRight size={16} />
              </Link>
            )}
          </div>
          <ul className="space-y-2.5">
            {stage.details.bullets.map((b) => (
              <li key={b} className="flex gap-2.5 text-sm">
                <Check size={16} className="mt-0.5 shrink-0 text-primary" /> <span>{b}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

// ── timeline ─────────────────────────────────────────────────────────────────
export function Timeline({ items }) {
  const done = items.filter((x) => x.status === 'done').length
  return (
    <div>
      <div className="mb-8 flex flex-wrap items-center gap-4">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink/[0.07]" role="progressbar" aria-valuenow={done} aria-valuemin={0} aria-valuemax={items.length} aria-label="Milestones completed">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
            initial={{ width: 0 }}
            whileInView={{ width: `${(done / items.length) * 100}%` }}
            viewport={{ once: true }}
            transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
          />
        </div>
        <span className="text-sm text-muted">
          <b className="text-ink">{done}</b> of {items.length} milestones complete
        </span>
      </div>
      <ol className="relative ml-3 border-l-2 border-line/20 sm:ml-0 sm:border-l-0">
        <div className="absolute bottom-0 left-1/2 top-0 hidden w-0.5 -translate-x-1/2 bg-line/20 sm:block" aria-hidden />
        {items.map((it, i) => {
          const left = i % 2 === 0
          const Icon = it.status === 'done' ? Check : it.status === 'active' ? Loader2 : CircleDot
          const tone = it.status === 'done' ? 'bg-ok text-white' : it.status === 'active' ? 'bg-primary text-white' : 'bg-elev text-muted ring-1 ring-line/30'
          return (
            <motion.li
              key={it.title}
              initial={{ opacity: 0, x: left ? -24 : 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: '-40px' }}
              transition={{ duration: 0.5, delay: 0.05 }}
              className={cx('relative mb-6 pl-8 sm:w-1/2 sm:pl-0', left ? 'sm:pr-10 sm:text-right' : 'sm:ml-auto sm:pl-10')}
            >
              <span className={cx('absolute -left-[13px] top-1 grid h-6 w-6 place-items-center rounded-full sm:top-1', left ? 'sm:-right-3 sm:left-auto' : 'sm:-left-3', tone)}>
                <Icon size={13} className={it.status === 'active' ? 'animate-spin [animation-duration:3s]' : ''} />
              </span>
              <div className={cx('glass p-4', isTodo(it.title) && 'border-dashed')}>
                <div className={cx('mb-1 flex flex-wrap items-center gap-2', left && 'sm:justify-end')}>
                  <Badge tone={it.status === 'done' ? 'ok' : it.status === 'active' ? 'primary' : 'muted'}>{it.phase}</Badge>
                  <span className="text-xs text-muted">{it.status === 'done' ? 'Completed' : it.status === 'active' ? 'In progress' : 'Planned'}</span>
                </div>
                <h3 className="font-display font-semibold">{it.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted">{it.body}</p>
              </div>
            </motion.li>
          )
        })}
      </ol>
    </div>
  )
}

// ── team ─────────────────────────────────────────────────────────────────────
function SocialLink({ href, icon: Icon, label }) {
  const todo = isTodo(href)
  return (
    <a
      href={todo ? '#' : href}
      target={todo ? undefined : '_blank'}
      rel="noreferrer"
      onClick={(e) => {
        if (todo) {
          e.preventDefault()
          toast.info(`${label} link not set yet`, { description: 'Add it in src/lib/content.js (TEAM).' })
        }
      }}
      className="grid h-9 w-9 place-items-center rounded-lg text-muted ring-1 ring-line/20 transition hover:bg-primary/10 hover:text-primary"
      aria-label={label}
    >
      <Icon size={16} />
    </a>
  )
}

export function TeamCard({ m, i }) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: i * 0.08 }}
      className="glass group p-5 text-center transition duration-300 hover:-translate-y-1.5 hover:shadow-glow"
    >
      <div className="relative mx-auto mb-4 h-28 w-28">
        <div className="absolute -inset-1 rounded-full bg-gradient-to-br from-primary to-accent opacity-60 blur-sm transition group-hover:opacity-100" />
        <img src={m.photo} alt={`Photo of ${m.name}`} className="relative h-28 w-28 rounded-full object-cover ring-4 ring-bg" loading="lazy" />
      </div>
      <h3 className="font-display text-lg font-semibold">{m.name}</h3>
      <p className="font-mono text-xs text-muted">{m.reg}</p>
      <div className="mt-4 flex justify-center gap-2">
        <SocialLink href={m.linkedin} icon={Linkedin} label={`${m.name} on LinkedIn`} />
        <SocialLink href={m.github} icon={Github} label={`${m.name} on GitHub`} />
      </div>
    </motion.article>
  )
}

const ICON_TONE = {
  primary: 'bg-primary/15 text-primary',
  accent: 'bg-accent/15 text-accent',
  cardio: 'bg-dz-cardio/15 text-dz-cardio',
  ok: 'bg-ok/15 text-ok',
}

export function StageLinkCard({ to, icon: Icon = Sparkles, title, body, tone = 'primary' }) {
  return (
    <Link to={to} className="glass group flex items-start gap-4 p-5 transition hover:-translate-y-1 hover:shadow-glow">
      <span className={cx('grid h-11 w-11 shrink-0 place-items-center rounded-xl', ICON_TONE[tone])}>
        <Icon size={20} />
      </span>
      <span className="flex-1">
        <span className="flex items-center gap-1 font-display font-semibold">
          {title} <ArrowRight size={15} className="transition group-hover:translate-x-1" />
        </span>
        <span className="mt-1 block text-sm text-muted">{body}</span>
      </span>
    </Link>
  )
}
