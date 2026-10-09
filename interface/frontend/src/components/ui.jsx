import { AnimatePresence, animate, motion, useInView } from 'framer-motion'
import { AlertTriangle, ChevronDown, FlaskConical, Info, Maximize2, X } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { cx } from '../lib/utils'

export function Reveal({ children, delay = 0, className, y = 18 }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

export function Section({ id, eyebrow, title, intro, children, className }) {
  return (
    <section id={id} className={cx('scroll-mt-28 py-16 sm:py-24', className)} aria-labelledby={id && `${id}-title`}>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {(eyebrow || title) && (
          <Reveal className="mb-10 max-w-3xl sm:mb-14">
            {eyebrow && <p className="label mb-3 text-primary">{eyebrow}</p>}
            {title && (
              <h2 id={id && `${id}-title`} className="text-3xl font-bold sm:text-4xl">
                {title}
              </h2>
            )}
            {intro && <p className="mt-4 text-base leading-relaxed text-muted sm:text-lg">{intro}</p>}
          </Reveal>
        )}
        {children}
      </div>
    </section>
  )
}

const trackPointer = (e) => {
  const r = e.currentTarget.getBoundingClientRect()
  e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
  e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
}

export function Card({ className, children, hover, spotlight = true, as: As = 'div', ...rest }) {
  return (
    <As
      onMouseMove={spotlight ? trackPointer : undefined}
      className={cx('glass p-5 sm:p-6', spotlight && 'spotlight', hover && 'transition duration-300 hover:-translate-y-1 hover:shadow-glow', className)}
      {...rest}
    >
      {children}
    </As>
  )
}

export function PageHeader({ eyebrow, title, intro, icon: Icon, children }) {
  return (
    <header className="relative overflow-hidden border-b hairline">
      <div className="grid-bg pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]" />
      <div className="relative mx-auto max-w-7xl px-4 pb-10 pt-28 sm:px-6 sm:pt-32 lg:px-8">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <p className="label mb-3 flex items-center gap-2 text-primary">
            {Icon && <Icon size={14} aria-hidden />} {eyebrow}
          </p>
          <h1 className="text-3xl font-bold sm:text-5xl">{title}</h1>
          {intro && <p className="mt-4 max-w-3xl text-base leading-relaxed text-muted sm:text-lg">{intro}</p>}
          {children}
        </motion.div>
      </div>
    </header>
  )
}

const TONES = {
  primary: 'bg-primary/12 text-primary ring-1 ring-primary/25',
  accent: 'bg-accent/12 text-accent ring-1 ring-accent/25',
  ok: 'bg-ok/12 text-ok ring-1 ring-ok/25',
  warn: 'bg-warn/12 text-warn ring-1 ring-warn/30',
  bad: 'bg-bad/12 text-bad ring-1 ring-bad/25',
  muted: 'bg-ink/5 text-muted ring-1 ring-line/20',
}

export function Badge({ tone = 'muted', children, className, icon: Icon }) {
  return (
    <span className={cx('chip', TONES[tone], className)}>
      {Icon && <Icon size={12} aria-hidden />}
      {children}
    </span>
  )
}

export function ModeBadge({ mode }) {
  if (!mode) return null
  return mode === 'mock' ? (
    <Badge tone="warn" icon={FlaskConical}>
      MOCK OUTPUT
    </Badge>
  ) : (
    <Badge tone="ok">
      <span className="h-1.5 w-1.5 rounded-full bg-ok" /> Real model
    </Badge>
  )
}

export function Notice({ tone = 'warn', children, className }) {
  const Icon = tone === 'bad' ? AlertTriangle : Info
  return (
    <div
      role={tone === 'bad' ? 'alert' : 'status'}
      className={cx('flex items-start gap-3 rounded-xl px-4 py-3 text-sm', TONES[tone], className)}
    >
      <Icon size={16} className="mt-0.5 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1 leading-relaxed">{children}</div>
    </div>
  )
}

export function Skeleton({ className }) {
  return (
    <div className={cx('relative overflow-hidden rounded-xl bg-ink/[0.06]', className)} aria-hidden>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-ink/[0.07] to-transparent" />
    </div>
  )
}

export function Counter({ value, decimals = 0, suffix = '', prefix = '', duration = 1.6, className }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    if (!inView) return
    const c = animate(0, value, { duration, ease: [0.16, 1, 0.3, 1], onUpdate: setDisplay })
    return () => c.stop()
  }, [inView, value, duration])
  const text = display.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
  return (
    <span ref={ref} className={cx('tabular-nums', className)} aria-label={`${prefix}${value}${suffix}`}>
      {prefix}
      {text}
      {suffix}
    </span>
  )
}

export function EmptyState({ icon: Icon, title, children, className }) {
  return (
    <div className={cx('flex flex-col items-center justify-center rounded-2xl border border-dashed hairline px-6 py-14 text-center', className)}>
      {Icon && (
        <div className="mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary">
          <Icon size={26} aria-hidden />
        </div>
      )}
      <p className="font-display text-lg font-semibold">{title}</p>
      {children && <div className="mt-2 max-w-sm text-sm text-muted">{children}</div>}
    </div>
  )
}

export function Tabs({ tabs, value, onChange, className, size = 'md' }) {
  const id = useId()
  const onKey = (e) => {
    const i = tabs.findIndex((t) => t.value === value)
    if (e.key === 'ArrowRight') onChange(tabs[(i + 1) % tabs.length].value)
    if (e.key === 'ArrowLeft') onChange(tabs[(i - 1 + tabs.length) % tabs.length].value)
  }
  return (
    <div role="tablist" onKeyDown={onKey} className={cx('inline-flex flex-wrap gap-1 rounded-xl bg-ink/[0.05] p-1 ring-1 ring-line/15', className)}>
      {tabs.map((t) => {
        const active = t.value === value
        return (
          <button
            key={t.value}
            role="tab"
            type="button"
            aria-selected={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(t.value)}
            className={cx(
              'relative rounded-lg font-medium transition',
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-1.5 text-sm',
              active ? 'text-ink' : 'text-muted hover:text-ink',
            )}
          >
            {active && (
              <motion.span
                layoutId={`tab-${id}`}
                className="absolute inset-0 rounded-lg bg-elev shadow-soft ring-1 ring-line/20"
                transition={{ type: 'spring', bounce: 0.2, duration: 0.45 }}
              />
            )}
            <span className="relative flex items-center gap-1.5">
              {t.icon && <t.icon size={14} aria-hidden />}
              {t.label}
            </span>
          </button>
        )
      })}
    </div>
  )
}

export function Accordion({ items, defaultOpen = 0 }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="divide-y divide-line/15 overflow-hidden rounded-2xl ring-1 ring-line/15">
      {items.map((it, i) => {
        const isOpen = open === i
        return (
          <div key={it.title} className="bg-card/[0.02]">
            <h3>
              <button
                type="button"
                aria-expanded={isOpen}
                onClick={() => setOpen(isOpen ? -1 : i)}
                className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left font-display font-semibold transition hover:bg-ink/[0.03]"
              >
                <span className="flex items-center gap-3">
                  {it.icon && <it.icon size={18} className="text-primary" aria-hidden />}
                  {it.title}
                </span>
                <ChevronDown size={18} className={cx('shrink-0 text-muted transition-transform', isOpen && 'rotate-180')} />
              </button>
            </h3>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                >
                  <div className="px-5 pb-5 text-sm leading-relaxed text-muted">{it.content}</div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      })}
    </div>
  )
}

/** Semicircle gauge, value in [0,1]. */
export function Gauge({ value, label, tone = 'primary', size = 180 }) {
  const r = 70
  const len = Math.PI * r
  const v = Math.max(0, Math.min(1, value ?? 0))
  return (
    <div className="flex flex-col items-center" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)} aria-label={label}>
      <svg width={size} height={size * 0.6} viewBox="0 0 180 105">
        <path d="M20 95 A70 70 0 0 1 160 95" fill="none" stroke="rgb(var(--line) / 0.18)" strokeWidth="14" strokeLinecap="round" />
        <motion.path
          d="M20 95 A70 70 0 0 1 160 95"
          fill="none"
          stroke={`rgb(var(--${tone}))`}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={len}
          initial={{ strokeDashoffset: len }}
          animate={{ strokeDashoffset: len * (1 - v) }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
        />
        <text x="90" y="88" textAnchor="middle" className="fill-ink font-display" fontSize="30" fontWeight="700">
          {Math.round(v * 100)}%
        </text>
      </svg>
      <p className="label -mt-1">{label}</p>
    </div>
  )
}

export function Figure({ src, caption, className, imgClass }) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])
  return (
    <figure className={cx('group', className)}>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="relative block w-full overflow-hidden rounded-xl bg-white ring-1 ring-line/15"
        aria-label={`Enlarge figure: ${caption}`}
      >
        <img src={src} alt={caption} loading="lazy" className={cx('w-full object-contain transition duration-500 group-hover:scale-[1.02]', imgClass)} />
        <span className="absolute right-2 top-2 grid h-8 w-8 place-items-center rounded-lg bg-black/60 text-white opacity-0 transition group-hover:opacity-100">
          <Maximize2 size={14} />
        </span>
      </button>
      {caption && <figcaption className="mt-2 text-xs text-muted">{caption}</figcaption>}
      <AnimatePresence>
        {open && (
          <motion.div
            className="fixed inset-0 z-[100] grid place-items-center bg-black/80 p-4 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(false)}
            role="dialog"
            aria-modal="true"
            aria-label={caption}
          >
            <motion.img
              src={src}
              alt={caption}
              className="max-h-[88vh] max-w-full rounded-xl bg-white object-contain"
              initial={{ scale: 0.94 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.94 }}
            />
            <button className="absolute right-4 top-4 grid h-10 w-10 place-items-center rounded-full bg-white/10 text-white hover:bg-white/20" aria-label="Close" autoFocus>
              <X size={20} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </figure>
  )
}

export function Stat({ label, value, sub, tone }) {
  return (
    <div className="rounded-xl bg-ink/[0.03] p-3 ring-1 ring-line/10">
      <p className="label">{label}</p>
      <p className={cx('mt-1 font-display text-xl font-semibold tabular-nums', tone && `text-${tone}`)}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </div>
  )
}
