import { AnimatePresence, motion, useScroll, useSpring } from 'framer-motion'
import { ArrowRight, ArrowUp, CornerDownLeft, FileText, Hash, HelpCircle, Home, Moon, ScanLine, Search, Sparkles, Workflow, X } from 'lucide-react'
import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { HOME_SECTIONS, HOW_TO } from '../lib/content'
import { cx } from '../lib/utils'

// ── shared open/close state for the help drawer and command palette ──────────
export const UiContext = createContext({ openHelp: () => {}, openPalette: () => {} })
export const useUi = () => useContext(UiContext)

/** Scroll to a home-page section, navigating to the home page first if needed. */
export function useGoToSection() {
  const nav = useNavigate()
  const loc = useLocation()
  return (id) => {
    const go = () => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    if (loc.pathname === '/') go()
    else {
      nav('/')
      setTimeout(go, 120)
    }
  }
}

// ── thin gradient bar showing scroll progress ────────────────────────────────
export function ScrollProgress() {
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 200, damping: 30, restDelta: 0.001 })
  return (
    <motion.div
      aria-hidden
      style={{ scaleX }}
      className="fixed inset-x-0 top-0 z-[60] h-0.5 origin-left bg-gradient-to-r from-primary via-accent to-dz-cardio"
    />
  )
}

export function BackToTop() {
  const [show, setShow] = useState(false)
  useEffect(() => {
    const on = () => setShow(window.scrollY > 700)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  return (
    <AnimatePresence>
      {show && (
        <motion.button
          type="button"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 12 }}
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="glass fixed bottom-6 left-6 z-40 grid h-11 w-11 place-items-center !rounded-full text-primary hover:shadow-glow"
          aria-label="Back to top"
        >
          <ArrowUp size={18} />
        </motion.button>
      )}
    </AnimatePresence>
  )
}

// ── "How to use" content (home section + drawer) ─────────────────────────────
export function HowToSteps({ compact }) {
  return (
    <ol className={cx('grid gap-4', compact ? 'grid-cols-1' : 'sm:grid-cols-2 lg:grid-cols-4')}>
      {HOW_TO.steps.map((s, i) => (
        <li key={s.title} className={cx('relative rounded-2xl ring-1 ring-line/15', compact ? 'bg-ink/[0.03] p-4' : 'glass p-5')}>
          <div className="mb-3 flex items-center gap-3">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-primary to-accent font-display text-sm font-bold text-white shadow-glow">
              {i + 1}
            </span>
            <p className="font-display font-semibold">{s.title}</p>
          </div>
          <p className="text-sm leading-relaxed text-muted">{s.body}</p>
          {!compact && i < HOW_TO.steps.length - 1 && (
            <ArrowRight size={16} className="absolute -right-3 top-1/2 hidden -translate-y-1/2 text-primary/60 lg:block" aria-hidden />
          )}
        </li>
      ))}
    </ol>
  )
}

const PAGE_LABEL = { '/': 'this portal', '/enhance': 'Stage 1 · Enhancement', '/detect': 'Stage 2 · Detection', '/report': 'Stage 3 · Report', '/pipeline': 'the Complete Pipeline' }

export function HelpDrawer({ open, onClose }) {
  const loc = useLocation()
  const tips = HOW_TO.pages[loc.pathname] || HOW_TO.pages['/']
  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  useEffect(() => onClose(), [loc.pathname]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-[80] bg-black/50 backdrop-blur-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.aside
            role="dialog"
            aria-modal="true"
            aria-label="How to use"
            className="fixed inset-y-0 right-0 z-[90] flex w-full max-w-md flex-col border-l hairline bg-elev shadow-2xl"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
          >
            <div className="flex items-center justify-between border-b hairline px-5 py-4">
              <div className="flex items-center gap-2">
                <HelpCircle size={18} className="text-primary" />
                <p className="font-display text-lg font-semibold">How to use</p>
              </div>
              <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-lg ring-1 ring-line/15 hover:bg-ink/5" aria-label="Close" autoFocus>
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 space-y-6 overflow-y-auto px-5 py-5">
              <div>
                <p className="label mb-3 text-primary">On {PAGE_LABEL[loc.pathname] || 'this page'}</p>
                <ul className="space-y-2">
                  {tips.map((t) => (
                    <li key={t} className="flex gap-2.5 rounded-xl bg-primary/[0.06] px-3 py-2.5 text-sm ring-1 ring-primary/15">
                      <Sparkles size={14} className="mt-0.5 shrink-0 text-primary" /> {t}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="label mb-3">In four steps</p>
                <HowToSteps compact />
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  ['/enhance', 'Enhancement', Sparkles],
                  ['/detect', 'Detection', ScanLine],
                  ['/report', 'Report', FileText],
                  ['/pipeline', 'Pipeline', Workflow],
                ].map(([to, l, I]) => (
                  <Link key={to} to={to} className="btn-ghost justify-start text-xs">
                    <I size={14} /> {l}
                  </Link>
                ))}
              </div>
              <p className="text-xs text-muted">
                Tip: press <Kbd>Ctrl</Kbd> <Kbd>K</Kbd> anywhere to jump to a page or section.
              </p>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

export function Kbd({ children }) {
  return <kbd className="rounded-md bg-ink/[0.07] px-1.5 py-0.5 font-mono text-[11px] text-ink ring-1 ring-line/20">{children}</kbd>
}

// ── command palette (Ctrl/⌘ + K) ─────────────────────────────────────────────
export function CommandPalette({ open, onClose, toggleTheme, openHelp }) {
  const nav = useNavigate()
  const goSection = useGoToSection()
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const input = useRef(null)

  const items = useMemo(
    () => [
      { group: 'Pages', label: 'Portal', icon: Home, run: () => nav('/') },
      { group: 'Pages', label: 'Stage 1 · Image Enhancement', icon: Sparkles, run: () => nav('/enhance') },
      { group: 'Pages', label: 'Stage 2 · Disease Detection', icon: ScanLine, run: () => nav('/detect') },
      { group: 'Pages', label: 'Stage 3 · Report Generation', icon: FileText, run: () => nav('/report') },
      { group: 'Pages', label: 'Complete Pipeline', icon: Workflow, run: () => nav('/pipeline') },
      ...HOME_SECTIONS.map((s) => ({ group: 'Portal sections', label: s.label, icon: Hash, run: () => goSection(s.id) })),
      { group: 'Actions', label: 'How to use', icon: HelpCircle, run: openHelp },
      { group: 'Actions', label: 'Toggle light / dark theme', icon: Moon, run: toggleTheme },
    ],
    [nav, goSection, openHelp, toggleTheme],
  )
  const shown = items.filter((x) => x.label.toLowerCase().includes(q.trim().toLowerCase()))

  useEffect(() => {
    if (open) {
      setQ('')
      setI(0)
      setTimeout(() => input.current?.focus(), 30)
    }
  }, [open])

  const pick = (x) => {
    onClose()
    setTimeout(x.run, 10)
  }
  const onKey = (e) => {
    if (e.key === 'Escape') onClose()
    if (e.key === 'ArrowDown') (e.preventDefault(), setI((v) => Math.min(v + 1, shown.length - 1)))
    if (e.key === 'ArrowUp') (e.preventDefault(), setI((v) => Math.max(v - 1, 0)))
    if (e.key === 'Enter' && shown[i]) pick(shown[i])
  }

  let lastGroup = null
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[95] flex items-start justify-center bg-black/50 px-4 pt-[12vh] backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            onClick={(e) => e.stopPropagation()}
            initial={{ scale: 0.96, y: -8 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.96, y: -8 }}
            className="w-full max-w-lg overflow-hidden rounded-2xl bg-elev shadow-2xl ring-1 ring-line/25"
          >
            <div className="flex items-center gap-3 border-b hairline px-4">
              <Search size={18} className="text-muted" />
              <input
                ref={input}
                value={q}
                onChange={(e) => (setQ(e.target.value), setI(0))}
                onKeyDown={onKey}
                placeholder="Jump to a page, section or action…"
                className="h-14 flex-1 bg-transparent text-sm outline-none placeholder:text-muted"
                aria-label="Search commands"
              />
              <Kbd>Esc</Kbd>
            </div>
            <ul className="max-h-[50vh] overflow-y-auto p-2" role="listbox">
              {shown.length === 0 && <li className="px-3 py-6 text-center text-sm text-muted">No matches</li>}
              {shown.map((x, k) => {
                const header = x.group !== lastGroup
                lastGroup = x.group
                return (
                  <li key={x.group + x.label}>
                    {header && <p className="label px-3 pb-1 pt-3">{x.group}</p>}
                    <button
                      type="button"
                      role="option"
                      aria-selected={k === i}
                      onMouseEnter={() => setI(k)}
                      onClick={() => pick(x)}
                      className={cx('flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition', k === i ? 'bg-primary/12 text-ink' : 'text-muted')}
                    >
                      <x.icon size={16} className={k === i ? 'text-primary' : ''} />
                      <span className="flex-1">{x.label}</span>
                      {k === i && <CornerDownLeft size={14} className="text-primary" />}
                    </button>
                  </li>
                )
              })}
            </ul>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ── sticky section bar for the home page (scroll-spy) ────────────────────────
export function SectionNav() {
  const [active, setActive] = useState('overview')
  const bar = useRef(null)

  useEffect(() => {
    const els = HOME_SECTIONS.map((s) => document.getElementById(s.id)).filter(Boolean)
    const obs = new IntersectionObserver(
      (entries) => {
        const hit = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0]
        if (hit) setActive(hit.target.id)
      },
      { rootMargin: '-120px 0px -55% 0px' },
    )
    els.forEach((e) => obs.observe(e))
    return () => obs.disconnect()
  }, [])

  // keep the active chip in view on narrow screens
  useEffect(() => {
    const chip = bar.current?.querySelector(`[data-id="${active}"]`)
    chip?.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' })
  }, [active])

  const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  return (
    <div className="fixed inset-x-0 top-16 z-40 border-b hairline bg-bg/80 backdrop-blur-xl">
      <nav aria-label="Page sections" className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <ul ref={bar} className="flex gap-1 overflow-x-auto py-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {HOME_SECTIONS.map((s) => (
            <li key={s.id} className="shrink-0">
              <button
                type="button"
                data-id={s.id}
                onClick={() => go(s.id)}
                aria-current={active === s.id ? 'location' : undefined}
                className={cx(
                  'rounded-lg px-3 py-1.5 text-xs font-medium transition sm:text-[13px]',
                  active === s.id ? 'bg-primary/15 text-primary ring-1 ring-primary/30' : 'text-muted hover:bg-ink/5 hover:text-ink',
                )}
              >
                {s.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
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
