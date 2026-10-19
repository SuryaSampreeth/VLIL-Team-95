import { Check } from 'lucide-react'
import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { cx } from '../lib/utils'

// Cinematic intro, shown once when the site is opened (App mounts once; route changes never re-trigger it).
// The ring plays the real pipeline on a real result (assets/hero): raw X-ray → enhanced → Score-CAM → verified.
const STEPS = [
  { label: 'Conditional Enhancement', short: 'Enhance' },
  { label: 'Multi-Disease Detection', short: 'Detect' },
  { label: 'Per-Disease Score-CAM', short: 'Explain' },
  { label: 'Grounded Report + RadCheck', short: 'Verify' },
]
const START = 900 // ms before the first step completes
const STEP_MS = 650
const FINALE = START + STEP_MS * (STEPS.length - 1) + 650 // all done → shockwave
const LEAVE = FINALE + 650
const TEAL = '45, 212, 191'
const VIOLET = '129, 140, 248'

/** Drifting nodes joined by faint links: a "neural network" backdrop. */
function NeuralField({ reduced }) {
  const ref = useRef(null)
  useEffect(() => {
    const c = ref.current
    const ctx = c.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    let w, h, raf
    const resize = () => {
      w = c.width = window.innerWidth * dpr
      h = c.height = window.innerHeight * dpr
    }
    resize()
    window.addEventListener('resize', resize)
    const n = Math.min(110, Math.floor((window.innerWidth * window.innerHeight) / 13000))
    const pts = Array.from({ length: n }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.45 * dpr,
      vy: (Math.random() - 0.5) * 0.45 * dpr,
      r: (Math.random() * 1.4 + 0.6) * dpr,
      c: Math.random() < 0.7 ? TEAL : VIOLET,
    }))
    const D = 140 * dpr
    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      for (const p of pts) {
        p.x = (p.x + p.vx + w) % w
        p.y = (p.y + p.vy + h) % h
      }
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          const dx = pts[i].x - pts[j].x
          const dy = pts[i].y - pts[j].y
          const d = Math.hypot(dx, dy)
          if (d < D) {
            ctx.strokeStyle = `rgba(${pts[i].c}, ${(1 - d / D) * 0.28})`
            ctx.lineWidth = dpr * 0.8
            ctx.beginPath()
            ctx.moveTo(pts[i].x, pts[i].y)
            ctx.lineTo(pts[j].x, pts[j].y)
            ctx.stroke()
          }
        }
      for (const p of pts) {
        ctx.fillStyle = `rgba(${p.c}, 0.85)`
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fill()
      }
      if (!reduced) raf = requestAnimationFrame(draw)
    }
    draw()
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [reduced])
  return <canvas ref={ref} className="absolute inset-0 h-full w-full" aria-hidden />
}

/** Glowing ring with the X-ray inside; the image evolves with each completed step. */
function Core({ done, finale }) {
  const verified = done >= 4
  return (
    <div className="relative mx-auto h-56 w-56 sm:h-64 sm:w-64">
      {/* soft halo */}
      <div
        className="absolute -inset-10 rounded-full blur-3xl transition-colors duration-700"
        style={{ background: `radial-gradient(circle, rgba(${verified ? '52, 211, 153' : TEAL}, 0.35), transparent 65%)` }}
      />
      {/* outer dashed ring */}
      <svg className="absolute inset-0 h-full w-full animate-[spin_18s_linear_infinite]" viewBox="0 0 200 200" aria-hidden>
        <circle cx="100" cy="100" r="97" fill="none" stroke={`rgba(${TEAL}, 0.35)`} strokeWidth="1" strokeDasharray="2 6" />
      </svg>
      {/* gradient arc, fast, reversed */}
      <svg className="absolute inset-2 h-[calc(100%-1rem)] w-[calc(100%-1rem)] animate-[spin_3.2s_linear_infinite_reverse]" viewBox="0 0 200 200" aria-hidden>
        <defs>
          <linearGradient id="sp-arc" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor={`rgb(${TEAL})`} />
            <stop offset="1" stopColor={`rgb(${VIOLET})`} stopOpacity="0" />
          </linearGradient>
        </defs>
        <circle cx="100" cy="100" r="94" fill="none" stroke="url(#sp-arc)" strokeWidth="3" strokeLinecap="round" strokeDasharray="200 400" />
      </svg>
      {/* progress ring: one quarter per step */}
      <svg className="absolute inset-4 h-[calc(100%-2rem)] w-[calc(100%-2rem)] -rotate-90" viewBox="0 0 200 200" aria-hidden>
        <circle cx="100" cy="100" r="92" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="4" />
        <circle
          cx="100"
          cy="100"
          r="92"
          fill="none"
          stroke={verified ? 'rgb(52, 211, 153)' : `rgb(${TEAL})`}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={2 * Math.PI * 92}
          strokeDashoffset={2 * Math.PI * 92 * (1 - done / 4)}
          style={{ transition: 'stroke-dashoffset 0.6s cubic-bezier(.22,1,.36,1), stroke 0.6s', filter: `drop-shadow(0 0 6px rgba(${TEAL},0.9))` }}
        />
      </svg>
      {/* the X-ray */}
      <div className="absolute inset-8 overflow-hidden rounded-full bg-black ring-1 ring-white/10">
        <img src="/hero/original.jpg" alt="" className="absolute inset-0 h-full w-full object-cover opacity-80" />
        <img
          src="/hero/enhanced.jpg"
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
          style={{ clipPath: done >= 1 ? 'inset(0 0 0 0)' : 'inset(0 0 100% 0)', transition: 'clip-path 0.7s cubic-bezier(.22,1,.36,1)' }}
        />
        <img src="/hero/edema.jpg" alt="" className={cx('absolute inset-0 h-full w-full object-cover transition-opacity duration-700', done >= 2 ? 'opacity-60' : 'opacity-0')} />
        <img src="/hero/cardiomegaly.jpg" alt="" className={cx('absolute inset-0 h-full w-full object-cover transition-opacity duration-700', done >= 3 ? 'opacity-90' : 'opacity-0')} />
        {/* scan line */}
        {!verified && (
          <div className="absolute inset-x-0 h-px animate-[splashscan_1.4s_ease-in-out_infinite_alternate] bg-[rgb(45,212,191)] shadow-[0_0_18px_4px_rgba(45,212,191,0.7)]" />
        )}
        <div className="absolute inset-0 rounded-full shadow-[inset_0_0_40px_10px_rgba(0,0,0,0.85)]" />
        {/* verified badge */}
        <div
          className={cx(
            'absolute inset-0 grid place-items-center bg-emerald-500/20 backdrop-blur-[1px] transition-all duration-500',
            verified ? 'scale-100 opacity-100' : 'scale-75 opacity-0',
          )}
        >
          <span className="grid h-16 w-16 place-items-center rounded-full bg-emerald-400 text-[#04070d] shadow-[0_0_40px_rgba(52,211,153,0.9)]">
            <Check size={34} strokeWidth={3.5} />
          </span>
        </div>
      </div>
      {/* finale shockwave */}
      {finale && (
        <>
          <span className="pointer-events-none absolute inset-0 animate-[shock_1.1s_cubic-bezier(.16,1,.3,1)_forwards] rounded-full border-2 border-emerald-300/80" />
          <span className="pointer-events-none absolute inset-0 animate-[shock_1.1s_.15s_cubic-bezier(.16,1,.3,1)_forwards] rounded-full border border-teal-300/50" />
        </>
      )}
    </div>
  )
}

export default function Splash({ onDone }) {
  const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  const [done, setDone] = useState(0)
  const [finale, setFinale] = useState(false)
  const [leaving, setLeaving] = useState(false)
  const [pct, setPct] = useState(0)
  const t0 = useRef(performance.now())

  useEffect(() => {
    const k = reduced ? 0.25 : 1
    t0.current = performance.now() // same clock as the step timers below
    const timers = STEPS.map((_, i) => setTimeout(() => setDone(i + 1), (START + STEP_MS * i) * k))
    timers.push(setTimeout(() => setFinale(true), FINALE * k))
    timers.push(setTimeout(() => setLeaving(true), LEAVE * k))
    let raf
    // 25% per step, filling smoothly towards the next step so the counter never runs ahead of the track
    const bounds = [0, ...STEPS.map((_, i) => (START + STEP_MS * i) * k)]
    const tick = () => {
      const e = performance.now() - t0.current
      const i = bounds.findIndex((b, j) => j < bounds.length - 1 && e < bounds[j + 1])
      const v = i === -1 ? 100 : 25 * (i + (e - bounds[i]) / (bounds[i + 1] - bounds[i]))
      setPct(Math.min(100, Math.round(v)))
      if (i !== -1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => (timers.forEach(clearTimeout), cancelAnimationFrame(raf))
  }, [reduced])

  // CSS fade/zoom out, then unmount (no JS exit animation that could stall)
  useEffect(() => {
    if (!leaving) return
    const t = setTimeout(onDone, 800)
    return () => clearTimeout(t)
  }, [leaving, onDone])

  useEffect(() => {
    const skip = () => setLeaving(true)
    window.addEventListener('keydown', skip)
    return () => window.removeEventListener('keydown', skip)
  }, [])

  const title = 'ReXTrust'
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label="Loading ReXTrust"
      onClick={() => setLeaving(true)}
      className={cx(
        'fixed inset-0 z-[200] cursor-pointer overflow-hidden text-white transition-all duration-700 ease-[cubic-bezier(.7,0,.84,0)]',
        leaving ? 'pointer-events-none scale-110 opacity-0 blur-sm' : 'scale-100 opacity-100',
      )}
      style={{ background: 'radial-gradient(ellipse at 50% 40%, #0b1a24 0%, #04070d 60%, #020306 100%)' }}
    >
      <NeuralField reduced={reduced} />
      {/* aurora */}
      <div className="pointer-events-none absolute -left-40 top-1/4 h-[30rem] w-[30rem] animate-[aurora_9s_ease-in-out_infinite_alternate] rounded-full bg-teal-400/20 blur-[120px]" />
      <div className="pointer-events-none absolute -right-40 bottom-0 h-[28rem] w-[28rem] animate-[aurora_11s_ease-in-out_infinite_alternate-reverse] rounded-full bg-indigo-400/20 blur-[120px]" />
      {/* perspective grid floor */}
      <div
        className="pointer-events-none absolute inset-x-[-50%] bottom-[-10%] h-[45%] opacity-40 [mask-image:linear-gradient(to_top,black,transparent)]"
        style={{
          transform: 'perspective(600px) rotateX(62deg)',
          backgroundImage: 'linear-gradient(rgba(45,212,191,.35) 1px, transparent 1px), linear-gradient(90deg, rgba(45,212,191,.35) 1px, transparent 1px)',
          backgroundSize: '56px 56px',
          animation: reduced ? undefined : 'gridmove 2.4s linear infinite',
        }}
      />
      {/* corner HUD */}
      <div className="pointer-events-none absolute left-5 top-5 font-mono text-[10px] uppercase tracking-[0.25em] text-white/40 sm:left-8 sm:top-8">
        Team 95 · XAI · Chest X-ray
      </div>
      <div className="pointer-events-none absolute right-5 top-5 hidden font-mono text-[10px] uppercase tracking-[0.25em] text-white/40 sm:right-8 sm:top-8 sm:block">
        {done >= 4 ? 'status · verified' : 'status · booting'}
      </div>

      <div className="relative flex h-full flex-col items-center justify-center px-6">
        <Core done={done} finale={finale} />

        {/* title */}
        <h1 className="mt-10 flex font-display text-5xl font-bold tracking-tight sm:text-7xl" aria-label={title}>
          {title.split('').map((ch, i) => (
            <motion.span
              key={i}
              initial={{ opacity: 0, y: 28, filter: 'blur(10px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              transition={{ delay: 0.25 + i * 0.06, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              className={i >= 3 ? 'bg-gradient-to-br from-teal-300 to-indigo-400 bg-clip-text text-transparent' : ''}
              aria-hidden
            >
              {ch}
            </motion.span>
          ))}
        </h1>
        <motion.p
          initial={{ opacity: 0, letterSpacing: '0.6em' }}
          animate={{ opacity: 1, letterSpacing: '0.28em' }}
          transition={{ delay: 0.8, duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
          className="mt-3 text-center font-mono text-[10px] uppercase text-white/55 sm:text-xs"
        >
          Trustworthy chest X-ray AI
        </motion.p>

        {/* pipeline track */}
        <div className="mt-10 w-full max-w-2xl">
          <div className="relative flex items-start justify-between">
            <div className="absolute left-[12.5%] right-[12.5%] top-[15px] h-px bg-white/10" />
            <div
              className="absolute left-[12.5%] top-[15px] h-px bg-gradient-to-r from-teal-300 to-indigo-400 shadow-[0_0_10px_rgba(45,212,191,0.9)] transition-[width] duration-700 ease-out"
              style={{ width: `${(Math.max(done - 1, 0) / (STEPS.length - 1)) * 75}%` }}
            />
            {STEPS.map((s, i) => {
              const ok = i < done
              const run = i === done
              return (
                <div key={s.label} className="relative z-10 flex w-1/4 flex-col items-center text-center">
                  <span
                    className={cx(
                      'grid h-8 w-8 place-items-center rounded-full border text-[11px] font-bold transition-all duration-500',
                      ok
                        ? 'scale-110 border-teal-300 bg-teal-300 text-[#04070d] shadow-[0_0_20px_rgba(45,212,191,0.8)]'
                        : run
                          ? 'animate-pulse border-teal-300/70 bg-[#04070d] text-teal-300'
                          : 'border-white/15 bg-[#04070d] text-white/30',
                    )}
                  >
                    {ok ? <Check size={15} strokeWidth={3} /> : i + 1}
                  </span>
                  <span className={cx('mt-2 font-mono text-[10px] uppercase tracking-wider transition-colors duration-500 sm:hidden', ok ? 'text-white' : 'text-white/35')}>{s.short}</span>
                  <span className={cx('mt-2 hidden text-xs font-medium transition-colors duration-500 sm:block', ok ? 'text-white' : 'text-white/35')}>{s.label}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* counter */}
        <div className="mt-10 flex items-baseline gap-3 font-mono">
          <span className="text-3xl font-semibold tabular-nums text-white sm:text-4xl">{String(pct).padStart(3, '0')}</span>
          <span className="text-xs uppercase tracking-[0.3em] text-teal-300/80">{done >= 4 ? 'System ready' : 'Initialising'}</span>
        </div>
      </div>

      <p className="pointer-events-none absolute inset-x-0 bottom-6 text-center font-mono text-[10px] uppercase tracking-[0.3em] text-white/30">
        Click or press any key to skip
      </p>
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
