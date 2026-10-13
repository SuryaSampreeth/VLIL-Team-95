import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { api } from './api'

// ── theme ────────────────────────────────────────────────────────────────────
export function useThemeState() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('rx-theme') || 'dark'
    } catch {
      return 'dark'
    }
  })
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('rx-theme', theme)
    } catch {
      /* private mode */
    }
  }, [theme])
  return [theme, () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))]
}

// ── backend health (polled) ──────────────────────────────────────────────────
export const BackendContext = createContext({ health: null, status: 'checking', refresh: () => {} })
export const useBackend = () => useContext(BackendContext)

export function useHealthState() {
  const [health, setHealth] = useState(null)
  const [status, setStatus] = useState('checking')
  const refresh = useCallback(async () => {
    try {
      const h = await api.health()
      setHealth(h)
      setStatus('online')
    } catch {
      setStatus('offline')
    }
  }, [])
  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 15000)
    return () => clearInterval(id)
  }, [refresh])
  return { health, status, refresh }
}

// ── model registry ───────────────────────────────────────────────────────────
const modelCache = {}
export function useModels(stage) {
  const { status } = useBackend()
  const [state, setState] = useState(() => modelCache[stage] || { models: [], default: null, loading: true })
  useEffect(() => {
    if (status !== 'online' || modelCache[stage]) return
    const ctl = new AbortController()
    api
      .models(stage, ctl.signal)
      .then((r) => {
        modelCache[stage] = { models: r.models, default: r.default, loading: false }
        setState(modelCache[stage])
      })
      .catch((e) => e.name !== 'AbortError' && setState((s) => ({ ...s, loading: false, error: e.message })))
    return () => ctl.abort()
  }, [stage, status])
  return status === 'offline' && !modelCache[stage] ? { ...state, loading: false, error: 'offline' } : state
}

// ── elapsed-time ticker for long model calls ─────────────────────────────────
export function useElapsed(running) {
  const [ms, setMs] = useState(0)
  const start = useRef(0)
  useEffect(() => {
    if (!running) return
    start.current = performance.now()
    setMs(0)
    const id = setInterval(() => setMs(performance.now() - start.current), 100)
    return () => clearInterval(id)
  }, [running])
  return ms
}

// ── theme context + chart colours (reference dataviz palette, slots 1-4, stepped per theme) ──
export const ThemeContext = createContext('dark')
export const useTheme = () => useContext(ThemeContext)

const CHART = {
  light: { series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'], grid: '#e2e8f0', axis: '#475569', muted: '#cbd5e1', surface: '#ffffff', text: '#0b1324' },
  dark: { series: ['#3987e5', '#d95926', '#199e70', '#c98500'], grid: '#1e293b', axis: '#94a3b8', muted: '#334155', surface: '#0c1322', text: '#e6edf7' },
}
export const useChartColors = () => CHART[useTheme()] || CHART.dark
