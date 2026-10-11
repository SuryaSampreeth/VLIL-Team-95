import { MotionConfig } from 'framer-motion'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, Route, Routes, useLocation } from 'react-router-dom'
import { Toaster } from 'sonner'
import { BackToTop, CommandPalette, HelpDrawer, ScrollProgress, UiContext } from './components/extras'
import { Footer, Navbar } from './components/Layout'
import Splash from './components/Splash'
import { BackendContext, ThemeContext, useHealthState, useThemeState } from './lib/hooks'
import Detect from './pages/Detect'
import Enhance from './pages/Enhance'
import Home from './pages/Home'
import Pipeline from './pages/Pipeline'
import Report from './pages/Report'

const TITLES = { '/': 'Portal', '/enhance': 'Stage 1 · Enhancement', '/detect': 'Stage 2 · Detection', '/report': 'Stage 3 · Report', '/pipeline': 'Complete Pipeline' }

function NotFound() {
  return (
    <div className="mx-auto grid min-h-[70vh] max-w-xl place-items-center px-4 pt-24 text-center">
      <div>
        <p className="font-display text-7xl font-bold text-gradient">404</p>
        <p className="mt-3 text-muted">This page does not exist.</p>
        <Link to="/" className="btn-primary mt-6">Back to the portal</Link>
      </div>
    </div>
  )
}

export default function App() {
  const [theme, toggleTheme] = useThemeState()
  const backend = useHealthState()
  const loc = useLocation()
  const [helpOpen, setHelpOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [splash, setSplash] = useState(true) // App mounts once per page load, so this shows only when the site is opened
  const hideSplash = useCallback(() => setSplash(false), [])
  const openHelp = useCallback(() => setHelpOpen(true), [])
  const closeHelp = useCallback(() => setHelpOpen(false), [])
  const openPalette = useCallback(() => setPaletteOpen(true), [])
  const ui = useMemo(() => ({ openHelp, openPalette }), [openHelp, openPalette])

  useEffect(() => {
    document.title = `${TITLES[loc.pathname] || 'Not found'} · ReXTrust`
    if (!loc.hash) window.scrollTo({ top: 0 })
  }, [loc.pathname, loc.hash])

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((o) => !o)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <ThemeContext.Provider value={theme}>
      <BackendContext.Provider value={backend}>
        <UiContext.Provider value={ui}>
          <MotionConfig reducedMotion="user">
            {splash && <Splash onDone={hideSplash} />}
            <ScrollProgress />
            <Navbar theme={theme} toggleTheme={toggleTheme} />
            {/* Page change = plain CSS fade-in keyed on the path. No exit phase, so a route can never get stuck blank
                (the previous AnimatePresence mode="wait" could hang waiting for an interrupted exit animation). */}
            <main id="main" key={loc.pathname} className="page-enter min-h-screen">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/enhance" element={<Enhance />} />
                <Route path="/detect" element={<Detect />} />
                <Route path="/report" element={<Report />} />
                <Route path="/pipeline" element={<Pipeline />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>
            <Footer />
            <BackToTop />
            <HelpDrawer open={helpOpen} onClose={closeHelp} />
            <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} toggleTheme={toggleTheme} openHelp={openHelp} />
            <Toaster theme={theme} position="bottom-right" richColors closeButton toastOptions={{ className: 'font-sans' }} />
          </MotionConfig>
        </UiContext.Provider>
      </BackendContext.Provider>
    </ThemeContext.Provider>
  )
}
