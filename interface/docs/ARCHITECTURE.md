# Interface architecture

```
Browser (React SPA)                       FastAPI (backend/app)                     Project root
──────────────────                        ─────────────────────                     ────────────
/enhance  ──POST /stage1/enhance──►  main.py ─► services.stage1 ─┐
/detect   ──POST /stage2/detect───►           ─► services.stage2 ─┼─► engine_real ──► pipeline_core.py
/report   ──POST /stage3/report───►           ─► services.stage3 ─┤        │            (AttnUNet, UNetV3, ZeroDCE,
/pipeline ──stage1 → stage2 → stage3 (sequential, live stepper)   │        │             CNN, Hybrid, CheXGCN,
           or POST /pipeline/run                                  │        │             MediVLM, RadCheck)
                                                                  └─► engine_mock (numpy + OpenCV)
                                    config.yaml ── model registry, mode, thresholds, metrics
```

**Engine selection.** `services._call` picks the engine for each stage:
- `mock` mode always uses the mock engine.
- `real` mode always uses the real engine, and an error comes back as HTTP 503.
- `auto` mode uses the real engine when the weights exist, and falls back to mock if the call fails. The response then carries `mode: "mock"` and a `notice` explaining why.

**Complete-pipeline page.** The frontend calls the three stage endpoints in sequence, so each intermediate result appears as soon as it is ready:
1. Stage 2 receives the Stage 1 enhanced PNG.
2. Stage 3 receives the same image plus the Stage 2 findings as JSON.

`/pipeline/run` does the same in a single request, for scripted use.

**Concurrency.** The real engine holds one re-entrant lock around model calls, and every call runs under `torch.no_grad()`. Gradient mode in PyTorch is thread-local, and FastAPI runs sync endpoints on worker threads, so `pipeline_core`'s global `set_grad_enabled(False)` doesn't carry over to those threads.

**Frontend layout:**
- `src/lib/content.js`: all project text and numbers taken from the PDFs. Edit this file to change site copy.
- `src/lib/api.js`: API client.
- `src/components/ui.jsx`: design primitives such as the glass card, tabs, accordion, gauge, counters, skeletons and figure lightbox.
- `src/components/demo.jsx`: uploader, model picker, comparison slider, heatmap viewer, report and evaluation views.
- `src/components/portal.jsx`: hero, pipeline diagram, timeline and team cards.
- `src/components/charts.jsx`: Recharts views of the PDF results. Colours come from the validated categorical palette, with separate light and dark steps.
