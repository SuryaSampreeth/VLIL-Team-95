# ReXTrust: Project Showcase and Live Demo

**ReXTrust: XAI-Driven Multi-Disease Diagnosis in Chest X-rays via Conditional Enhancement and Grounded Reporting**
(Team 95, Type 1 Research Project. Guide: Dr. Anupa Vijai)

This folder contains two things:

- a **project portal**: objectives, problem statement, an interactive methodology diagram, results, novelty, team, progress, documents, GitHub and video;
- a **working demo** of the three-stage pipeline:

| Page | What it does |
|---|---|
| `/enhance` | **Stage 1**: quality gate, then one of 7 enhancement models. Before/after slider and BRISQUE / Entropy / Sharpness / CII / EME. |
| `/detect` | **Stage 2**: one of 4 detectors. Gives per-disease probabilities and an independent **Score-CAM** heatmap for each disease, with an opacity slider and side-by-side or grid views. |
| `/report` | **Stage 3**: **MediVLM** writes the report and **RadCheck RAG** verifies it. Shows a claims table, verdict, confidence gauge, consistency with the detector, highlighted sentences and a PDF export. |
| `/pipeline` | All three stages in sequence, with an animated stepper, the intermediate outputs, a summary dashboard and a ZIP download of everything. |

> **For research and educational purposes only. Not for clinical use.**

```
interface/
├── backend/            FastAPI + model registry
│   ├── config.yaml     ← models, weight paths, metrics, mode (auto | real | mock)
│   ├── requirements.txt
│   └── app/
│       ├── main.py         routes + upload validation
│       ├── services.py     stage logic + per-stage auto fallback
│       ├── engine_real.py  runs the trained models via ../../pipeline_core.py
│       ├── engine_mock.py  placeholder outputs (numpy + OpenCV only)
│       ├── imaging.py      I/O, quality metrics, quality gate, report helpers
│       └── config.py       config loader / registry lookups
├── frontend/           React 18 + Vite + Tailwind + Framer Motion + Recharts
│   └── src/{pages,components,lib}
├── assets/             served as the frontend's public folder
│   ├── samples/        4 demo X-rays
│   ├── figures/        figures taken from the review PDFs
│   ├── hero/           real pipeline output (results/1) used in the hero
│   ├── team/           team photos (from the review deck)
│   ├── pdf/            TEAM_95_PR2.pdf, Review_2.pdf
│   └── mock/           real MediVLM + RadCheck outputs replayed in mock mode
└── docs/               API.md, MODELS.md, ARCHITECTURE.md
```

---

## 1. Run it

You need **Python 3.10+** and **Node 18+**. The backend expects to sit inside the project folder: `pipeline_core.py`, `radcheck.py`, `Models/` and `.env` should be two levels up, as they are now.

### Backend (port 8000)

```bash
cd interface/backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Frontend (port 5173)

```bash
cd interface/frontend
npm install
npm run dev          # → http://localhost:5173
```

The frontend calls `http://localhost:8000` by default. To use another address, copy `.env.example` to `.env` and set `VITE_API_URL`.

Production build: run `npm run build` (output goes to `frontend/dist/`), then `npm run preview`.

### Expected timings (real models, CPU)

| Stage | Time |
|---|---|
| Enhancement | ~0.1–5 s (the first call loads the weights) |
| Detection + Score-CAM | ~30–60 s (256 masked forward passes) |
| MediVLM + RadCheck | ~60–90 s (MediVLM loads ~3 GB once; RadCheck calls the LLM) |
| Complete pipeline | ~1.5–2.5 min |

A CUDA GPU is used automatically when one is available. To make Score-CAM faster, lower `stage2.scorecam_channels` in `config.yaml` (for example to 128).

---

## 2. Modes: real, mock and auto

Set the mode in `backend/config.yaml`. This is the one-line switch:

```yaml
mode: auto     # auto | real | mock
```

You can also override it without editing the file: `REXTRUST_MODE=mock uvicorn app.main:app --port 8000`.

| Mode | Behaviour |
|---|---|
| `auto` (default) | Runs the real models when their weights load. If a stage's weights are missing or the model fails, **that stage alone** falls back to mock output, and the response says why. |
| `real` | Real models only. Failures come back as HTTP 503 instead of mock output. |
| `mock` | Never imports torch or loads weights. Responses are instant, so the UI works on any laptop. |

**What mock mode returns:**
- **Stage 1**: an OpenCV CLAHE variant.
- **Stage 2**: deterministic probabilities, plus heatmaps built from anatomical priors.
- **Stage 3**: replays **real** MediVLM + RadCheck outputs saved from the sample images (`assets/mock/reports.json`).

Every mock response carries `"mode": "mock"`, and the UI shows a yellow **MOCK OUTPUT** badge, so placeholder numbers are never mistaken for model output.

**RadCheck** needs `GOOGLE_API_KEY`, set either in the environment or in the project's `.env`. Without it, the report is still generated and is marked *unverified*.

---

## 3. Plug in real weights / add a model

All models are entries in `backend/config.yaml`. Weight paths are relative to `pipeline_root` (the project folder).

**Swap weights for an existing model.** Point `weights:` at the new file:

```yaml
- id: chexgcn
  weights: Models/Disease Detection/best_chexgcn_20k.pth
  thresholds: [0.38, 0.37, 0.38]      # F1-optimal on your validation split
  auc: [...]; f1: [...]               # shown in the UI
```

**Add a model with an existing architecture.** Add a new list item with a new `id` and an `arch` that already exists:

- Stage 1 archs: `attn_unet | unet_v3 | zero_dce`, with `adapter: network` and `clahe: true|false`
- Stage 2 archs: `cnn | cnn_dropout | hybrid | chexgcn`

The model appears in every selector automatically.

**Add a new architecture.**
1. Add its `nn.Module` to `pipeline_core.py`.
2. Register it in the `archs` dict in `backend/app/engine_real.py` (one line).
3. Stage 2 models must expose a `cam_layer` property for Score-CAM.

**UniMIE** is listed (`adapter: unavailable`) with its benchmark numbers. Its diffusion weights aren't bundled, so it runs in mock mode only. Add an adapter in `engine_real.py` to enable it.

See [`docs/MODELS.md`](docs/MODELS.md) for details.

---

## 4. API

| Method | Path | Body (multipart) | Returns |
|---|---|---|---|
| GET | `/health` | | mode, device, weight availability, RadCheck key status |
| GET | `/models/{stage}` | | registry for stage `1` or `2` (stage `3` gives report-model info) |
| POST | `/stage1/enhance` | `image`, `model?`, `gate_mode=always\|auto` | original + enhanced image, quality gate verdict, metrics before/after |
| POST | `/stage2/detect` | `image`, `model?`, `heatmaps=true` | per-disease probability / threshold / AUC / F1 + heatmap and overlay images |
| POST | `/stage3/report` | `image`, `findings?` (JSON), `crosscheck_model?` | report (findings / impression), per-sentence status, RadCheck evaluation, detector consistency |
| POST | `/pipeline/run` | `image`, `stage1_model?`, `stage2_model?`, `gate_mode=auto` | `{stage1, stage2, stage3, modes, timing_ms}` |

**Uploads:**
- Accepted formats: `.png .jpg .jpeg .bmp .tif .tiff .dcm`.
- Maximum size: 10 MB.
- Errors: 415 for an unsupported type, 413 for an oversize file, 422 for a file that isn't an image.

Full schema: [`docs/API.md`](docs/API.md) or `/docs`.

---

## 5. Placeholders to fill in before presenting

| Placeholder | Where |
|---|---|
| `TODO_GITHUB_URL` | `frontend/src/lib/content.js` → `LINKS.github` |
| `TODO_DEMO_VIDEO_URL` | `LINKS.video`: a YouTube **embed** URL (`https://www.youtube.com/embed/<id>`) or a file such as `/video/demo.mp4` placed in `assets/video/` |
| `TODO_ROLE`, `TODO_LINKEDIN`, `TODO_GITHUB` | `TEAM` in `content.js` (one entry per member) |
| `TODO_NEXT_MILESTONE` | `TIMELINE` in `content.js`: the remaining milestones and dates (the PDFs give none) |
| `TODO_UNIMIE_WEIGHTS` | `backend/config.yaml` (optional) |

Everything else on the site (objectives, problem statement, models, metrics, results, team and guide) comes from `TEAM_95_PR2.pdf` and `Review_2.pdf`. Team photos come from the review deck.

---

## 6. Troubleshooting

- **The navbar shows "API offline".** Start the backend, and check that `VITE_API_URL` matches its address.
- **A stage shows the MOCK OUTPUT badge in auto mode.** The yellow notice and the backend log explain which weights or packages are missing.
- **"RadCheck could not run".** Set `GOOGLE_API_KEY` in `../.env`, then check your network connection and quota.
- **The first Stage 3 call is slow.** MediVLM (~3 GB) loads on the first use and is cached after that.
