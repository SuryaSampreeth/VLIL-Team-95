# ReXTrust API

Base URL: `http://localhost:8000`. The interactive OpenAPI docs are at `/docs`.

Every image field in a response is a `data:image/...;base64` URL. Every stage response contains:
- `mode`: `"real"` or `"mock"`;
- `notice`: why a stage fell back to mock, or `null`;
- `timing_ms`.

## GET /health
```json
{ "status": "ok", "mode": "auto", "device": "cpu",
  "weights": { "stage1": {"attn_unet_clahe": true, "unimie": false, ...}, "stage2": {...}, "stage3": true },
  "radcheck_key_configured": true, "upload": { "max_mb": 10, "extensions": [".png", ...] } }
```

## GET /models/{stage}
`stage` = `1` or `2`.
```json
{ "stage": 2, "default": "cnn_baseline", "mode": "auto",
  "models": [{ "id": "chexgcn", "name": "CheXGCN", "family": "Graph neural network",
               "description": "...", "weights_file": "best_chexgcn.pth", "real_available": true, "default": false,
               "metrics": [{ "disease": "Cardiomegaly", "auc": 0.829, "f1": 0.667, "threshold": 0.382156 }, ...] }] }
```
Stage 1 entries carry `benchmark` (`brisque`, `entropy`, `sharpness`, `cii`, `avg_rank`) and `proposed`.

## POST /stage1/enhance
Form fields:
- `image` (file);
- `model` (id, optional);
- `gate_mode`: `always` (default) or `auto` (enhance only if the quality gate fails).

```json
{ "stage": 1, "mode": "real", "model": {"id": "attn_unet_clahe", "name": "Attention U-Net + CLAHE"},
  "gate": { "needs_enhancement": true, "reason": "BRISQUE 28.66 > 5.0; Entropy 6.80 < 7.95",
            "thresholds": {"brisque_max": 5.0, "entropy_min": 7.95}, "mode": "always" },
  "was_enhanced": true, "original": "data:image/png;base64,...", "enhanced": "data:image/png;base64,...",
  "metrics": { "before": {"brisque": 28.66, "entropy": 6.805, "sharpness": 916.9, "cii": 1.0, "eme": 25.6},
               "after":  {"brisque": 17.34, "entropy": 7.646, "sharpness": 2604.1, "cii": 1.687, "eme": 31.13} },
  "benchmark": {...}, "size": [512, 512], "timing_ms": 312 }
```

## POST /stage2/detect
Form fields: `image`, `model` (optional), `heatmaps` (default `true`).
```json
{ "stage": 2, "mode": "real", "model": {...}, "explainer": "Score-CAM (denseblock4, one map per disease)",
  "base_image": "data:image/jpeg;base64,...",
  "diseases": [{ "name": "Cardiomegaly", "probability": 0.2034, "threshold": 0.382156, "positive": false,
                 "auc": 0.829, "f1": 0.667, "heatmap": "data:...", "overlay": "data:..." }, ...],
  "positive": [], "predicted_class": "No finding (of the 3 target diseases)",
  "top_disease": "Edema", "top_probability": 0.2902, "timing_ms": 43549 }
```
`heatmap` is the JET-coloured Score-CAM map, and `overlay` is the same map blended 50/50 over the X-ray.

## POST /stage3/report
Form fields:
- `image` (file);
- `findings`: optional JSON, `{"positive": ["Edema"], "probabilities": {...}, "model": "..."}`;
- `crosscheck_model`: optional Stage 2 id. If no `findings` are given, this detector runs and the report is cross-checked against its predictions.

```json
{ "stage": 3, "mode": "real", "model": {"id": "medivlm", "name": "MediVLM (GPT-2 decoder)"},
  "report": { "text": "FINDINGS: ...\nIMPRESSION: ...", "findings": "...", "impression": "...", "raw": "..." },
  "sentences": [{ "section": "findings", "text": "The heart size is normal.", "status": "verified", "claims": ["heart size"] }],
  "evaluation": { "available": true, "verifier": "RadCheck RAG", "overall_confidence": 1.0,
                  "hallucination_detected": false, "fully_verified": true,
                  "counts": {"claims": 7, "verified": 7, "flagged": 0, "typical": 7, "atypical": 0, "pubmed_grounded": 0, "contradicted": 0},
                  "claims": [{ "finding": "pleural effusion", "target": "pleura", "status": "absent",
                               "baseline_classification": "typical", "pubmed_grounded": false,
                               "is_contradicted": false, "grounding_score": 1.0, "hallucination_risk": false, ... }],
                  "contradictions": [], "graph_stats": {"num_entities": 15, "num_relations": 9} },
  "consistency": { "detector": {"source": "CNN — Baseline Recipe", "positive": [], "probabilities": {...}},
                   "per_disease": {"Pleural Effusion": {"report": "absent", "detector": "negative", "agreement": "agree"}, ...},
                   "agreement_rate": 1.0 },
  "input_image": "data:...", "timing_ms": 81234 }
```

**Sentence `status` values:**
- `verified`: every claim in the sentence passed RadCheck;
- `unsupported`: hallucination risk;
- `contradicted`: the MKG contradiction engine found a conflict;
- `unchecked`: no claim was extracted from the sentence.

If RadCheck can't run (no key, quota, or network), `evaluation` is `{"available": false, "reason": "..."}` and the report is still returned.

## POST /pipeline/run
Form fields: `image`, `stage1_model`, `stage2_model`, `gate_mode` (default `auto`).

Response: `{ "stage1": {...}, "stage2": {...}, "stage3": {...}, "modes": {...}, "timing_ms": 99993 }`.

Stage 2 runs on the Stage 1 output, and Stage 3 receives the Stage 2 findings.

## Errors
| Code | When |
|---|---|
| 404 | Unknown model id or stage |
| 413 | Upload larger than `upload.max_mb` |
| 415 | Unsupported extension |
| 422 | File is not a decodable image, image is smaller than 64×64, or a parameter is invalid |
| 503 | `mode: real` and the model failed (in `auto` mode the stage falls back to mock instead) |

---

## Development Setup

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm
- CUDA-capable GPU (optional, for real inference)

### Quick Start
1. Clone: `git clone https://github.com/SuryaSampreeth/VLIL-Team-95.git`
2. Backend: `cd interface/backend && pip install -r requirements.txt && uvicorn app.main:app --reload`
3. Frontend: `cd interface/frontend && npm install && npm run dev`
4. Open http://localhost:5173

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| VITE_API_URL | Backend API URL | http://localhost:8000 |
| VLIL_MODE | mock or real | mock |
| VLIL_MODEL_DIR | Model weights path | ./models |
