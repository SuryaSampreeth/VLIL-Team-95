# Model registry

Every model the UI offers is defined in `backend/config.yaml`. `GET /models/{stage}` serves that list, and the selectors are built from it. Nothing about the models is hard-coded in the frontend.

## Stage 1: Enhancement

| id | Name | Adapter / arch | Weights | Real? |
|---|---|---|---|---|
| `attn_unet_clahe` | Attention U-Net + CLAHE (**ours**, default) | network / `attn_unet`, CLAHE | `Models/Enhancement/unet_attention_v4 .pt` | ✅ |
| `unet_v3_clahe` | U-Net V3 + CLAHE | network / `unet_v3`, CLAHE | `Models/Enhancement/unet_xray_v3.pt` | ✅ |
| `clahe` | CLAHE | clahe (no weights) | – | ✅ |
| `attn_unet` | Attention U-Net + CBAM | network / `attn_unet` | same as above | ✅ |
| `zero_dce` | Zero-DCE | network / `zero_dce` | `Models/Enhancement/zerodce_xray.pt` | ✅ |
| `unet_v3` | U-Net V3 | network / `unet_v3` | `Models/Enhancement/unet_xray_v3.pt` | ✅ |
| `unimie` | UniMIE | unavailable | `TODO_UNIMIE_WEIGHTS` | mock only |

**Quality gate.** The gate is set by `stage1.quality_gate`: BRISQUE above 5.0 or entropy below 7.95 means the image needs enhancement. These thresholds match `pipeline_core.py`.

**The `benchmark` block.** Each model's `benchmark` holds the mean over 23 low-quality CheXpert images from TEAM_95_PR2, slide 9.

**Live metrics.** The metrics shown for an uploaded image are computed live:
- **BRISQUE** uses piq, so it is real mode only.
- **Entropy** is the Shannon entropy of the histogram.
- **Sharpness** is the variance of the Laplacian.
- **CII** is the mean 3×3 Michelson contrast of the processed image divided by that of the original.
- **EME** is the mean over 8×8 blocks of 20·log₁₀(max/min).

## Stage 2: Detection

| id | Name | arch | Weights |
|---|---|---|---|
| `cnn_baseline` | CNN — Baseline Recipe (default) | `cnn` | `best_cnn.pth` |
| `cnn_matched` | CNN — Matched Recipe | `cnn_dropout` | `best_cnn_matched.pth` |
| `hybrid` | Hybrid CNN-Transformer | `hybrid` | `best_hybrid.pth` |
| `chexgcn` | CheXGCN | `chexgcn` | `best_chexgcn.pth` |

Each model also has the following in `config.yaml`:
- `thresholds`: F1-optimal on the validation split, applied to decide which predictions count as positive;
- `auc` / `f1`: held-out test set at 6k images per disease.

**Score-CAM** runs on `model.cam_layer` (DenseNet121 `denseblock4`), using `stage2.scorecam_channels` masks. Each disease's map uses that disease's own class score.

## Stage 3: Report

- `report_model.weights` is the MediVLM checkpoint (`Models/Report/mimic_sample.ckpt`). `pipeline_core.py` clones the MediVLM repo on first use if it is missing.
- `verifier.env_key` is the environment variable that holds the Gemini key used by RadCheck (`radcheck.py`).

## Adding a model

```yaml
stage2:
  models:
    - id: chexgcn_20k
      name: CheXGCN (20k)
      family: Graph neural network
      arch: chexgcn
      weights: Models/Disease Detection/best_chexgcn_20k.pth
      description: CheXGCN retrained at 20k images per disease.
      thresholds: [0.38, 0.37, 0.38]
      auc: [0.85, 0.81, 0.85]
      f1: [0.66, 0.70, 0.77]
```

Restart the backend and the new model appears in every selector. To add a **new architecture**:
1. Add its class to `pipeline_core.py`.
2. Add one line to the `archs` dict in `backend/app/engine_real.py`.

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
