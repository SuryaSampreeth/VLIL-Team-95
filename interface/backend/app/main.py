"""ReXTrust API.

    cd interface/backend
    uvicorn app.main:app --port 8000          # docs at http://localhost:8000/docs
"""
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import imaging as im
from . import services
from .config import ASSETS_DIR, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
S = get_settings()


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=services.warmup, daemon=True).start()
    yield


app = FastAPI(title="ReXTrust API", version="1.0.0", lifespan=lifespan,
              description="Conditional enhancement → multi-disease detection with per-disease Score-CAM → "
                          "MediVLM report + RadCheck RAG verification. For research and educational purposes only.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("REXTRUST_CORS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


async def _read_image(image: UploadFile):
    name = image.filename or "upload"
    if Path(name).suffix.lower() not in S.extensions:
        raise HTTPException(415, f"Unsupported file type '{Path(name).suffix}'. Allowed: {', '.join(S.extensions)}")
    data = await image.read(S.max_bytes + 1)
    if len(data) > S.max_bytes:
        raise HTTPException(413, f"File is larger than {S.max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        return im.decode(data, name)
    except im.ImageError as e:
        raise HTTPException(422, str(e)) from e


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e
    except services.StageError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health():
    def stage_ok(stage):
        return {m["id"]: S.has_weights(m) for m in S.models(stage)}
    env_key = S.stage3["verifier"]["env_key"]
    dotenv = S.pipeline_root / ".env"
    key_set = bool(os.environ.get(env_key)) or (
        dotenv.exists() and f"{env_key}=" in dotenv.read_text(encoding="utf-8", errors="ignore"))
    return {
        "status": "ok", "mode": S.mode, "device": services.REAL.device(),
        "pipeline_root": str(S.pipeline_root),
        "weights": {"stage1": stage_ok(1), "stage2": stage_ok(2),
                    "stage3": S.weights_path(S.stage3["report_model"]["weights"]).exists()},
        "radcheck_key_configured": key_set,
        "upload": {"max_mb": S.max_bytes // (1024 * 1024), "extensions": list(S.extensions)},
    }


@app.get("/models/{stage}")
def list_models(stage: int):
    if stage in (1, 2):
        block = S.stage1 if stage == 1 else S.stage2
        return {"stage": stage, "default": block["default"], "mode": S.mode,
                "models": [S.public_model(stage, m) for m in block["models"]]}
    if stage == 3:
        return {"stage": 3, "mode": S.mode, "report_model": S.stage3["report_model"]["name"],
                "verifier": S.stage3["verifier"]["name"], "diseases": S.diseases}
    raise HTTPException(404, "Stage must be 1, 2 or 3.")


@app.get("/samples")
def samples():
    files = sorted(p.name for p in (ASSETS_DIR / "samples").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return [{"name": f, "url": f"/assets/samples/{f}"} for f in files]


@app.post("/stage1/enhance")
async def stage1_enhance(image: UploadFile = File(...), model: Optional[str] = Form(None),
                         gate_mode: str = Form("always")):
    """gate_mode: 'always' = enhance with the chosen model; 'auto' = enhance only if the quality gate fails."""
    if gate_mode not in ("always", "auto"):
        raise HTTPException(422, "gate_mode must be 'always' or 'auto'.")
    gray = await _read_image(image)
    out = _run(services.stage1, gray, model, gate_mode)
    out.pop("_enhanced", None)
    return out


@app.post("/stage2/detect")
async def stage2_detect(image: UploadFile = File(...), model: Optional[str] = Form(None),
                        heatmaps: bool = Form(True)):
    gray = await _read_image(image)
    return _run(services.stage2, gray, model, heatmaps)


@app.post("/stage3/report")
async def stage3_report(image: UploadFile = File(...), findings: Optional[str] = Form(None),
                        crosscheck_model: Optional[str] = Form(None)):
    """findings: optional JSON {"positive": [...], "probabilities": {...}, "model": "..."} from Stage 2.
    crosscheck_model: if no findings are given, run this Stage 2 model to cross-check the report."""
    parsed = None
    if findings:
        try:
            parsed = json.loads(findings)
        except json.JSONDecodeError as e:
            raise HTTPException(422, f"findings is not valid JSON: {e}") from e
    gray = await _read_image(image)
    return _run(services.stage3, gray, parsed, crosscheck_model or None)


@app.post("/pipeline/run")
async def pipeline_run(image: UploadFile = File(...), stage1_model: Optional[str] = Form(None),
                       stage2_model: Optional[str] = Form(None), gate_mode: str = Form("auto")):
    if gate_mode not in ("always", "auto"):
        raise HTTPException(422, "gate_mode must be 'always' or 'auto'.")
    gray = await _read_image(image)
    return _run(services.pipeline, gray, stage1_model, stage2_model, gate_mode)
