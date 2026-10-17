"""Mock engine: realistic placeholder outputs with numpy + OpenCV only (no torch, no weights).

Every response is marked mode="mock" and the UI shows a MOCK badge, so placeholder numbers are never mistaken for
model output. Outputs are deterministic per (image, model) so re-running gives the same result.
"""
import hashlib
import json

import cv2
import numpy as np

from . import imaging as im
from .config import ASSETS_DIR

_REPORTS = json.loads((ASSETS_DIR / "mock" / "reports.json").read_text(encoding="utf-8"))["reports"]


def _seed(gray: np.ndarray, salt: str) -> np.random.Generator:
    h = hashlib.sha1(gray[::4, ::4].tobytes() + salt.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


class MockEngine:
    name = "mock"

    def __init__(self, settings):
        self.s = settings

    # Stage 1 — CLAHE stand-in, tuned a little per model so switching models visibly changes the output
    def enhance(self, gray256: np.ndarray, spec: dict) -> np.ndarray:
        rng = _seed(gray256, spec["id"])
        clip = {"attn_unet_clahe": 2.0, "unet_v3_clahe": 1.8, "clahe": 2.0}.get(spec["id"], 1.2)
        out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray256)
        if spec["id"] in ("attn_unet", "unet_v3", "unimie"):
            out = cv2.addWeighted(out, 0.6, gray256, 0.4, 0)
        if spec["id"] == "zero_dce":
            out = np.uint8(255 * (gray256 / 255.0) ** 0.8)
        if spec["id"] in ("attn_unet_clahe", "unet_v3_clahe"):  # light unsharp mask for the "learned" variants
            blur = cv2.GaussianBlur(out, (0, 0), 1.5 + rng.random())
            out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
        return out

    # Stage 2 — probabilities from the image hash, heatmaps from anatomical priors weighted by local intensity
    def detect(self, gray256: np.ndarray, spec: dict) -> dict:
        rng = _seed(gray256, spec["id"])
        probs = np.clip(rng.beta(1.6, 3.2, size=len(self.s.diseases)), 0.02, 0.97)
        return {d: float(p) for d, p in zip(self.s.diseases, probs)}

    def cams(self, gray256: np.ndarray, spec: dict) -> dict:
        n = 64
        yy, xx = np.mgrid[0:n, 0:n] / n
        g = lambda cy, cx, sy, sx: np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2)
        priors = {  # image right = patient left
            "Cardiomegaly": g(0.62, 0.56, 0.14, 0.16),
            "Edema": g(0.45, 0.32, 0.15, 0.1) + g(0.45, 0.68, 0.15, 0.1),
            "Pleural Effusion": g(0.80, 0.2, 0.09, 0.12) + g(0.80, 0.8, 0.09, 0.12),
        }
        tex = cv2.resize(255 - gray256, (n, n)).astype(np.float32) / 255
        rng = _seed(gray256, spec["id"] + "cam")
        out = {}
        for d in self.s.diseases:
            cam = priors.get(d, g(0.5, 0.5, 0.25, 0.25)) * (0.7 + 0.3 * tex) + 0.05 * rng.random((n, n))
            cam = cv2.GaussianBlur(cam.astype(np.float32), (0, 0), 3)
            out[d] = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return out

    # Stage 3 — replays a real MediVLM + RadCheck output saved from the sample images
    def report(self, gray256: np.ndarray) -> dict:
        pick = _REPORTS[int(_seed(gray256, "report").integers(len(_REPORTS)))]
        return {"text": pick["report"], "raw": None, "source": pick["source"]}

    def radcheck(self, report_text: str) -> dict:
        for r in _REPORTS:
            if r["report"] == report_text:
                return r["radcheck"]
        return _REPORTS[0]["radcheck"]

# ─── Configuration Defaults ──────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".dcm"]
REQUEST_TIMEOUT_SECONDS = 120
BATCH_SIZE = 8
NUM_WORKERS = 4

def validate_image_dimensions(width, height):
    MIN_DIM, MAX_DIM = 224, 4096
    return MIN_DIM <= width <= MAX_DIM and MIN_DIM <= height <= MAX_DIM

def sanitize_filename(name):
    import re
    return re.sub(r'[^\w\-.]', '_', name)

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Pleural Effusion", "No Finding"
]
SEVERITY_LEVELS = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
CONFIDENCE_THRESHOLD = 0.65

# ─── Logging Setup ──────────────────────────────────────────────
import logging
import sys

def get_logger(module_name):
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
