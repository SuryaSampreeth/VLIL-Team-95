"""Loads config.yaml and exposes the model registry (models per stage, looked up by id)."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent
INTERFACE_DIR = BACKEND_DIR.parent
ASSETS_DIR = INTERFACE_DIR / "assets"
MODES = ("auto", "real", "mock")


class Settings:
    def __init__(self, path: Path):
        self.raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.mode = os.environ.get("REXTRUST_MODE", self.raw.get("mode", "auto")).lower()
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.pipeline_root = (path.parent / self.raw.get("pipeline_root", "../..")).resolve()
        self.max_bytes = int(self.raw["upload"]["max_mb"] * 1024 * 1024)
        self.extensions = tuple(e.lower() for e in self.raw["upload"]["extensions"])
        self.diseases = list(self.raw["diseases"])
        self.stage1, self.stage2, self.stage3 = self.raw["stage1"], self.raw["stage2"], self.raw["stage3"]

    def weights_path(self, rel: str) -> Path:
        return self.pipeline_root / rel

    def models(self, stage: int) -> list:
        return {1: self.stage1, 2: self.stage2}[stage]["models"]

    def model(self, stage: int, model_id: str = None) -> dict:
        block = {1: self.stage1, 2: self.stage2}[stage]
        model_id = model_id or block["default"]
        for m in block["models"]:
            if m["id"] == model_id:
                return m
        raise KeyError(f"Unknown stage {stage} model {model_id!r}. "
                       f"Available: {[m['id'] for m in block['models']]}")

    def has_weights(self, spec: dict) -> bool:
        if spec.get("adapter") == "clahe":
            return True
        if spec.get("adapter") == "unavailable" or not spec.get("weights"):
            return False
        return self.weights_path(spec["weights"]).exists()

    def public_model(self, stage: int, spec: dict) -> dict:
        """Registry entry as returned by GET /models/{stage}."""
        out = {k: v for k, v in spec.items() if k not in ("weights", "arch", "adapter", "thresholds")}
        out["weights_file"] = Path(spec["weights"]).name if spec.get("weights") else None
        out["real_available"] = self.mode != "mock" and self.has_weights(spec)
        out["default"] = spec["id"] == {1: self.stage1, 2: self.stage2}[stage]["default"]
        if stage == 2:
            out["metrics"] = [{"disease": d, "auc": a, "f1": f, "threshold": t}
                              for d, a, f, t in zip(self.diseases, spec["auc"], spec["f1"], spec["thresholds"])]
            out.pop("auc"), out.pop("f1")
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings(Path(os.environ.get("REXTRUST_CONFIG", BACKEND_DIR / "config.yaml")))
