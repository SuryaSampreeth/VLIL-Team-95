"""Real engine: runs the trained models through the project's pipeline_core.py.

Networks are built from the architecture classes in pipeline_core and the weight paths in config.yaml, then cached.
Adding a model = one config entry (plus one line in ARCHS if it is a new architecture).
"""
import contextlib
import io
import os
import sys
import threading

import numpy as np


class RealEngine:
    name = "real"

    def __init__(self, settings):
        self.s = settings
        self._pc = None
        self._cache = {}
        self.lock = threading.RLock()  # one model call at a time; torch on CPU gains nothing from overlap

    # ── lazy import of the project's pipeline ────────────────────────────────
    @property
    def pc(self):
        if self._pc is None:
            with self.lock:
                if self._pc is None:
                    root = str(self.s.pipeline_root)
                    if root not in sys.path:
                        sys.path.insert(0, root)
                    with contextlib.redirect_stdout(io.StringIO()):
                        import pipeline_core
                    self._pc = pipeline_core
        return self._pc

    def device(self) -> str:
        return str(self._pc.DEVICE) if self._pc else "not loaded"

    @contextlib.contextmanager
    def _infer(self):
        """Serialise model calls and disable autograd. pipeline_core's set_grad_enabled(False) is thread-local,
        and FastAPI runs requests on worker threads, so it has to be set again here."""
        with self.lock, self.pc.torch.no_grad():
            yield

    def _cached(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def _network(self, spec: dict):
        pc = self.pc
        archs = {
            "attn_unet": pc.AttnUNet, "unet_v3": pc.UNetV3, "zero_dce": pc.ZeroDCE,
            "cnn": lambda: pc.CNNClassifier(), "cnn_dropout": lambda: pc.CNNClassifier(dropout=0.3),
            "hybrid": lambda: pc.HybridCNNTransformer(), "chexgcn": lambda: pc.CheXGCN(),
        }

        def build():
            path = self.s.weights_path(spec["weights"])
            if not path.exists():
                raise FileNotFoundError(f"weights not found: {path}")
            m = archs[spec["arch"]]()
            m.load_state_dict(pc.torch.load(path, map_location="cpu"), strict=True)
            return m.to(pc.DEVICE).eval()

        return self._cached((spec["arch"], spec["weights"]), build)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    def brisque_fn(self):
        pc = self.pc
        if pc.piq is None:
            return None
        return lambda g: pc.piq.brisque(pc.torch.from_numpy(g).float().div(255)[None, None], data_range=1.0)

    def enhance(self, gray256: np.ndarray, spec: dict) -> np.ndarray:
        if spec["adapter"] == "clahe":
            return self.pc.CLAHE.apply(gray256)
        if spec["adapter"] != "network":
            raise RuntimeError(f"{spec['name']} has no runnable weights in this project")
        pc = self.pc
        with self._infer():
            x = pc.torch.from_numpy(gray256).float().div(255)[None, None].repeat(1, 3, 1, 1).to(pc.DEVICE)
            out = (self._network(spec)(x).mean(1)[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return pc.CLAHE.apply(out) if spec.get("clahe") else out

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    def detect(self, gray256: np.ndarray, spec: dict) -> dict:
        pc = self.pc
        with self._infer():
            logits = self._network(spec)(pc.cnn_input(gray256))
            probs = pc.torch.sigmoid(logits)[0].cpu().numpy()
        return {d: float(p) for d, p in zip(self.s.diseases, probs)}

    def cams(self, gray256: np.ndarray, spec: dict) -> dict:
        """Score-CAM, one map per disease from that disease's own class score (pipeline_core.score_cam)."""
        pc, torch, F = self.pc, self.pc.torch, self.pc.F
        model = self._network(spec)
        k = int(self.s.stage2.get("scorecam_channels") or 0)
        with self._infer():
            x = pc.cnn_input(gray256)
            acts = {}
            hook = model.cam_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("A", o))
            try:
                model(x)
            finally:
                hook.remove()
            A = acts["A"][0]
            if k and k < A.shape[0]:
                A = A[A.flatten(1).var(1).topk(k).indices]
            up = F.interpolate(A[None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
            lo, hi = up.flatten(1).min(1)[0][:, None, None], up.flatten(1).max(1)[0][:, None, None]
            masks = (up - lo) / (hi - lo + 1e-8)
            scores = torch.cat([model(x * masks[i:i + 32, None]) for i in range(0, len(masks), 32)])
            weights = torch.softmax(scores, dim=0)
            cams = F.relu(torch.einsum("kc,khw->chw", weights, up))
            cams = cams - cams.flatten(1).min(1)[0][:, None, None]
            cams = cams / (cams.flatten(1).max(1)[0][:, None, None] + 1e-7)
        return {d: cams[i].cpu().numpy() for i, d in enumerate(self.s.diseases)}

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    def report(self, gray256: np.ndarray) -> dict:
        ckpt = self.s.weights_path(self.s.stage3["report_model"]["weights"])
        if not ckpt.exists():
            raise FileNotFoundError(f"MediVLM checkpoint not found: {ckpt}")
        with self._infer(), contextlib.redirect_stdout(io.StringIO()):
            r = self.pc.generate_report(gray256, num_beams=self.s.stage3["report_model"].get("num_beams", 4))
        return {"text": r["report"], "raw": r["raw"], "source": "MediVLM"}

    def radcheck(self, report_text: str):
        """Returns the RadCheck result, or raises with a readable reason (no key, network, quota)."""
        if not os.environ.get(self.s.stage3["verifier"]["env_key"]):
            self.pc  # importing pipeline_core loads keys from <pipeline_root>/.env
        if not os.environ.get(self.s.stage3["verifier"]["env_key"]):
            raise RuntimeError(f"{self.s.stage3['verifier']['env_key']} is not set, so RadCheck cannot call the LLM")
        result, _log = self.pc.run_radcheck(report_text)
        if result is None or "error" in result:
            raise RuntimeError((result or {}).get("error", "RadCheck returned no result"))
        return result
