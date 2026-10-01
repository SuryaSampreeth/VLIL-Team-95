"""
ReXTrust pipeline core — every model from full_pipeline.ipynb plus the alternative checkpoints in Models/,
selectable by name and lazily loaded (each model is built once, on first use, then cached).

  Enhancement (Models/Enhancement):        Attention U-Net v4 + CLAHE | U-Net V3 + CLAHE | Zero-DCE | None
  Disease detection (Models/Disease Detection, rextrust-stage2-scaled_6k.ipynb):
                                           CNN baseline | CNN matched | Hybrid CNN-Transformer | CheXGCN
  Report (Models/Report):                  MediVLM  ->  RadCheck (AMG-RAG, Gemma via GOOGLE_API_KEY)
"""
import os, sys, re, json, shutil, subprocess, contextlib, io, time, warnings
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)  # inference only

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "Models"
ENH_DIR = MODELS / "Enhancement"
DET_DIR = MODELS / "Disease Detection"
MEDIVLM_CKPT = MODELS / "Report" / "mimic_sample.ckpt"
MEDIVLM_REPO = MODELS / "Report" / "MediVLM"
MEDIVLM_URL, MEDIVLM_COMMIT = "https://github.com/sonai-commits/MediVLM.git", "f5a6e1dd0c4139f9fbd3cd075545a3b22a1a14b8"
RESULTS_DIR = ROOT / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISEASES = ["Cardiomegaly", "Edema", "Pleural Effusion"]
CNN_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BRISQUE_THRESHOLD, ENTROPY_THRESHOLD = 5.0, 7.95
DEFAULT_SCORECAM_CHANNELS = 1024 if DEVICE.type == "cuda" else 256

# API keys: environment first, then the project's .env file
if (ROOT / ".env").exists():
    from decouple import Config, RepositoryEnv
    _env = Config(RepositoryEnv(str(ROOT / ".env")))
    for _k in ("GOOGLE_API_KEY", "pubmed_api"):
        if not os.environ.get(_k) and _env(_k, default=""):
            os.environ[_k] = _env(_k)

_CACHE: dict = {}


def _cached(key, builder):
    if key not in _CACHE:
        _CACHE[key] = builder()
    return _CACHE[key]


def is_loaded(key) -> bool:
    return key in _CACHE


# ════════════════════════════════════════════════════════════════════════════
# Input + quality gate
# ════════════════════════════════════════════════════════════════════════════
def load_xray(path) -> np.ndarray:
    """Load a chest X-ray from JPEG/PNG/DICOM; returns a uint8 grayscale array.

    Supports 8-bit and 16-bit images; 16-bit images are min-max normalised to uint8.
    DICOM files are decoded with pydicom (install separately).
    """
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        import pydicom
        img = pydicom.dcmread(str(path)).pixel_array.astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:  # 16-bit PNG / DICOM -> min-max to uint8
        img = img.astype(np.float32)
        img = (img - img.min()) / max(float(img.max() - img.min()), 1e-6) * 255
        img = img.astype(np.uint8)
    return img


try:
    import piq
except ImportError:
    piq = None


def assess_quality(gray256: np.ndarray) -> dict:
    """Compute BRISQUE, Shannon entropy, and Laplacian sharpness for a grayscale image.

    Returns dict with keys: brisque, entropy, sharpness, needs_enhancement, reason.
    needs_enhancement is True when BRISQUE exceeds BRISQUE_THRESHOLD or entropy falls
    below ENTROPY_THRESHOLD.
    """
    hist = np.bincount(gray256.ravel(), minlength=256) / gray256.size
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
    brisque = float("nan")
    if piq is not None:
        brisque = float(piq.brisque(torch.from_numpy(gray256).float().div(255)[None, None], data_range=1.0))
    sharpness = float(cv2.Laplacian(gray256, cv2.CV_64F).var())

    reasons = []
    if not np.isnan(brisque) and brisque > BRISQUE_THRESHOLD:
        reasons.append(f"BRISQUE {brisque:.2f} > {BRISQUE_THRESHOLD}")
    if entropy < ENTROPY_THRESHOLD:
        reasons.append(f"entropy {entropy:.2f} < {ENTROPY_THRESHOLD}")
    return {"brisque": round(brisque, 2), "entropy": round(entropy, 2), "sharpness": round(sharpness, 1),
            "needs_enhancement": bool(reasons), "reason": "; ".join(reasons) or "passes quality gate"}


# ════════════════════════════════════════════════════════════════════════════
# Enhancement models
# ════════════════════════════════════════════════════════════════════════════
# --- Attention U-Net v4 (rebuilt from the checkpoint's 247 keys; see full_pipeline.ipynb) ---
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(c, c // r, bias=False), nn.ReLU(inplace=True),
                                 nn.Linear(c // r, c, bias=False))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a)[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca, self.sa = ChannelAttention(c), SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class AttentionGate(nn.Module):
    def __init__(self, cg, cx, ci):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(cg, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.W_x = nn.Sequential(nn.Conv2d(cx, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.psi = nn.Sequential(nn.Conv2d(ci, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * a, a


class AttnUNet(nn.Module):
    def __init__(self, cin=3, cout=3, f=(32, 64, 128, 256, 512)):
        super().__init__()
        self.enc1, self.enc2, self.enc3, self.enc4 = (ResBlock(cin, f[0]), ResBlock(f[0], f[1]),
                                                      ResBlock(f[1], f[2]), ResBlock(f[2], f[3]))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResBlock(f[3], f[4])
        self.cbam = CBAM(f[4])
        self.up4, self.up3, self.up2, self.up1 = [nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in (3, 2, 1, 0)]
        self.ag4, self.ag3, self.ag2, self.ag1 = [AttentionGate(f[i], f[i], f[i] // 2) for i in (3, 2, 1, 0)]
        self.dec4, self.dec3, self.dec2, self.dec1 = [ResBlock(f[i] * 2, f[i]) for i in (3, 2, 1, 0)]
        self.head = nn.Sequential(nn.Conv2d(f[0], cout, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d = self.cbam(self.bottleneck(self.pool(e4)))
        for i, skip in ((4, e4), (3, e3), (2, e2), (1, e1)):
            g = getattr(self, f"up{i}")(d)
            gated, _ = getattr(self, f"ag{i}")(g, skip)
            d = getattr(self, f"dec{i}")(torch.cat([g, gated], 1))
        return self.head(d)


# --- U-Net V3 (rebuilt from unet_xray_v3.pt: 3 levels, conv-BN-ReLU x2 blocks, 1x1 head) ---
def _double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNetV3(nn.Module):
    def __init__(self, cin=3, cout=3):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = _double_conv(cin, 32), _double_conv(32, 64), _double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, stride=2), _double_conv(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, stride=2), _double_conv(64, 32)
        self.final = nn.Conv2d(32, cout, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.final(d1))


# --- Zero-DCE (standard DCE-Net: 7 convs, 8 light-enhancement curve iterations) ---
class ZeroDCE(nn.Module):
    def __init__(self, nf=32):
        super().__init__()
        self.e_conv1 = nn.Conv2d(3, nf, 3, padding=1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, padding=1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, padding=1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for a in torch.split(r, 3, dim=1):
            x = x + a * (x * x - x)
        return x.clamp(0, 1)


def _attn_unet_path():
    return next(p for p in [ENH_DIR / "unet_attention_v4.pt", ENH_DIR / "unet_attention_v4 .pt"] if p.exists())


ENHANCERS = {
    "Attention U-Net v4 + CLAHE": dict(cls=AttnUNet, path=_attn_unet_path, clahe=True, tag="attn_unet_clahe",
                                       info="ResBlocks + CBAM + attention gates, 8.24 M params. Pipeline default; "
                                            "best BRISQUE (1.50) / sharpness in the README benchmark."),
    "U-Net V3 + CLAHE": dict(cls=UNetV3, path=lambda: ENH_DIR / "unet_xray_v3.pt", clahe=True, tag="unet_v3_clahe",
                             info="Plain 3-level U-Net (0.47 M params) + CLAHE. README benchmark: BRISQUE 22.89."),
    "Zero-DCE": dict(cls=ZeroDCE, path=lambda: ENH_DIR / "zerodce_xray.pt", clahe=False, tag="zero_dce",
                     info="Zero-reference curve estimation (79 K params), no CLAHE. README benchmark: BRISQUE 14.74."),
}
NO_ENHANCEMENT = "None (skip enhancement)"
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def get_enhancer(name: str) -> nn.Module:
    def build():
        spec = ENHANCERS[name]
        m = spec["cls"]()
        m.load_state_dict(torch.load(spec["path"](), map_location="cpu"), strict=True)
        return m.to(DEVICE).eval()
    return _cached(("enh", name), build)


def enhance_image(image_path, model_name: str = "Attention U-Net v4 + CLAHE", mode: str = "auto") -> dict:
    """mode: 'auto' = enhance only when the quality gate fails, 'always' = enhance every image."""
    original = load_xray(image_path)
    resized = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)
    q_before = assess_quality(resized)

    run = model_name != NO_ENHANCEMENT and (mode == "always" or q_before["needs_enhancement"])
    if run:
        spec = ENHANCERS[model_name]
        x = torch.from_numpy(resized).float().div(255)[None, None].repeat(1, 3, 1, 1).to(DEVICE)
        out = (get_enhancer(model_name)(x).mean(1)[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        enhanced, method = (CLAHE.apply(out) if spec["clahe"] else out), spec["tag"]
    else:
        enhanced, method = resized.copy(), "passthrough"

    return {"original": original, "original_resized": resized, "enhanced": enhanced, "model": model_name,
            "was_enhanced": method != "passthrough", "method": method,
            "quality_before": q_before, "quality_after": assess_quality(enhanced)}


# ════════════════════════════════════════════════════════════════════════════
# Disease detection models (rextrust-stage2-scaled_6k.ipynb, Techniques 1-4)
# ════════════════════════════════════════════════════════════════════════════
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.0):
        super().__init__()
        self.backbone = models.densenet121(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        feat = F.relu(self.backbone(x))
        return self.classifier(self.dropout(self.pool(feat).flatten(1)))

    @property
    def cam_layer(self):
        return self.backbone.denseblock4


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, num_transformer_layers=2, nhead=4):
        super().__init__()
        self.cnn_backbone = models.densenet121(weights=None).features
        d = 1024
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, 200, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=d * 2, dropout=dropout,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_classes)

    def forward(self, x):
        feat_map = F.relu(self.cnn_backbone(x))
        tokens = feat_map.flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), tokens], 1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        return self.classifier(self.dropout(self.transformer(tokens)[:, 0, :]))

    @property
    def cam_layer(self):
        return self.cnn_backbone.denseblock4

    def attention_rollout(self, x) -> np.ndarray:
        """CLS -> patch attention rollout (detection notebook, Stage 3 output (e))."""
        feat_map = F.relu(self.cnn_backbone(x))
        _, _, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)
        src = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), tokens], 1)
        src = src + self.pos_embed[:, :src.shape[1], :]
        attn = []
        for layer in self.transformer.layers:
            out, w = layer.self_attn(src, src, src, need_weights=True, average_attn_weights=True)
            src = layer.norm1(src + layer.dropout1(out))
            src = layer.norm2(src + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))))
            attn.append(w)
        eye = torch.eye(attn[0].shape[-1], device=x.device)[None]
        rollout = eye
        for A in attn:
            A = A + eye
            rollout = (A / A.sum(-1, keepdim=True)) @ rollout
        m = rollout[0, 0, 1:].reshape(H, W).cpu().numpy()
        return (m - m.min()) / (m.max() - m.min() + 1e-8)


class GraphConvolution(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x, adj):
        return adj @ self.weight(x)


# Symmetrized label co-occurrence from the detection notebook's train split (Stage 4 figure; the matrix itself
# was not saved with the weights, so these are the printed 2-decimal values).
CHEXGCN_COOCCURRENCE = np.array([[1.00, 0.46, 0.42],
                                 [0.46, 1.00, 0.56],
                                 [0.42, 0.56, 1.00]], dtype=np.float32)


def _normalized_adjacency(a: np.ndarray) -> torch.Tensor:
    a = a + np.eye(len(a), dtype=np.float32)
    d = np.diag(np.power(a.sum(1), -0.5))
    return torch.tensor(d @ a @ d, dtype=torch.float32)


class CheXGCN(nn.Module):
    def __init__(self, num_labels=3, feature_dim=1024, label_embed_dim=300, gcn_hidden=512, dropout=0.3):
        super().__init__()
        self.backbone = models.densenet121(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.label_embeddings = nn.Parameter(torch.zeros(num_labels, label_embed_dim))
        self.gcn1 = GraphConvolution(label_embed_dim, gcn_hidden)
        self.gcn2 = GraphConvolution(gcn_hidden, feature_dim)
        self.register_buffer("adj", _normalized_adjacency(CHEXGCN_COOCCURRENCE), persistent=False)

    def forward(self, x):  # single-argument forward (adjacency fixed), same as the notebook's CAM wrapper
        feat = self.dropout(self.pool(F.relu(self.backbone(x))).flatten(1))
        label_repr = self.gcn2(F.relu(self.gcn1(self.label_embeddings, self.adj)), self.adj)
        return feat @ label_repr.T

    @property
    def cam_layer(self):
        return self.backbone.denseblock4


# Thresholds = F1-optimal on the validation split; metrics = held-out test set (detection notebook, Stage 5).
DETECTORS = {
    "CNN — baseline (DenseNet121)": dict(
        cls=lambda: CNNClassifier(), file="best_cnn.pth", thresholds=[0.245157, 0.404027, 0.506576],
        auc=[0.832, 0.803, 0.841], f1=[0.668, 0.719, 0.776],
        info="Technique 1: literal CheXNet recipe (BCE, no dropout/scheduler). Pipeline default."),
    "CNN — matched recipe (DenseNet121)": dict(
        cls=lambda: CNNClassifier(dropout=0.3), file="best_cnn_matched.pth", thresholds=[0.377337, 0.362627, 0.339481],
        auc=[0.826, 0.801, 0.839], f1=[0.654, 0.720, 0.782],
        info="Technique 2: same architecture, Focal loss + dropout 0.3 + LR scheduler."),
    "Hybrid CNN-Transformer": dict(
        cls=lambda: HybridCNNTransformer(), file="best_hybrid.pth", thresholds=[0.349390, 0.331475, 0.323119],
        auc=[0.835, 0.790, 0.845], f1=[0.674, 0.706, 0.785],
        info="Technique 3: DenseNet121 tokens + [CLS] + 2-layer Transformer. Also gives an attention-rollout map."),
    "CheXGCN (label graph)": dict(
        cls=lambda: CheXGCN(), file="best_chexgcn.pth", thresholds=[0.382156, 0.377137, 0.378709],
        auc=[0.829, 0.804, 0.842], f1=[0.667, 0.721, 0.773],
        info="Technique 4: DenseNet121 features x GCN over the disease co-occurrence graph."),
}

_MEAN = torch.tensor(IMAGENET_MEAN, device=DEVICE)[:, None, None]
_STD = torch.tensor(IMAGENET_STD, device=DEVICE)[:, None, None]


def get_detector(name: str) -> nn.Module:
    def build():
        spec = DETECTORS[name]
        m = spec["cls"]()
        m.load_state_dict(torch.load(DET_DIR / spec["file"], map_location="cpu"), strict=True)
        return m.to(DEVICE).eval()
    return _cached(("det", name), build)


def cnn_input(enhanced: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(enhanced, (CNN_SIZE, CNN_SIZE)), cv2.COLOR_GRAY2RGB)
    x = torch.from_numpy(rgb).to(DEVICE).permute(2, 0, 1).float().div(255)
    return ((x - _MEAN) / _STD)[None]


def detect_disease(enhanced: np.ndarray, model_name: str) -> dict:
    spec = DETECTORS[model_name]
    probs = torch.sigmoid(get_detector(model_name)(cnn_input(enhanced)))[0].cpu().numpy()
    positive = [d for d, p, t in zip(DISEASES, probs, spec["thresholds"]) if p > t]
    top = int(probs.argmax())
    return {"model": model_name,
            "probabilities": {d: float(p) for d, p in zip(DISEASES, probs)},
            "thresholds": dict(zip(DISEASES, spec["thresholds"])),
            "positive": positive,
            "predicted_class": ", ".join(positive) if positive else "No finding (of the 3 target diseases)",
            "top_disease": DISEASES[top], "top_confidence": float(probs[top])}


# ════════════════════════════════════════════════════════════════════════════
# Explainability — Score-CAM (one map per disease), works for any detector above
# ════════════════════════════════════════════════════════════════════════════
def score_cam(enhanced: np.ndarray, model_name: str, max_channels=DEFAULT_SCORECAM_CHANNELS,
              batch_size=32, progress=None) -> dict:
    model = get_detector(model_name)
    x = cnn_input(enhanced)
    acts = {}
    hook = model.cam_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("A", o))
    try:
        model(x)
    finally:
        hook.remove()
    A = acts["A"][0]                                                    # (1024, 7, 7)
    if max_channels and max_channels < A.shape[0]:
        A = A[A.flatten(1).var(1).topk(int(max_channels)).indices]
    up = F.interpolate(A[None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
    lo, hi = up.flatten(1).min(1)[0][:, None, None], up.flatten(1).max(1)[0][:, None, None]
    masks = (up - lo) / (hi - lo + 1e-8)

    chunks = []
    for i in range(0, len(masks), batch_size):
        chunks.append(model(x * masks[i:i + batch_size, None]))
        if progress:
            progress(min(1.0, (i + batch_size) / len(masks)))
    weights = torch.softmax(torch.cat(chunks), dim=0)                     # (K, 3)
    cams = F.relu(torch.einsum("kc,khw->chw", weights, up))
    cams = cams - cams.flatten(1).min(1)[0][:, None, None]
    cams = cams / (cams.flatten(1).max(1)[0][:, None, None] + 1e-7)
    return {d: cams[i].cpu().numpy() for i, d in enumerate(DISEASES)}


def overlay(gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam = cv2.resize(cam, gray.shape[::-1])
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8(0.5 * heat + 0.5 * cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


# ════════════════════════════════════════════════════════════════════════════
# Report — MediVLM (see full_pipeline.ipynb for why the load is done this way)
# ════════════════════════════════════════════════════════════════════════════
def _ensure_medivlm_repo():
    if not (MEDIVLM_REPO / "medivlm").exists():
        subprocess.run(["git", "clone", "-q", MEDIVLM_URL, str(MEDIVLM_REPO)], check=True)
        subprocess.run(["git", "-C", str(MEDIVLM_REPO), "checkout", "-q", MEDIVLM_COMMIT], check=True)
    for d in MEDIVLM_REPO.rglob("__pycache__"):
        shutil.rmtree(d)
    if str(MEDIVLM_REPO) not in sys.path:
        sys.path.insert(0, str(MEDIVLM_REPO))


def _load_medivlm():
    _ensure_medivlm_repo()
    from unittest import mock
    from transformers import AutoConfig, AutoModel, CLIPVisionConfig, CLIPVisionModel, GPT2LMHeadModel
    from medivlm.models.medivlm import MediVLM
    from medivlm.models import detector as detector_module
    from medivlm.utils.config import MediVLMConfig, _update_dataclass
    from medivlm.data.transforms import build_image_transform

    ckpt = torch.load(MEDIVLM_CKPT, map_location="cpu", mmap=True, weights_only=False)
    cfg = _update_dataclass(MediVLMConfig(), ckpt["config"])
    patches = [
        mock.patch.object(CLIPVisionModel, "from_pretrained",
                          classmethod(lambda cls, name, *a, **k: cls._from_config(CLIPVisionConfig.from_pretrained(name)))),
        mock.patch.object(GPT2LMHeadModel, "from_pretrained",
                          classmethod(lambda cls, name, *a, **k: cls._from_config(AutoConfig.from_pretrained(name)))),
        mock.patch.object(AutoModel, "from_pretrained",
                          lambda name, *a, **k: AutoModel.from_config(AutoConfig.from_pretrained(name))),
        mock.patch.object(detector_module, "resnet34", lambda weights=None: models.resnet34(weights=None)),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        model = MediVLM(cfg)

    expected = model.state_dict().keys()
    state = {}
    for k, v in ckpt["model"].items():
        k2 = k.replace("image_encoder.vision.vision_model.", "image_encoder.vision.")
        state[k if k in expected else k2] = v
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model, build_image_transform(cfg.image_encoder.image_size, train=False)


def get_medivlm():
    return _cached("medivlm", _load_medivlm)


def get_radcheck():
    """None when no GOOGLE_API_KEY is configured."""
    def build():
        if not os.environ.get("GOOGLE_API_KEY"):
            return None
        import radcheck
        return radcheck.AMG_RAG_ReportSystem(google_api_key=os.environ["GOOGLE_API_KEY"])
    return _cached("radcheck", build)


REPORT_MAX_TOKENS = 150  # the checkpoint's own limit (77) cuts the first report off mid-sentence


def generate_report(enhanced: np.ndarray, num_beams: int = 4, max_tokens: int = REPORT_MAX_TOKENS) -> dict:
    """Returns {"report": cleaned text, "raw": decoder output}. See clean_report for what is removed."""
    model, transform = get_medivlm()
    x = transform(Image.fromarray(enhanced).convert("RGB")).unsqueeze(0).to(DEVICE)
    raw = model.generate(x, num_beams=int(num_beams), max_length=int(max_tokens))[0].strip()
    return {"report": clean_report(raw), "raw": raw}


# The MIMIC training text is a stringified Python list ("['findings: impression: ...', 'findings:' ..."), so the
# decoder writes one report, closes the string, then keeps going with extra, often contradictory "reports".
_ELEMENT_END = re.compile(r"""['"]\s*(?:,|\])""")
_SECTION = re.compile(r"\b(findings|impression)\s*:\s*", re.I)
# "xxxx" is MIMIC's de-identified prior-study date; there is no prior study here, so comparisons are dropped.
_COMPARISON = re.compile(r"^(?:in comparison (?:with|to)|as compared (?:to|with)|compared (?:to|with)|since|"
                         r"when compared (?:to|with))\b[^,]*?(?:xxxx|prior|previous)[^,]*,\s*", re.I)
_CHANGE_ONLY = re.compile(r"^(?:there (?:is|has been) )?(?:little|no|minimal) (?:significant |interval )?change"
                          r"(?: in the appearance)?\.?$|^(?:unchanged|stable)\.?$", re.I)


def _clean_sentences(text: str) -> list:
    out = []
    for s in re.split(r"(?<=\.)\s+", text.strip()):
        s = _COMPARISON.sub("", s.strip(" '\"[],")).strip()
        if not s or _CHANGE_ONLY.match(s) or "xxxx" in s.lower():
            continue
        if not s.endswith("."):  # cut off by the token limit
            continue
        out.append(s[0].upper() + s[1:])
    return out


def clean_report(raw: str) -> str:
    text = raw.strip().lstrip("[").lstrip("'\"")
    m = _ELEMENT_END.search(text)
    first = text[:m.start()] if m else text            # keep only the first report
    first = first.replace("\\n", " ").replace("\n", " ")

    # split into FINDINGS / IMPRESSION; "findings: impression:" (empty findings) is a common training artefact
    parts, label, pos = {"findings": [], "impression": []}, "findings", 0
    for sm in _SECTION.finditer(first):
        parts[label].append(first[pos:sm.start()])
        label, pos = sm.group(1).lower(), sm.end()
    parts[label].append(first[pos:])
    findings = _clean_sentences(" ".join(parts["findings"]))
    impression = _clean_sentences(" ".join(parts["impression"]))
    if not findings and len(impression) > 1:            # whole report landed under "impression:"
        findings, impression = impression[:-1], impression[-1:]

    lines = []
    if findings:
        lines.append("FINDINGS: " + " ".join(findings))
    if impression:
        lines.append("IMPRESSION: " + " ".join(impression))
    return "\n".join(lines) or first.strip()


def run_radcheck(report: str) -> tuple:
    """Returns (result dict or None, captured console log)."""
    system = get_radcheck()
    if system is None:
        return None, "RadCheck skipped: GOOGLE_API_KEY is not set (environment or .env)."
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return system.evaluate_medical_report(report), buf.getvalue()
    except Exception as e:  # network / quota errors shouldn't kill the pipeline
        return {"error": str(e)}, buf.getvalue()


# Simple phrase matching to read each target disease's status out of free-text reports
_TERMS = {
    "Cardiomegaly": r"cardiomegaly|enlarged (?:heart|cardiac)|(?:heart|cardiac silhouette) (?:is |appears )?enlarged|cardiac enlargement",
    "Edema": r"o?edema|vascular congestion|pulmonary congestion",
    "Pleural Effusion": r"effusions?",
}
_NORMAL = {
    "Cardiomegaly": r"(?:heart|cardiac) size (?:is )?(?:within )?normal|normal (?:heart|cardiac) size|cardiac silhouette (?:is )?(?:within )?normal|cardiomediastinal (?:silhouette|contours?) (?:is |are )?(?:within )?normal",
}
_NEG = r"\b(?:no|without|negative for|free of|resolved|absence of|not|nor)\b"


def report_vs_detection(report: str, detection: dict) -> dict:
    sentences = re.split(r"(?<=[.;])\s+", report.lower())
    out = {}
    for d in DISEASES:
        status = "not mentioned"
        for s in sentences:
            m = re.search(_TERMS[d], s)
            if m:
                status = "absent" if re.search(_NEG, s[:m.start()]) else "present"
                break
            if d in _NORMAL and re.search(_NORMAL[d], s):
                status = "absent"
                break
        cnn_pos = d in detection["positive"]
        if status == "not mentioned":
            agreement = "report silent" + (" (detector positive)" if cnn_pos else "")
        else:
            agreement = "agree" if (status == "present") == cnn_pos else "DISAGREE"
        out[d] = {"report": status, "detector": "positive" if cnn_pos else "negative", "agreement": agreement}
    return out


def save_result(r: dict) -> Path:
    """results/<image>/ : original, enhanced, heatmaps (PNG) + result.json — same layout as the notebook."""
    out = RESULTS_DIR / Path(r["image_path"]).stem
    out.mkdir(parents=True, exist_ok=True)
    enh = r["enhancement"]
    cv2.imwrite(str(out / "original.png"), enh["original"])
    cv2.imwrite(str(out / "enhanced.png"), enh["enhanced"])
    for d, cam in (r.get("saliency") or {}).items():
        cv2.imwrite(str(out / f"heatmap_{d.lower().replace(' ', '_')}.png"),
                    cv2.cvtColor(overlay(enh["enhanced"], cam), cv2.COLOR_RGB2BGR))
    for model_name, cams in (r.get("saliency_all") or {}).items():  # every detector's maps, when computed
        slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        (out / "heatmaps").mkdir(exist_ok=True)
        for d, cam in cams.items():
            cv2.imwrite(str(out / "heatmaps" / f"{slug}__{d.lower().replace(' ', '_')}.png"),
                        cv2.cvtColor(overlay(enh["enhanced"], cam), cv2.COLOR_RGB2BGR))
    record = {
        "image_path": r["image_path"],
        "enhancement": {k: v for k, v in enh.items() if not isinstance(v, np.ndarray)},
        "original_size": list(enh["original"].shape),
        "detection": r["detection"],
        "detection_all_models": r.get("detection_all_models"),
        "report": r.get("report"),
        "report_raw": r.get("report_raw"),
        "radcheck": r.get("radcheck"),
        "consistency": r.get("consistency"),
        "timing_s": r["timing_s"],
    }
    (out / "result.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return out
