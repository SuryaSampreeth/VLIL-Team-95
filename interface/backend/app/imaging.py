"""Image I/O, encoding and the no-reference quality metrics used in the PDFs (BRISQUE, Entropy, Sharpness, CII, EME).

Kept free of torch so mock mode runs with only numpy + OpenCV. BRISQUE needs piq (torch); the real engine passes it in.
"""
import base64
import io
import re

import cv2
import numpy as np

SIZE = 256  # every stage works on the 256x256 grayscale image, as in pipeline_core.py
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class ImageError(ValueError):
    pass


def decode(data: bytes, filename: str = "") -> np.ndarray:
    """Upload bytes -> uint8 grayscale. Accepts PNG/JPEG/BMP/TIFF (8 or 16 bit) and DICOM when pydicom is installed."""
    if filename.lower().endswith(".dcm") or data[128:132] == b"DICM":
        try:
            import pydicom
        except ImportError as e:
            raise ImageError("DICOM upload needs `pip install pydicom`; or upload a PNG/JPEG export.") from e
        img = pydicom.dcmread(io.BytesIO(data)).pixel_array.astype(np.float32)
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ImageError("The file could not be decoded as an image.")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:  # 16-bit PNG / DICOM -> min-max to uint8
        img = img.astype(np.float32)
        img = (img - img.min()) / max(float(img.max() - img.min()), 1e-6) * 255
        img = img.astype(np.uint8)
    if min(img.shape[:2]) < 64:
        raise ImageError(f"Image is too small ({img.shape[1]}x{img.shape[0]}); at least 64x64 is required.")
    return img


def resize(gray: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Square resize without keeping aspect ratio, matching how the models were trained."""
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA if gray.shape[0] > size else cv2.INTER_CUBIC)


def to_data_url(img: np.ndarray, fmt: str = ".png") -> str:
    """numpy gray or RGB -> data URL."""
    bgr = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(fmt, bgr)
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf).decode()


def colormap(cam: np.ndarray, size: int) -> np.ndarray:
    """CAM in [0,1] -> RGB jet heatmap (the colormap used for the Score-CAM figures)."""
    cam = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(gray: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return np.uint8(alpha * heat + (1 - alpha) * cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


# ── quality metrics ──────────────────────────────────────────────────────────
def entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _local_contrast(gray: np.ndarray) -> float:
    """Mean Michelson contrast over 3x3 neighbourhoods (the C in CII = C_processed / C_original)."""
    g = gray.astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    mx, mn = cv2.dilate(g, k), cv2.erode(g, k)
    return float(((mx - mn) / (mx + mn + 1e-6)).mean())


def cii(original: np.ndarray, processed: np.ndarray) -> float:
    return _local_contrast(processed) / max(_local_contrast(original), 1e-6)


def eme(gray: np.ndarray, blocks: int = 8) -> float:
    """Measure of Enhancement: mean over k1 x k2 blocks of 20*log10(I_max / I_min)."""
    g = gray.astype(np.float32) + 1.0
    h, w = g.shape
    bh, bw = h // blocks, w // blocks
    vals = [20 * np.log10(b.max() / b.min())
            for i in range(blocks) for j in range(blocks)
            for b in [g[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]]]
    return float(np.mean(vals))


def quality(gray: np.ndarray, reference: np.ndarray = None, brisque_fn=None) -> dict:
    out = {"brisque": None, "entropy": round(entropy(gray), 3), "sharpness": round(sharpness(gray), 1),
           "cii": round(cii(reference if reference is not None else gray, gray), 3), "eme": round(eme(gray), 2)}
    if brisque_fn is not None:
        try:
            out["brisque"] = round(float(brisque_fn(gray)), 2)
        except Exception:  # BRISQUE can fail on flat images; the gate then uses entropy only
            pass
    return out


def gate(q: dict, brisque_max: float, entropy_min: float) -> dict:
    """Conditional-enhancement quality gate (TEAM_95_PR2 objective 1: enhance only when needed)."""
    reasons = []
    if q.get("brisque") is not None and q["brisque"] > brisque_max:
        reasons.append(f"BRISQUE {q['brisque']:.2f} > {brisque_max}")
    if q["entropy"] < entropy_min:
        reasons.append(f"Entropy {q['entropy']:.2f} < {entropy_min}")
    return {"needs_enhancement": bool(reasons), "reason": "; ".join(reasons) or "Passes the quality gate",
            "thresholds": {"brisque_max": brisque_max, "entropy_min": entropy_min}}


# ── report helpers (ported from pipeline_core.report_vs_detection so mock mode needs no torch) ──────────────
_TERMS = {
    "Cardiomegaly": r"cardiomegaly|enlarged (?:heart|cardiac)|(?:heart|cardiac silhouette) (?:is |appears )?enlarged|cardiac enlargement",
    "Edema": r"o?edema|vascular congestion|pulmonary congestion",
    "Pleural Effusion": r"effusions?",
}
_NORMAL = {
    "Cardiomegaly": r"(?:heart|cardiac) size (?:is )?(?:within )?normal|normal (?:heart|cardiac) size|cardiac silhouette (?:is )?(?:within )?normal|cardiomediastinal (?:silhouette|contours?) (?:is |are )?(?:within )?normal",
}
_NEG = r"\b(?:no|without|negative for|free of|resolved|absence of|not|nor)\b"


def report_vs_detection(report: str, positive: list, diseases: list) -> dict:
    sentences = re.split(r"(?<=[.;])\s+", report.lower())
    out = {}
    for d in diseases:
        status = "not mentioned"
        for s in sentences:
            m = re.search(_TERMS.get(d, re.escape(d.lower())), s)
            if m:
                status = "absent" if re.search(_NEG, s[:m.start()]) else "present"
                break
            if d in _NORMAL and re.search(_NORMAL[d], s):
                status = "absent"
                break
        det_pos = d in positive
        if status == "not mentioned":
            agreement = "silent"
        else:
            agreement = "agree" if (status == "present") == det_pos else "disagree"
        out[d] = {"report": status, "detector": "positive" if det_pos else "negative", "agreement": agreement}
    return out


def split_report(text: str) -> dict:
    """'FINDINGS: ... \\nIMPRESSION: ...' -> sections."""
    findings, impression = "", ""
    for line in text.splitlines():
        m = re.match(r"\s*(findings|impression)\s*:\s*(.*)", line, re.I)
        if m and m.group(1).lower() == "findings":
            findings = m.group(2).strip()
        elif m:
            impression = m.group(2).strip()
        elif line.strip():
            findings = (findings + " " + line.strip()).strip()
    return {"findings": findings, "impression": impression}


def annotate_sentences(sections: dict, claims: list) -> list:
    """Attach RadCheck claims to the sentence they came from so the UI can highlight unsupported statements."""
    out = []
    for section in ("findings", "impression"):
        for s in re.split(r"(?<=\.)\s+", sections.get(section, "").strip()):
            if not s:
                continue
            low = s.lower()
            matched = [c for c in claims if c.get("finding") and c["finding"].lower() in low]
            if any(c.get("is_contradicted") for c in matched):
                status = "contradicted"
            elif any(c.get("hallucination_risk") for c in matched):
                status = "unsupported"
            elif matched:
                status = "verified"
            else:
                status = "unchecked"
            out.append({"section": section, "text": s, "status": status,
                        "claims": list(dict.fromkeys(c["finding"] for c in matched))})
    return out
