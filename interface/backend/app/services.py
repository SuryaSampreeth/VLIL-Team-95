"""Stage logic shared by the standalone endpoints and /pipeline/run.

Engine choice per call (config `mode`):
  mock -> MockEngine;  real -> RealEngine, errors propagate;  auto -> RealEngine, falling back to MockEngine per stage.
"""
import logging
import time

import numpy as np

from . import imaging as im
from .config import get_settings
from .engine_mock import MockEngine
from .engine_real import RealEngine

log = logging.getLogger("rextrust")
S = get_settings()
MOCK, REAL = MockEngine(S), RealEngine(S)


class StageError(RuntimeError):
    pass


def _short(e: Exception) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return msg[:220]


def _call(available: bool, what: str, real_fn, mock_fn):
    """-> (value, mode, notice)."""
    if S.mode == "mock":
        return mock_fn(), "mock", None
    if S.mode == "real" or available:
        try:
            return real_fn(), "real", None
        except Exception as e:  # noqa: BLE001 — any model failure falls back in auto mode
            log.exception("%s failed", what)
            if S.mode == "real":
                raise StageError(f"{what} failed: {_short(e)}") from e
            return mock_fn(), "mock", f"{what}: the real model failed ({_short(e)}), so this is mock output."
    return mock_fn(), "mock", f"{what}: weights are not available in this install, so this is mock output."


def _brisque():
    if S.mode == "mock":
        return None
    try:
        return REAL.brisque_fn()
    except Exception:  # torch / piq missing -> entropy-only gate, same as pipeline_core
        return None


def _model_ref(spec: dict) -> dict:
    return {"id": spec["id"], "name": spec["name"]}


# ── Stage 1 ──────────────────────────────────────────────────────────────────
def stage1(gray: np.ndarray, model_id: str = None, gate_mode: str = "always") -> dict:
    t0 = time.perf_counter()
    spec = S.model(1, model_id)
    r256 = im.resize(gray)
    bf = _brisque()
    q_before = im.quality(r256, brisque_fn=bf)
    qg = S.stage1["quality_gate"]
    verdict = im.gate(q_before, qg["brisque_max"], qg["entropy_min"])

    run = gate_mode == "always" or verdict["needs_enhancement"]
    if run:
        enhanced, mode, notice = _call(S.has_weights(spec), f"Enhancement ({spec['name']})",
                                       lambda: REAL.enhance(r256, spec), lambda: MOCK.enhance(r256, spec))
    else:
        enhanced, mode, notice = r256.copy(), ("mock" if S.mode == "mock" else "real"), None
    q_after = im.quality(enhanced, reference=r256, brisque_fn=bf)
    return {
        "stage": 1, "mode": mode, "notice": notice, "model": _model_ref(spec),
        "gate": {**verdict, "mode": gate_mode}, "was_enhanced": run,
        "original": im.to_data_url(r256), "enhanced": im.to_data_url(enhanced),
        "metrics": {"before": q_before, "after": q_after},
        "benchmark": spec.get("benchmark"),
        "size": [int(gray.shape[1]), int(gray.shape[0])],
        "timing_ms": round((time.perf_counter() - t0) * 1000),
        "_enhanced": enhanced,  # stripped before the response; used by /pipeline/run
    }


# ── Stage 2 ──────────────────────────────────────────────────────────────────
def stage2(gray: np.ndarray, model_id: str = None, heatmaps: bool = True) -> dict:
    t0 = time.perf_counter()
    spec = S.model(2, model_id)
    r256 = im.resize(gray)
    what = f"Detection ({spec['name']})"
    probs, mode, notice = _call(S.has_weights(spec), what,
                                lambda: REAL.detect(r256, spec), lambda: MOCK.detect(r256, spec))
    diseases = []
    thresholds = dict(zip(S.diseases, spec["thresholds"]))
    metrics = {d: {"auc": a, "f1": f} for d, a, f in zip(S.diseases, spec["auc"], spec["f1"])}

    cams, cam_mode = {}, mode
    if heatmaps:
        if mode == "mock":
            cams = MOCK.cams(r256, spec)
        else:
            cams, cam_mode, cnotice = _call(True, f"Score-CAM ({spec['name']})",
                                            lambda: REAL.cams(r256, spec), lambda: MOCK.cams(r256, spec))
            if cam_mode == "mock":
                notice = cnotice
    base = im.resize(gray, 512)
    for d in S.diseases:
        entry = {"name": d, "probability": round(probs[d], 4), "threshold": thresholds[d],
                 "positive": probs[d] > thresholds[d], **metrics[d]}
        if d in cams:
            heat = im.colormap(cams[d], 512)
            entry["heatmap"] = im.to_data_url(heat, ".jpg")
            entry["overlay"] = im.to_data_url(im.overlay(base, heat), ".jpg")
        diseases.append(entry)

    positive = [d["name"] for d in diseases if d["positive"]]
    top = max(diseases, key=lambda d: d["probability"])
    return {
        "stage": 2, "mode": mode, "notice": notice, "model": _model_ref(spec),
        "explainer": ("Score-CAM (denseblock4, one map per disease)" if cam_mode == "real"
                      else "Mock heatmaps from anatomical priors (not Score-CAM)"),
        "base_image": im.to_data_url(base, ".jpg"), "diseases": diseases, "positive": positive,
        "predicted_class": ", ".join(positive) if positive else "No finding (of the 3 target diseases)",
        "top_disease": top["name"], "top_probability": top["probability"],
        "timing_ms": round((time.perf_counter() - t0) * 1000),
    }


# ── Stage 3 ──────────────────────────────────────────────────────────────────
def _evaluation(radcheck: dict, mode: str) -> dict:
    claims = [{k: c.get(k) for k in ("finding", "target", "status", "baseline_classification",
                                       "baseline_reasoning", "pubmed_grounded", "is_contradicted",
                                       "contradiction_detail", "grounding_score", "hallucination_risk")}
              for c in (radcheck.get("detailed_findings") or [])]
    flagged = [c for c in claims if c["hallucination_risk"] or c["is_contradicted"]]
    return {
        "available": True, "mode": mode, "verifier": S.stage3["verifier"]["name"],
        "overall_confidence": radcheck.get("overall_confidence"),
        "hallucination_detected": bool(radcheck.get("hallucination_detected")),
        "fully_verified": bool(radcheck.get("fully_verified")),
        "counts": {
            "claims": len(claims), "verified": len(claims) - len(flagged), "flagged": len(flagged),
            "typical": sum(c["baseline_classification"] == "typical" for c in claims),
            "atypical": sum(c["baseline_classification"] == "atypical" for c in claims),
            "pubmed_grounded": sum(bool(c["pubmed_grounded"]) for c in claims),
            "contradicted": sum(bool(c["is_contradicted"]) for c in claims),
        },
        "claims": claims,
        "contradictions": radcheck.get("contradictions") or [],
        "graph_stats": radcheck.get("graph_stats") or {},
    }


def stage3(gray: np.ndarray, findings: dict = None, crosscheck_model: str = None) -> dict:
    t0 = time.perf_counter()
    r256 = im.resize(gray)
    ckpt = S.weights_path(S.stage3["report_model"]["weights"])
    rep, mode, notice = _call(ckpt.exists(), "Report generation (MediVLM)",
                              lambda: REAL.report(r256), lambda: MOCK.report(r256))
    sections = im.split_report(rep["text"])

    if mode == "mock":
        evaluation = _evaluation(MOCK.radcheck(rep["text"]), "mock")
    else:
        try:
            evaluation = _evaluation(REAL.radcheck(rep["text"]), "real")
        except Exception as e:  # noqa: BLE001 — a missing key or quota error should not lose the report
            log.warning("RadCheck unavailable: %s", e)
            evaluation = {"available": False, "reason": f"RadCheck could not run: {_short(e)}"}

    detection = None
    if findings and findings.get("positive") is not None:
        detection = {"source": findings.get("model", "provided findings"), "positive": list(findings["positive"]),
                     "probabilities": findings.get("probabilities")}
    elif crosscheck_model:
        d2 = stage2(gray, crosscheck_model, heatmaps=False)
        detection = {"source": d2["model"]["name"], "positive": d2["positive"], "mode": d2["mode"],
                     "probabilities": {d["name"]: d["probability"] for d in d2["diseases"]}}

    consistency = None
    if detection:
        table = im.report_vs_detection(rep["text"], detection["positive"], S.diseases)
        mentioned = [v for v in table.values() if v["agreement"] != "silent"]
        consistency = {"detector": detection, "per_disease": table,
                       "agreement_rate": (sum(v["agreement"] == "agree" for v in mentioned) / len(mentioned))
                       if mentioned else None}

    return {
        "stage": 3, "mode": mode, "notice": notice,
        "model": {"id": S.stage3["report_model"]["id"], "name": S.stage3["report_model"]["name"]},
        "report": {"text": rep["text"], "raw": rep.get("raw"), **sections},
        "sentences": im.annotate_sentences(sections, evaluation.get("claims", [])),
        "evaluation": evaluation, "consistency": consistency,
        "input_image": im.to_data_url(r256),
        "timing_ms": round((time.perf_counter() - t0) * 1000),
    }


# ── Complete pipeline ────────────────────────────────────────────────────────
def pipeline(gray: np.ndarray, stage1_model: str = None, stage2_model: str = None, gate_mode: str = "auto") -> dict:
    t0 = time.perf_counter()
    s1 = stage1(gray, stage1_model, gate_mode)
    enhanced = s1.pop("_enhanced")
    s2 = stage2(enhanced, stage2_model)
    findings = {"model": s2["model"]["name"], "positive": s2["positive"],
                "probabilities": {d["name"]: d["probability"] for d in s2["diseases"]}}
    s3 = stage3(enhanced, findings)
    return {"stage1": s1, "stage2": s2, "stage3": s3,
            "modes": {"stage1": s1["mode"], "stage2": s2["mode"], "stage3": s3["mode"]},
            "timing_ms": round((time.perf_counter() - t0) * 1000)}


def warmup():
    """Import torch + pipeline_core and build the default models in the background (auto/real mode)."""
    if S.mode == "mock":
        return
    t0 = time.perf_counter()
    try:
        _ = REAL.pc
        for stage in (1, 2):
            spec = S.model(stage)
            if spec.get("adapter", "network") == "network" and S.has_weights(spec):
                REAL._network(spec)
        log.info("Warm-up done in %.1fs on %s", time.perf_counter() - t0, REAL.device())
    except Exception as e:  # noqa: BLE001
        log.warning("Warm-up failed (%s); auto mode will use mock outputs where needed", _short(e))
