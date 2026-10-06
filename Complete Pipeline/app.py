"""
ReXTrust — interactive interface for the full pipeline (full_pipeline.ipynb) with selectable models.

    python app.py            -> http://127.0.0.1:7860

Choose the enhancement model (3 + skip) and the disease-detection model (4). All 4 detectors always run for
comparison; Score-CAM heatmaps can be drawn for the selected detector or all 4. Then MediVLM writes the report
and RadCheck verifies it. Models load on first use and stay cached.
"""
import base64
import html
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

import pipeline_core as pc

SAMPLES = [str(pc.ROOT / n) for n in ["1.jpg", "2.jpg", "3.jpg", "9_IM-2407-1001.dcm.png"] if (pc.ROOT / n).exists()]
ENH_CHOICES = list(pc.ENHANCERS) + [pc.NO_ENHANCEMENT]
DET_CHOICES = list(pc.DETECTORS)
MODE_AUTO, MODE_ALWAYS = "Auto: only if the quality gate fails", "Always enhance"
CAM_ALL, CAM_SELECTED, CAM_OFF = "All 4 detectors", "Selected detector only", "Off"
HYBRID = "Hybrid CNN-Transformer"
SHORT = {"CNN — baseline (DenseNet121)": "CNN baseline", "CNN — matched recipe (DenseNet121)": "CNN matched",
         "Hybrid CNN-Transformer": "Hybrid CNN-Transformer", "CheXGCN (label graph)": "CheXGCN"}
esc = html.escape


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def img_uri(arr: np.ndarray, quality: int = 90) -> str:
    """numpy image (gray or RGB) -> inline JPEG data URI for HTML panels."""
    bgr = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def chip(text: str, kind: str = "neutral") -> str:
    """Render a styled HTML chip/badge with an optional semantic kind class."""
    return f'<span class="rx-chip {kind}">{esc(text)}</span>'


def empty(msg: str) -> str:
    """Return a placeholder div shown when a panel has no content yet."""
    return f'<div class="rx-empty">{esc(msg)}</div>'


def enh_info(name: str) -> str:
    """Return a short description for the selected enhancement model."""
    if name == pc.NO_ENHANCEMENT:
        return "Only resizes to 256×256. Detection and the report see the raw X-ray."
    spec = pc.ENHANCERS[name]
    return f"{spec['info']}  \n`{spec['path']().name}`"


def det_info(name: str) -> str:
    """Return a short description for the selected disease-detection model."""
    s = pc.DETECTORS[name]
    rows = " · ".join(f"{d} {a:.3f}" for d, a in zip(pc.DISEASES, s["auc"]))
    return f"{s['info']}  \nTest AUC: {rows}  \n`{s['file']}`"


def report_status() -> str:
    key = "set" if os.environ.get("GOOGLE_API_KEY") else "**not set**, RadCheck will be skipped"
    model = os.environ.get("RADCHECK_MODEL", "gemini-2.5-flash")
    return (f"MediVLM loads ~3 GB on first use (~1 min). RadCheck model: `{model}` · GOOGLE_API_KEY {key}. "
            "MediVLM was trained on reports clipped to 4 sentences, so expect short reports.")


# ──────────────────────────────────────────────────────────────────────────────
# HTML panels
# ──────────────────────────────────────────────────────────────────────────────
def header_html() -> str:
    """Generate the branded ReXTrust header HTML block."""
    steps = ["Quality gate", "Enhancement", "Detection ×4", "Score-CAM", "MediVLM report", "RadCheck"]
    flow = '<span class="rx-arrow">→</span>'.join(f'<span class="rx-step">{s}</span>' for s in steps)
    return (f'<div class="rx-hero"><div class="rx-hero-top"><div><div class="rx-hero-title">ReXTrust</div>'
            f'<div class="rx-hero-sub">Explainable multi-disease chest X-ray diagnosis with grounded reporting</div>'
            f'</div><div class="rx-hero-meta">{chip(pc.DEVICE.type.upper(), "info")}'
            f'{chip("Research use only", "warn")}</div></div><div class="rx-flow">{flow}</div></div>')


def prob_rows(det: dict) -> str:
    out = []
    for d in pc.DISEASES:
        p, t = det["probabilities"][d], det["thresholds"][d]
        pos = d in det["positive"]
        out.append(
            f'<div class="rx-prob"><div class="rx-prob-name">{d}</div>'
            f'<div class="rx-track"><div class="rx-fill {"pos" if pos else ""}" style="width:{p * 100:.1f}%"></div>'
            f'<div class="rx-thr" style="left:{t * 100:.1f}%"></div></div>'
            f'<div class="rx-prob-val">{p:.1%}</div>'
            f'<div>{chip("Positive", "bad") if pos else chip("Negative", "muted")}</div></div>')
    return "".join(out)


def radcheck_verdict(rc) -> tuple:
    if rc is None:
        return "UNVERIFIED", "neutral", "RadCheck was not run"
    if "error" in rc:
        return "UNVERIFIED", "neutral", f"RadCheck error: {rc['error'][:140]}"
    errors = rc.get("llm_errors") or []
    if rc["hallucination_detected"]:
        return "HALLUCINATION DETECTED", "bad", f"confidence {rc['overall_confidence']:.2f}" + \
            (f"; {len(errors)} step(s) incomplete" if errors else "")
    if errors:  # failed LLM steps default to "typical / no contradiction", so a pass here proves nothing
        return "UNVERIFIED", "neutral", "incomplete check: " + "; ".join(errors)[:200]
    return "CLINICALLY CONSISTENT", "good", f"confidence {rc['overall_confidence']:.2f}"


def overview_html(r: dict) -> str:
    enh, det = r["enhancement"], r["detection"]
    findings = "".join(chip(d, "bad") for d in det["positive"]) or chip("No finding among the 3 target diseases", "good")
    tiles = [
        ("Enhancement", esc(enh["model"]) if enh["was_enhanced"] else "Skipped", esc(enh["quality_before"]["reason"])),
        ("Detector", esc(SHORT[det["model"]]), f"top: {det['top_disease']} {det['top_confidence']:.1%}"),
    ]
    if r.get("report") is not None:
        v, cls, sub = radcheck_verdict(r.get("radcheck"))
        tiles.append(("RadCheck", f'<span class="rx-verdict {cls}">{v}</span>', esc(sub)))
        dis = sum(c["agreement"] == "DISAGREE" for c in r["consistency"].values())
        tiles.append(("Report vs detector", "Agree" if dis == 0 else f"{dis} disagreement{'s' if dis > 1 else ''}",
                      "on the 3 target diseases"))
    tile_html = "".join(f'<div class="rx-tile"><div class="rx-k">{k}</div><div class="rx-v">{v}</div>'
                        f'<div class="rx-s">{s}</div></div>' for k, v, s in tiles)
    t = r["timing_s"]
    timing = "".join(f'<span class="rx-time-step">{k} <b>{v:.1f}s</b></span>' for k, v in t.items())
    return (f'<div class="rx-card rx-overview"><div class="rx-ov-img"><img src="{img_uri(enh["enhanced"])}"/>'
            f'<div class="rx-s">Image fed downstream (256×256)</div></div><div class="rx-ov-body">'
            f'<div class="rx-k">Predicted findings</div><div class="rx-chips">{findings}</div>'
            f'<div class="rx-k" style="margin-top:14px">Probabilities · {esc(SHORT[det["model"]])} '
            f'(bar marker = decision threshold)</div>{prob_rows(det)}'
            f'<div class="rx-tiles">{tile_html}</div>'
            f'<div class="rx-timing">{timing}<span class="rx-time-step">total <b>{sum(t.values()):.1f}s</b></span>'
            f'</div></div></div>')


def quality_html(enh: dict) -> str:
    qb, qa = enh["quality_before"], enh["quality_after"]

    def row(name, key, better_low):
        b, a = qb[key], qa[key]
        if any(isinstance(v, float) and np.isnan(v) for v in (a, b)):
            delta = ""
        else:
            improved = (a < b) if better_low else (a > b)
            delta = chip(f"{a - b:+.2f}", "good" if improved else "bad")
        return f"<tr><td>{name}</td><td>{b}</td><td>{a}</td><td>{delta}</td></tr>"

    note = (f"<b>{esc(enh['model'])}</b> applied" if enh["was_enhanced"] else
            "<b>Enhancement skipped</b>: the image passed the quality gate")
    return (f'<div class="rx-card"><div class="rx-k">Quality gate</div><div class="rx-v">{note}</div>'
            f'<div class="rx-s">Gate before: {esc(qb["reason"])}</div>'
            f'<table class="rx-table"><thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Change</th></tr>'
            f'</thead><tbody>{row("BRISQUE (lower is better)", "brisque", True)}'
            f'{row("Entropy (higher is better)", "entropy", False)}'
            f'{row("Sharpness (higher is better)", "sharpness", False)}</tbody></table></div>')


def detection_html(all_det: dict, selected: str) -> str:
    head = "".join(f"<th>{d}</th>" for d in pc.DISEASES)
    rows = []
    for name, det in all_det.items():
        spec = pc.DETECTORS[name]
        cells = []
        for d, auc in zip(pc.DISEASES, spec["auc"]):
            p, pos = det["probabilities"][d], d in det["positive"]
            cells.append(f'<td><div class="rx-mini"><div class="rx-mini-fill {"pos" if pos else ""}" '
                         f'style="width:{p * 100:.1f}%"></div><div class="rx-mini-thr" '
                         f'style="left:{det["thresholds"][d] * 100:.1f}%"></div></div>'
                         f'<div class="rx-mini-val">{p:.1%} {"●" if pos else ""}</div>'
                         f'<div class="rx-s">test AUC {auc:.3f}</div></td>')
        sel = name == selected
        rows.append(f'<tr class="{"rx-sel" if sel else ""}"><td><div class="rx-v">{esc(SHORT[name])}</div>'
                    f'<div class="rx-s">{"selected · " if sel else ""}mean AUC {np.mean(spec["auc"]):.3f}</div></td>'
                    f'{"".join(cells)}<td>{esc(det["predicted_class"])}</td></tr>')
    return (f'<div class="rx-card"><div class="rx-k">All 4 detectors on this image</div>'
            f'<div class="rx-s">Bar = probability, marker = that model\'s validation-tuned threshold, '
            f'● = positive. AUC from the held-out CheXpert test split.</div>'
            f'<table class="rx-table rx-det"><thead><tr><th>Model</th>{head}<th>Prediction</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def heatmaps_html(enhanced, cams_by_model: dict, all_det: dict, selected: str, rollout=None) -> str:
    if not cams_by_model:
        return empty("Score-CAM was turned off for this run.")
    cards = []
    for name, cams in cams_by_model.items():
        det = all_det[name]
        tiles = []
        for j, (d, cam) in enumerate(cams.items()):
            p, pos = det["probabilities"][d], d in det["positive"]
            tiles.append(f'<figure class="rx-fig"><img src="{img_uri(pc.overlay(enhanced, cam))}"/>'
                         f'<figcaption><span>{"ABC"[j]} · {d}</span>'
                         f'{chip(f"{p:.0%}", "bad" if pos else "muted")}</figcaption></figure>')
        if rollout is not None and name == HYBRID:
            tiles.append(f'<figure class="rx-fig"><img src="{img_uri(pc.overlay(enhanced, rollout))}"/>'
                         f'<figcaption><span>Attention rollout</span>{chip("CLS→patches", "info")}</figcaption>'
                         f'</figure>')
        sel = name == selected
        cards.append(f'<div class="rx-card rx-cam {"rx-sel-card" if sel else ""}"><div class="rx-cam-head">'
                      f'<div class="rx-v">{esc(SHORT[name])}</div>{chip("selected", "info") if sel else ""}'
                      f'<div class="rx-s">{esc(det["predicted_class"])}</div></div>'
                      f'<div class="rx-figs">{"".join(tiles)}</div></div>')
    return ('<div class="rx-s" style="margin-bottom:10px">One Score-CAM map per disease, computed from that '
            'disease\'s own class score. Red = regions that raise the score most.</div>' + "".join(cards))


def _sentences(text: str) -> list:
    """Split report text into sentences for per-sentence highlighting."""
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]


def report_html(r: dict, image_name: str) -> str:
    if r.get("report") is None:
        return empty("Report generation was turned off for this run.")
    findings, impression = [], []
    for line in r["report"].splitlines():
        if line.startswith("FINDINGS:"):
            findings += _sentences(line[len("FINDINGS:"):])
        elif line.startswith("IMPRESSION:"):
            impression += _sentences(line[len("IMPRESSION:"):])
        else:
            findings += _sentences(line)
    v, cls, sub = radcheck_verdict(r.get("radcheck"))
    det = r["detection"]
    findings_html = "".join(f"<li>{esc(s)}</li>" for s in findings) or "<li>No findings section generated.</li>"
    impression_html = " ".join(esc(s) for s in impression) or "No impression generated."

    rc, claims = r.get("radcheck"), ""
    if rc and "detailed_findings" in rc:
        # A claim only counts as "Supported" when the steps that judge it (2: typical/atypical,
        # 4: contradictions) actually ran; failed steps silently default to "fine".
        failed = {s for s in ("Step 2", "Step 4") if any(e.startswith(s) for e in rc.get("llm_errors") or [])}
        unchecked = ("Not checked", "muted") if len(failed) == 2 else ("Partly checked", "warn") if failed else None
        items = []
        for f in rc["detailed_findings"]:
            bad = f["hallucination_risk"]
            label, kind = ("Hallucinated", "bad") if bad else unchecked or ("Supported", "good")
            notes = []
            cls = str(f["baseline_classification"]).lower()
            if cls == "impossible":
                notes.append("impossible as written")
            elif cls == "rare":
                ev = f.get("evidence_verdict") or "not grounded"
                notes.append("rare, PubMed: " + ev.replace("_", " "))
            if f["contradiction_detail"]:
                notes.append(f["contradiction_detail"])
            score = f'{f["grounding_score"]:.2f}' if kind != "muted" else "–"
            items.append(f'<li class="{"bad" if bad else ""}"><div class="rx-claim-top"><span>{esc(f["finding"])}'
                         f'{"<div class=rx-s>" + esc("; ".join(notes)) + "</div>" if notes else ""}</span>'
                         f'<span class="rx-claim-right"><span class="rx-s">score {score}</span>{chip(label, kind)}'
                         f'</span></div></li>')
        errs = "".join(f'<div class="rx-warnline">Incomplete step: {esc(e)}</div>' for e in rc.get("llm_errors") or [])
        claims = (f'<div class="rx-section-title">RadCheck: claim by claim</div>{errs}'
                  f'<ul class="rx-claims">{"".join(items)}</ul>')

    cons = "".join(
        f'<div class="rx-cons"><div class="rx-v">{d}</div><div class="rx-s">Report: {esc(c["report"])}</div>'
        f'<div class="rx-s">Detector: {esc(c["detector"])}</div>'
        f'{chip(c["agreement"], "bad" if c["agreement"] == "DISAGREE" else "good" if c["agreement"] == "agree" else "muted")}'
        f'</div>' for d, c in r["consistency"].items())

    return (f'<div class="rx-report"><div class="rx-report-head"><div><div class="rx-report-title">'
            f'Chest X-ray Report</div><div class="rx-s">{esc(image_name)} · generated '
            f'{datetime.now():%d %b %Y, %H:%M} · MediVLM + {esc(SHORT[det["model"]])}</div></div>'
            f'<span class="rx-verdict-pill {cls}">{v}</span></div>'
            f'<div class="rx-section-title">Findings</div><ul class="rx-findings">{findings_html}</ul>'
            f'<div class="rx-section-title">Impression</div><div class="rx-impression">{impression_html}</div>'
            f'<div class="rx-s rx-verdict-sub">RadCheck: {esc(sub)}</div>'
            f'{claims}<div class="rx-section-title">Report vs detector</div><div class="rx-cons-row">{cons}</div>'
            f'<div class="rx-disclaimer">AI-generated draft for research use. Not a diagnosis.</div></div>')


def status_html(text: str, kind: str = "good") -> str:
    return f'<div class="rx-status {kind}">{esc(text)}</div>'


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline run
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()  # torch.set_grad_enabled(False) in pipeline_core is thread-local; Gradio runs this in a worker
def run(image_path, enh_model, enh_mode, det_model, cam_scope, cam_channels, compare_enh,
        do_report, num_beams, report_tokens, do_radcheck, save, progress=gr.Progress()):
    if not image_path:
        raise gr.Error("Upload a chest X-ray or pick a sample first.")
    try:
        timing, t = {}, time.time()
        progress(0.02, desc=f"Enhancement: {enh_model}")
        enh = pc.enhance_image(image_path, enh_model, "always" if enh_mode == MODE_ALWAYS else "auto")
        enh_gallery = None
        if compare_enh:
            enh_gallery = [(enh["original_resized"], "Original (resized)")]
            for name in pc.ENHANCERS:
                e = pc.enhance_image(image_path, name, "always")
                q = e["quality_after"]
                enh_gallery.append((e["enhanced"], f"{name} · BRISQUE {q['brisque']} · entropy {q['entropy']}"))
        timing["enhance"] = time.time() - t

        t = time.time()
        progress(0.1, desc="Detection: all 4 models")
        all_det = {n: pc.detect_disease(enh["enhanced"], n) for n in pc.DETECTORS}
        det = all_det[det_model]
        timing["detect"] = time.time() - t

        t = time.time()
        cam_models = {CAM_ALL: list(pc.DETECTORS), CAM_SELECTED: [det_model]}.get(cam_scope, [])
        cams_all, rollout = {}, None
        for k, name in enumerate(cam_models):
            lo, span = 0.15 + 0.4 * k / len(cam_models), 0.4 / len(cam_models)
            cams_all[name] = pc.score_cam(enh["enhanced"], name, max_channels=int(cam_channels),
                                          progress=lambda f, lo=lo, span=span, name=name:
                                          progress(lo + span * f, desc=f"Score-CAM: {SHORT[name]}"))
        if HYBRID in cams_all:
            rollout = pc.get_detector(HYBRID).attention_rollout(pc.cnn_input(enh["enhanced"]))
        timing["explain"] = time.time() - t

        report = report_raw = radcheck = consistency = None
        rc_log = ""
        if do_report:
            t = time.time()
            progress(0.6, desc="MediVLM report (loading the model on first run…)" if not pc.is_loaded("medivlm")
                     else "MediVLM report")
            gen = pc.generate_report(enh["enhanced"], num_beams, report_tokens)
            report, report_raw = gen["report"], gen["raw"]
            consistency = pc.report_vs_detection(report, det)
            timing["report"] = time.time() - t
            if do_radcheck:
                t = time.time()
                progress(0.8, desc="RadCheck verification")
                radcheck, rc_log = pc.run_radcheck(report)
                timing["radcheck"] = time.time() - t

        result = {"image_path": str(image_path), "enhancement": enh, "detection": det,
                  "saliency": cams_all.get(det_model), "saliency_all": cams_all or None,
                  "detection_all_models": {n: {k: v for k, v in d.items() if k != "thresholds"}
                                           for n, d in all_det.items()},
                  "report": report, "report_raw": report_raw, "radcheck": radcheck, "consistency": consistency,
                  "timing_s": {k: round(v, 1) for k, v in timing.items()}}
        progress(0.97, desc="Saving")
        json_file = None
        if save:
            json_file = str(pc.save_result(result) / "result.json")

        name = Path(image_path).name
        return (overview_html(result), enh["original"], enh["enhanced"], quality_html(enh),
                gr.update(value=enh_gallery, visible=bool(enh_gallery)),
                detection_html(all_det, det_model),
                heatmaps_html(enh["enhanced"], cams_all, all_det, det_model, rollout),
                report_html(result, name), rc_log or "", report_raw or "", json_file,
                status_html(f"Done in {sum(result['timing_s'].values()):.1f}s"
                            + (f" · saved to results/{Path(image_path).stem}/" if json_file else "")))
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Look
# ──────────────────────────────────────────────────────────────────────────────
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                       radius_size="lg")

CSS = """
.gradio-container {max-width: 1480px !important}
:root {--rx-accent: #4f46e5; --rx-bad: #dc2626; --rx-good: #059669; --rx-warn: #d97706; --rx-info: #2563eb}
.rx-hero {border-radius: 18px; padding: 22px 26px; color: #fff;
          background: linear-gradient(120deg, #312e81 0%, #4f46e5 45%, #0d9488 100%);
          box-shadow: 0 10px 30px rgba(49, 46, 129, .25)}
.rx-hero-top {display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; align-items: flex-start}
.rx-hero-title {font-size: 30px; font-weight: 800; letter-spacing: -.02em; line-height: 1.1}
.rx-hero-sub {opacity: .85; margin-top: 4px; font-size: 15px}
.rx-hero-meta {display: flex; gap: 8px}
.rx-hero .rx-hero-meta .rx-chip {background: rgba(255,255,255,.16); color: #fff; border-color: rgba(255,255,255,.4)}
.rx-flow {display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 16px}
.rx-step {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
          padding: 4px 12px; font-size: 13px; font-weight: 500}
.rx-arrow {opacity: .7}
.rx-panel-title {font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
                 opacity: .6; margin: 4px 0 -4px}
.rx-card {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 18px 20px;
          background: var(--background-fill-primary); box-shadow: 0 1px 3px rgba(15, 23, 42, .06); margin-bottom: 14px}
.rx-empty {border: 1.5px dashed var(--border-color-primary); border-radius: 16px; padding: 48px 20px;
           text-align: center; color: var(--body-text-color-subdued)}
.rx-k {font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
       color: var(--body-text-color-subdued)}
.rx-v {font-weight: 650; font-size: 15px; margin: 2px 0}
.rx-s {font-size: 12.5px; color: var(--body-text-color-subdued)}
.rx-chip {display: inline-flex; align-items: center; font-size: 12px; font-weight: 600; padding: 3px 10px;
          border-radius: 999px; border: 1px solid transparent; white-space: nowrap}
.rx-chip.bad {background: rgba(220, 38, 38, .12); color: var(--rx-bad); border-color: rgba(220, 38, 38, .3)}
.rx-chip.good {background: rgba(5, 150, 105, .12); color: var(--rx-good); border-color: rgba(5, 150, 105, .3)}
.rx-chip.warn {background: rgba(217, 119, 6, .12); color: var(--rx-warn); border-color: rgba(217, 119, 6, .3)}
.rx-chip.info {background: rgba(79, 70, 229, .12); color: var(--rx-accent); border-color: rgba(79, 70, 229, .3)}
.rx-chip.muted, .rx-chip.neutral {background: var(--background-fill-secondary); color: var(--body-text-color-subdued);
          border-color: var(--border-color-primary)}
.rx-chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px}
.rx-chips .rx-chip {font-size: 14px; padding: 6px 14px}
.rx-overview {display: grid; grid-template-columns: 220px 1fr; gap: 22px}
.rx-ov-img img {width: 100%; border-radius: 12px; display: block; background: #000}
.rx-ov-img .rx-s {text-align: center; margin-top: 6px}
.rx-prob {display: grid; grid-template-columns: 150px 1fr 64px 88px; align-items: center; gap: 12px; margin-top: 10px}
.rx-prob-name {font-weight: 600; font-size: 14px}
.rx-prob-val {font-variant-numeric: tabular-nums; font-weight: 700; text-align: right}
.rx-track {position: relative; height: 10px; border-radius: 999px; background: var(--background-fill-secondary)}
.rx-fill {height: 100%; border-radius: 999px; background: linear-gradient(90deg, #818cf8, #4f46e5)}
.rx-fill.pos {background: linear-gradient(90deg, #f87171, #dc2626)}
.rx-thr {position: absolute; top: -5px; bottom: -5px; width: 2px; border-radius: 2px; background: var(--body-text-color)}
.rx-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-top: 18px}
.rx-tile {background: var(--background-fill-secondary); border-radius: 12px; padding: 10px 12px}
.rx-verdict.good {color: var(--rx-good)} .rx-verdict.bad {color: var(--rx-bad)}
.rx-verdict.neutral {color: var(--body-text-color-subdued)}
.rx-timing {display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px}
.rx-time-step {font-size: 12px; padding: 3px 10px; border-radius: 8px; background: var(--background-fill-secondary);
               color: var(--body-text-color-subdued)}
.rx-table {width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13.5px}
.rx-table th {text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
              color: var(--body-text-color-subdued); padding: 8px 10px; border-bottom: 1px solid var(--border-color-primary)}
.rx-table td {padding: 10px; border-bottom: 1px solid var(--border-color-primary); vertical-align: middle}
.rx-table tr:last-child td {border-bottom: none}
.rx-det tr.rx-sel td {background: rgba(79, 70, 229, .07)}
.rx-det tr.rx-sel td:first-child {box-shadow: inset 3px 0 0 var(--rx-accent)}
.rx-mini {position: relative; height: 6px; border-radius: 999px; background: var(--background-fill-secondary);
          min-width: 90px; margin: 4px 0}
.rx-mini-fill {height: 100%; border-radius: 999px; background: #818cf8}
.rx-mini-fill.pos {background: var(--rx-bad)} .rx-mini-fill.ok {background: var(--rx-good)}
.rx-mini-thr {position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--body-text-color)}
.rx-mini-val {font-weight: 650; font-variant-numeric: tabular-nums}
.rx-cam-head {display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px}
.rx-cam-head .rx-s {margin-left: auto}
.rx-sel-card {border-color: var(--rx-accent); box-shadow: 0 0 0 1px var(--rx-accent), 0 6px 18px rgba(79, 70, 229, .12)}
.rx-figs {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px}
.rx-fig {margin: 0; border-radius: 12px; overflow: hidden; background: var(--background-fill-secondary)}
.rx-fig img {width: 100%; display: block; aspect-ratio: 1; object-fit: cover}
.rx-fig figcaption {display: flex; justify-content: space-between; align-items: center; gap: 6px;
                    padding: 8px 10px; font-size: 12.5px; font-weight: 600}
.rx-report {max-width: 900px; margin: 0 auto; border: 1px solid var(--border-color-primary); border-radius: 18px;
            background: var(--background-fill-primary); padding: 26px 30px; box-shadow: 0 4px 20px rgba(15, 23, 42, .06)}
.rx-report-head {display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap;
                 padding-bottom: 14px; border-bottom: 2px solid var(--border-color-primary)}
.rx-report-title {font-size: 22px; font-weight: 800; letter-spacing: -.01em}
.rx-verdict-pill {font-weight: 700; font-size: 12.5px; padding: 6px 14px; border-radius: 999px; letter-spacing: .03em}
.rx-verdict-pill.good {background: rgba(5, 150, 105, .12); color: var(--rx-good)}
.rx-verdict-pill.bad {background: rgba(220, 38, 38, .12); color: var(--rx-bad)}
.rx-verdict-pill.neutral {background: var(--background-fill-secondary); color: var(--body-text-color-subdued)}
.rx-verdict-sub {margin-top: 6px}
.rx-section-title {font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .1em;
                   color: var(--rx-accent); margin: 20px 0 8px}
.rx-findings {margin: 0; padding-left: 20px; line-height: 1.75; font-size: 15px}
.rx-impression {font-size: 16px; font-weight: 600; padding: 12px 16px; border-radius: 12px;
                background: rgba(79, 70, 229, .07); border-left: 4px solid var(--rx-accent)}
.rx-claims {list-style: none; margin: 0; padding: 0; border: 1px solid var(--border-color-primary);
             border-radius: 12px; overflow: hidden}
.rx-claims li {padding: 9px 14px; border-bottom: 1px solid var(--border-color-primary)}
.rx-claims li:last-child {border-bottom: none}
.rx-claim-right {display: inline-flex; align-items: center; gap: 10px; flex-shrink: 0}
.rx-claims li.bad {background: rgba(220, 38, 38, .07); box-shadow: inset 3px 0 0 var(--rx-bad)}
.rx-claim-top {display: flex; justify-content: space-between; gap: 10px; align-items: center; font-size: 14px}
.rx-warnline {font-size: 12.5px; color: var(--rx-warn); margin-bottom: 6px}
.rx-cons-row {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px}
.rx-cons {background: var(--background-fill-secondary); border-radius: 12px; padding: 10px 12px}
.rx-cons .rx-chip {margin-top: 6px}
.rx-disclaimer {margin-top: 20px; font-size: 12px; color: var(--body-text-color-subdued); text-align: center}
.rx-status {border-radius: 12px; padding: 10px 14px; font-size: 13.5px; font-weight: 600}
.rx-status.good {background: rgba(5, 150, 105, .1); color: var(--rx-good)}
.rx-status.idle {background: var(--background-fill-secondary); color: var(--body-text-color-subdued)}
#rx-run {font-size: 16px; font-weight: 700; min-height: 48px}
@media (max-width: 900px) {
  .rx-overview {grid-template-columns: 1fr}
  .rx-prob {grid-template-columns: 1fr 64px; row-gap: 4px}
  .rx-prob .rx-track {grid-column: 1 / -1; order: 3}
  .rx-cons-row {grid-template-columns: 1fr}
  .rx-figs {grid-template-columns: repeat(2, minmax(0, 1fr))}
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="ReXTrust") as demo:
    gr.HTML(header_html())

    with gr.Row(equal_height=False):
        # ── Controls ──────────────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=340):
            gr.HTML('<div class="rx-panel-title">Input</div>')
            image_in = gr.Image(type="filepath", label="Chest X-ray (JPG / PNG)", height=280,
                                sources=["upload", "clipboard"])
            if SAMPLES:
                gr.Examples(SAMPLES, inputs=image_in, label="Sample X-rays")

            gr.HTML('<div class="rx-panel-title">Models</div>')
            with gr.Accordion("1 · Image enhancement", open=True):
                enh_model = gr.Radio(ENH_CHOICES, value=ENH_CHOICES[0], show_label=False)
                enh_mode = gr.Radio([MODE_AUTO, MODE_ALWAYS], value=MODE_AUTO, label="When to enhance")
                enh_md = gr.Markdown(enh_info(ENH_CHOICES[0]))
                compare_enh = gr.Checkbox(False, label="Also compare all 3 enhancers side by side")
            with gr.Accordion("2 · Disease detection", open=True):
                det_model = gr.Radio(DET_CHOICES, value=DET_CHOICES[0], show_label=False)
                det_md = gr.Markdown(det_info(DET_CHOICES[0]))
                gr.Markdown("All 4 detectors always run for comparison; the selected one drives the summary "
                            "and the report-consistency check.")
            with gr.Accordion("3 · Explainability", open=True):
                cam_scope = gr.Radio([CAM_ALL, CAM_SELECTED, CAM_OFF], value=CAM_ALL, label="Score-CAM heatmaps for")
                cam_channels = gr.Slider(64, 1024, value=128 if pc.DEVICE.type == "cpu" else 1024, step=64,
                                         label="Score-CAM channels (1024 = exact; on CPU ~10–25 s per model at 128)")
            with gr.Accordion("4 · Report", open=False):
                do_report = gr.Checkbox(True, label="Generate the MediVLM report")
                do_radcheck = gr.Checkbox(bool(os.environ.get("GOOGLE_API_KEY")),
                                          label="Verify it with RadCheck (Gemini API)")
                num_beams = gr.Slider(1, 6, value=4, step=1, label="Beam search width")
                report_tokens = gr.Slider(77, 256, value=pc.REPORT_MAX_TOKENS, step=1, label="Max report tokens")
                gr.Markdown(report_status())
            save = gr.Checkbox(True, label="Save outputs to results/<image>/")
            run_btn = gr.Button("Run pipeline", variant="primary", elem_id="rx-run")
            status = gr.HTML(status_html("Pick an X-ray and press Run.", "idle"))

        # ── Results ───────────────────────────────────────────────────────────
        with gr.Column(scale=8):
            with gr.Tabs():
                with gr.Tab("Overview"):
                    overview = gr.HTML(empty("Results appear here after you run the pipeline."))
                with gr.Tab("Enhancement"):
                    with gr.Row():
                        orig_img = gr.Image(label="Original", height=380, interactive=False)
                        enh_img = gr.Image(label="Enhanced (256×256, fed downstream)", height=380, interactive=False)
                    quality = gr.HTML()
                    enh_gallery = gr.Gallery(label="All enhancers", columns=4, height=300, visible=False)
                with gr.Tab("Detection"):
                    detection = gr.HTML(empty("Run the pipeline to compare the 4 detectors."))
                with gr.Tab("Explainability"):
                    heatmaps = gr.HTML(empty("Run the pipeline to see the Score-CAM heatmaps."))
                with gr.Tab("Report"):
                    report = gr.HTML(empty("Run the pipeline to generate a report."))
                    with gr.Accordion("RadCheck log (all 5 steps)", open=False):
                        rc_log = gr.Textbox(lines=16, max_lines=40, show_label=False)
                    with gr.Accordion("Raw MediVLM output (before cleaning)", open=False):
                        gr.Markdown("The decoder writes a stringified list and keeps generating extra, often "
                                    "contradictory reports after the first one. Only the first report is kept, "
                                    "and comparisons to the de-identified prior study (`xxxx`) and cut-off "
                                    "sentences are removed. RadCheck and the consistency check see the cleaned text.")
                        raw_report = gr.Textbox(lines=6, max_lines=20, show_label=False)
                with gr.Tab("Export"):
                    json_out = gr.File(label="result.json (heatmaps and images are saved next to it)")

    enh_model.change(enh_info, enh_model, enh_md)
    enh_model.change(lambda m: gr.update(interactive=m != pc.NO_ENHANCEMENT), enh_model, enh_mode)
    det_model.change(det_info, det_model, det_md)
    cam_scope.change(lambda s: gr.update(interactive=s != CAM_OFF), cam_scope, cam_channels)
    do_report.change(lambda v: (gr.update(interactive=v),) * 3, do_report, [num_beams, report_tokens, do_radcheck])

    run_btn.click(
        run,
        [image_in, enh_model, enh_mode, det_model, cam_scope, cam_channels, compare_enh,
         do_report, num_beams, report_tokens, do_radcheck, save],
        [overview, orig_img, enh_img, quality, enh_gallery, detection, heatmaps, report, rc_log, raw_report,
         json_out, status],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(theme=THEME, css=CSS, inbrowser=True,
                                                   allowed_paths=[str(pc.RESULTS_DIR)])
