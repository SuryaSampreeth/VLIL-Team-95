# ReXTrust: XAI-Driven Multi-Disease Diagnosis in Chest X-rays via Conditional Enhancement and Grounded Reporting

## Overview

ReXTrust is an end-to-end, explainable AI pipeline for multi-disease diagnosis from chest X-rays. It addresses three unresolved challenges in AI-based chest X-ray diagnosis:

1. **Degraded image quality** — X-rays from low-resource settings often suffer from noise, low contrast, and poor illumination.
2. **Lack of per-disease spatial verification** — existing explainability methods produce a single blended saliency map instead of independently verifiable, per-disease heatmaps.
3. **Ungrounded AI-generated reports** — vision-language models can generate fluent but hallucinated clinical text with no verification against external medical knowledge.

The system is built as a five-stage pipeline:

```
Chest X-ray → Image Enhancement → Disease Detection → Explainability (Score-CAM)
           → Report Generation (MediVLM) → Grounding & Validation (RadCheck RAG)
```

---

## Pipeline Architecture

1. **Input** — Chest X-ray from the CheXpert dataset (DICOM/JPEG).
2. **Image Quality Assessment** — Routes each image to enhancement if quality is low, or passes it unchanged if acceptable.
3. **Image Enhancement** — Attention U-Net + CLAHE improves contrast and denoises without destroying anatomical/clinical detail.
4. **Disease Detection** — Multi-label classifier producing per-disease confidence scores.
5. **Explainability** — Score-CAM generates an independent saliency heatmap per predicted disease.
6. **Report Generation** — MediVLM (vision-language model) generates a radiology report (findings and impression) from the enhanced X-ray. The detected diseases are not passed to it; the report is compared with the detector's output afterwards.
7. **RadCheck — Report Grounding** — Validates the generated report against external medical evidence and internal consistency, producing a confidence-scored, hallucination-checked report.
8. **Structured Diagnostic Output** — Enhanced image + disease confidence scores + per-disease saliency maps + RAG-grounded clinical report.

---

## Module 1: Image Enhancement

**Objective:** Improve visual quality and contrast of chest X-rays while preserving anatomical detail needed by downstream detection and reporting stages.

### Methods Implemented & Compared

**Classical & learned baselines**
- **CLAHE** — clip-limited adaptive histogram equalization (classical baseline)
- **UniMIE** — pretrained diffusion-prior backbone (Fei et al., 2025)
- **Zero-DCE** — unsupervised DCE-Net with spatial-consistency, exposure, color-constancy, and illumination-smoothness losses (Guo et al., 2020)
- **U-Net** — supervised model trained on CLAHE-target pairs with L1 + SSIM + Laplacian loss, 60 epochs (Ronneberger et al., 2015)

**Proposed hybrid pipelines**
- **U-Net + CLAHE** — U-Net denoises, CLAHE boosts contrast on the clean output
- **Attention U-Net + CBAM** — 4-stage ResBlock encoder/decoder with CBAM channel+spatial attention at the bottleneck and Attention Gates on every skip connection
- **Attention U-Net + CLAHE (Ours — final proposed pipeline)** — best average rank across all 5 evaluation metrics

### Proposed Architecture: Attention U-Net + CLAHE

- **Residual Blocks** — preserve important features while improving gradient/feature flow
- **CBAM** — channel + spatial attention applied at the bottleneck
- **Attention Gates** — selectively pass relevant encoder features to the decoder via skip connections
- **CLAHE (post-processing)** — enhances local contrast (clip limit 2.0, tile size 8×8) while limiting noise amplification

### Evaluation

Evaluated on 23 low-quality CheXpert images across 8 methods using **BRISQUE (↓), Entropy (↑), Sharpness (↑), CII (↑), EME (↑)**.

| Method | BRISQUE ↓ | Entropy ↑ | Sharpness ↑ | CII | Avg Rank |
|---|---|---|---|---|---|
| **Attn U-Net + CLAHE (Ours)** | 13.35 | **7.92** | 2082.3 | **1.90** | **2.2** |
| U-Net V3 + CLAHE | 17.51 | 7.91 | 1859.7 | 1.76 | 3.2 |
| CLAHE | 7.01 | 7.84 | 3609.6 | 1.71 | 3.8 |
| Attention U-Net | 13.22 | 7.86 | 1295.7 | 1.37 | 4.6 |
| Original | 5.62 | 7.85 | 534.3 | 1.00 | 5.2 |
| Zero-DCE | 19.59 | 6.54 | 1766.4 | 1.27 | 5.4 |
| U-Net V3 | 17.96 | 7.86 | 1190.9 | 1.29 | 5.6 |
| UniMIE | -0.73 | 7.85 | 725.5 | 0.93 | 6.0 |

**Key takeaways:**
- Best average rank (2.2) across all 5 metrics on 23 CheXpert test images.
- Highest Entropy (7.92) → retains the most diagnostically-relevant structural detail.
- Highest CII (1.90) → best contrast improvement, clearer anatomical boundaries.
- Stays in the top 2 methods on 4 of 5 metrics.
- Support devices (wires/tubes/leads) remain visible after enhancement — quality improves without erasing clinically relevant structural detail.

---

## Module 2: Disease Detection

**Objective:** Predict Cardiomegaly, Edema, and Pleural Effusion from chest X-rays, and compare architectures under one identical, leak-free data/evaluation protocol so gains are attributable to model design, not data variation.

### Shared Data Foundation (built once, reused by all techniques)

```
Raw CheXpert Data → Balanced Subsample (frontal only, per-disease sampling)
                  → Patient-Level Split (70/15/15, zero leakage)
                  → Shared Loaders (Train / Val / Test)
```

All four techniques train, validate, and test on the exact same images with thresholds tuned only on the validation split, so architecture is the only variable being compared.

### Techniques Compared

1. **CNN (Baseline Recipe)** — DenseNet121 → GAP → Linear(3). Plain BCE loss, no dropout, no scheduler (literal CheXNet recipe, literature-comparable reference).
2. **CNN (Matched Recipe)** — Same architecture as baseline, with Focal Loss + dropout (0.3) + LR scheduler; isolates the "recipe" effect.
3. **Hybrid CNN-Transformer** — DenseNet121 features → patch tokens + [CLS] → 2-layer Transformer encoder → classify off [CLS]. Uses the matched recipe.
4. **CheXGCN** — Image features + a GCN over the disease co-occurrence graph (built from the train split only), dot-product classifier. Uses the matched recipe.

Thresholds are tuned per disease on the validation set only, then frozen and applied to the test set for Precision/Recall/F1.

### Evaluation Across Data Scales (1k, 6k, 20k images per disease)

**Key takeaways:**
- **CheXGCN** shows the largest improvement with increased data — lowest AUC at 1k, tied-highest AUC on Cardiomegaly and Pleural Effusion at 20k, consistent with the graph-based co-occurrence signal needing more samples to be reliable.
- **Most performance gains occur between 1k and 6k images**, with a plateau (and occasional slight decrease) from 6k to 20k — diminishing returns beyond 6k.
- **Hybrid CNN-Transformer** needs more data to be competitive — lowest/near-lowest F1/AUC at 1k, consistent with higher data requirements of Transformer attention vs. the built-in spatial inductive bias of CNNs.
- **Relative rankings shift substantially with scale** — at 6k/20k, no single architecture dominates; Hybrid and CheXGCN each top at least one disease while the CNN baseline drops to 2nd/3rd on several occasions. Architecture choice matters less than data volume at this scale.

---

## Module 3: Explainability — Score-CAM

```
Conv Feature Maps → Upsample & Normalize → Mask Input & Forward Pass
                  → Weighted Sum + ReLU → Upsampled Heatmap
```

- **Why Score-CAM:** Gradient-free — instead of backpropagating gradients (as in Grad-CAM), each activation map is used directly as an image mask and scored via a forward pass, avoiding gradient noise/saturation in deep CNNs.
- **Target layer:** Final DenseNet121 conv block (`denseblock4` / `norm5`) — deep enough to encode disease-relevant spatial features, coarse enough to localize well after upsampling.
- **Independent per disease:** Run separately for each of the 3 diseases using that disease's own class score, so each heatmap is independently, spatially verifiable — not one blended saliency map.
- CheXGCN was selected for the clearest per-disease spatial differentiation, independent of raw AUC/F1 ranking.

---

## Module 4: Report Generation — MediVLM

**Objective:** Generate a radiology report from a chest X-ray using region-aware visual understanding combined with medical language representation.

The MediVLM model code is not in this repository. The notebooks clone the public implementation from [sonai-commits/MediVLM](https://github.com/sonai-commits/MediVLM), and the end-to-end pipeline (`Complete Pipeline/pipeline_core.py`) pins it to commit `f5a6e1d`. The architecture below is taken from that code.

### Architecture

```
Chest X-ray (224×224, values in [0, 1])
  → Faster R-CNN (ResNet-34 + FPN) → 8 highest-scoring boxes → 8 crops of 28×28
  → each crop resized to 224×224 → CLIP ViT-L/14 → 1024-d feature per crop
  + Position MLP over the normalized box (x, y, w, h) → 8 visual tokens
  → Projection head (512-d) → Cross-attention fusion → Linear (768-d) prefix → GPT-2 → Report
```

1. **Region detector** — torchvision Faster R-CNN with a ResNet-34 + FPN backbone, configured with `num_anatomical_classes: 29` for the MS-CXR anatomical labels. The 8 highest-scoring boxes are cropped and resized to 28×28 patches. If fewer boxes are found, the whole image fills the remaining slots.
2. **Image encoder** — each patch is resized to 224×224 and encoded by CLIP ViT-L/14 (pooled output, 1024-d). A 3-layer MLP turns the box's normalized (x, y, w, h) into a 1024-d position embedding, which is added to the CLIP feature.
3. **Text encoder (training only)** — ClinicalBERT (`medicalai/ClinicalBERT`, up to 128 tokens) encodes the reference report.
4. **Projection and contrastive alignment** — two MLP heads project the visual and text features into a shared 512-d space. A symmetric InfoNCE loss (τ = 0.07) aligns the mean visual token with the mean-pooled report embedding.
5. **Cross-attention fusion** — an 8-head attention block with the visual tokens as queries. During training, the keys and values are the ClinicalBERT report tokens. At inference there is no report, so the visual tokens attend to themselves.
6. **Decoder** — the fused tokens are projected to GPT-2's 768-d space and prepended as a prefix (soft prompt). GPT-2 generates the report with beam search: 4 beams, repetition penalty 1.2, no repeated 3-grams, and 77 new tokens by default.

> **Detector weights:** every config in this repo sets `detector.weights_path: null`, so no MS-CXR-trained detector is loaded. The ResNet-34 backbone starts from ImageNet weights, the detection heads are untrained, and the detector stays frozen. The 8 patches are therefore not guaranteed to be anatomical regions.

**Why this architecture (from the MediVLM paper):** selective salient patching improves performance over whole-image processing, and ClinicalBERT gives stronger text representations than the alternative CLIP text encoder.

### Training Setup

- **Loss:** 1.0 × cross-entropy over the report tokens + 0.7 × InfoNCE.
- **Optimizer:** AdamW, LR = 2 × 10⁻⁵, weight decay 1 × 10⁻⁴, 500 warm-up steps then cosine decay (floor 5%), gradient clipping 1.0, mixed precision.
- **Frozen:** the detector, CLIP, ClinicalBERT, and GPT-2's position embeddings and first 8 blocks.
- **Trained:** the position MLP, projection heads, fusion block, prefix projection, and GPT-2's last 4 blocks, final layer norm and LM head (tied to the token embeddings). That is 72,272,896 of 606,206,176 parameters.
- **Checkpoint selection:** `best.ckpt` is the epoch with the highest validation BLEU-4.

### Data

- **Source:** a Kaggle export of MIMIC-CXR (`mimic_cxr_aug_train.csv`, `mimic_cxr_aug_validate.csv`, images under `official_data_iccv_final/`), unzipped from Google Drive. Each training notebook builds `annotations.json` from the CSVs: 64,586 training rows. The validation CSV is used for both `val` and `test`, so test scores are not independent of validation.
- **Images:** only the first image listed for each report is used. Rows whose image is missing from the zip are skipped; in the 1-epoch run, 45,224 of 64,586 training rows and 338 of 500 validation rows remain.
- **Reports:** reports are lower-cased, de-identification placeholders become `xxxx`, and each report is cut to 4 sentences. The report text in the CSV is a stringified Python list (`['findings: ...', ...]`), and the model learns to write in that format.

The MediVLM paper trains on the full MIMIC-CXR split (369K / 3.0K / 5.2K images) for 30–50 epochs at batch size 32. The runs in this repo are much smaller.

### Experiments in This Repo

| Notebook | Training data | Epochs | Status |
|---|---|---|---|
| `MediVLM_MIMIC_Sample_Train.ipynb` | First 20,000 training rows with images, batch 32 | 3 | Run. Its `best.ckpt` (epoch 1) is the `mimic_sample.ckpt` used by the pipeline: the checkpoint's stored metrics match epoch 1. |
| `MediVLM_MIMIC_1_Epoch.ipynb` | All 45,224 training rows with images, batch 32 | 1 | Run |
| `MediVLM_MIMIC_Full_Training.ipynb` | Full `annotations.json`, every component unfrozen, batch 8 | 20 | Config only. Never run (no saved outputs). |
| `qwen_mimic_sample.ipynb` | All training rows with images, batch 2 | 1 | Experimental decoder. No saved outputs. |

The Qwen experiment replaces GPT-2 with Qwen2.5-3B-Instruct, loaded in 4-bit NF4 with LoRA (r = 8, α = 16, on `q_proj` / `v_proj`). It also replaces ClinicalBERT with DistilBERT (`distilbert-base-uncased`), allows up to 150 tokens, and trains with its own loop.

### Results

The MediVLM trainer computes BLEU-1–4 (NLTK corpus BLEU), METEOR (NLTK) and ROUGE-L (`rouge-score` F1). The test split is the validation split in every run.

| Run | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L |
|---|---|---|---|---|---|---|
| 20k sample, epoch 1, val (`mimic_sample.ckpt`) | 0.3752 | 0.2510 | 0.1820 | 0.1316 | 0.3792 | 0.2624 |
| 20k sample, epoch 3, test | 0.3631 | 0.2401 | 0.1707 | 0.1228 | 0.3583 | 0.2535 |
| 45,224 rows, 1 epoch, test | 0.3554 | 0.2340 | 0.1658 | 0.1192 | 0.3500 | 0.2464 |

The Qwen notebook has no saved outputs and no evaluation step. TEAM_95_PR2 (slide 25) reports BLEU-1 0.2111, BLEU-2 0.1444, BLEU-3 0.1061, BLEU-4 0.0784, METEOR 0.3456, ROUGE-L 0.2237 and BERTScore 0.7864 for it, but these numbers cannot be reproduced from this repo.

The MediVLM repo also implements BERTScore, RadGraph-F1, RaTEScore and a TF-IDF severity score (`scripts/evaluate.py`), but no notebook here runs them.

### Inference in the ReXTrust Pipeline

`generate_report` in `Complete Pipeline/pipeline_core.py`:

- **Input:** the enhanced 256×256 grayscale X-ray, converted to RGB and resized to 224×224. The disease-detection labels are **not** passed to MediVLM; `report_vs_detection` compares the report with the detector afterwards.
- **Loading:** the model config is read from the checkpoint. The Hugging Face sub-models are built from their configs only, because every weight comes from `mimic_sample.ckpt`. The state dict is loaded strictly, with CLIP keys remapped for transformers ≥ 5.
- **Generation:** 4 beams and up to 150 new tokens, because the checkpoint's 77-token limit cuts reports off mid-sentence.
- **Cleaning:** `clean_report` keeps only the first report from the stringified list and splits it into FINDINGS and IMPRESSION. It removes comparison clauses that refer to a prior study, and drops sentences containing `xxxx`, change-only sentences ("no change", "stable") and sentences cut off by the token limit. The cleaned report goes to RadCheck RAG, and the raw decoder output is kept alongside it.

### Files (`MediVLM/`)

| File | Purpose |
|---|---|
| `MediVLM_MIMIC_Sample_Train.ipynb` | 20k-sample training run (Colab) |
| `MediVLM_MIMIC_1_Epoch.ipynb` | 1-epoch run on all available rows; patches the dataset class to skip missing images |
| `MediVLM_MIMIC_Full_Training.ipynb` | Fully unfrozen 20-epoch config (not run) |
| `qwen_mimic_sample.ipynb` | Qwen2.5-3B + LoRA decoder experiment |
| `MediVLM_Colab_Inference.ipynb` | Single-image inference from a saved checkpoint |
| `app.py` | Gradio app: MediVLM report, then AMG-RAG check (run from inside a MediVLM clone) |
| `rag_module_source.py` | AMG-RAG module used by `app.py` |
| `requirements.txt` | MediVLM dependencies |

---

## Module 5: RadCheck RAG — Clinical Hallucination Detection

**Why it's needed:** LLM-generated radiology reports can hallucinate findings not actually present in the image — sounding fluent but being clinically wrong. RadCheck RAG detects this.

**Core idea:** Hallucination detection combines external evidence (is the finding medically documented?) with internal consistency checks (does it contradict another finding in the same report?).

### Previous Architecture

```
Report → Clinical Claims Extraction → Construct MKG (Medical Knowledge Graph)
       → DFS Traversal + Confidence Propagation → PubMed Grounding → Trust Score → Final Verdict
```

**Drawbacks:**
- No explicit contradiction detection — only confidence propagation via BFS
- PubMed grounding applied uniformly rather than selectively
- No bidirectional relationship modeling between claims and KG nodes
- Lacked a structured typical-vs-atypical decision path, increasing unnecessary computation

### Current Architecture

```
Report → Clinical Claims Extraction → LLM Baseline Check (typical / atypical)
       → [if atypical] PubMed Grounding → MKG Contradiction Engine
       → Final Verdict: Hallucinated / Not Hallucinated (with confidence score)
```

**Pipeline stages:**
1. **Clinical Claims Extraction** — extracts discrete claims (target + status) from the report.
2. **LLM Baseline Check** — flags each claim as typical/atypical.
3. **PubMed Grounding** — invoked only for atypical findings, reducing unnecessary computation.
4. **MKG Contradiction Engine** — builds a Medical Knowledge Graph from the report's claims and runs contradiction detection.
5. **Final Verdict** — per-claim PASS/FAIL with confidence score, plus an overall report-level verdict (e.g., "REPORT IS CLINICALLY CONSISTENT").

### Architecture Comparison (System A vs. System B)

| Metric | System A | System B |
|---|---|---|
| Accuracy | 86.96% | 73.91% |
| Precision | 91.67% | 76.92% |
| Recall | 84.62% | 76.92% |
| F1-score | 88.00% | 76.92% |
| Specificity | 90.00% | 70.00% |

*Note: Evaluation was performed on a curated set of manually labeled clinical statements rather than raw dataset reports, since MIMIC-CXR/IU-Xray reports have no ground-truth contradiction/hallucination labels needed to compute precision, recall, and F1.*

---

## Technical Enrichment — Courses (Ongoing)

- **AI for Medical Diagnosis** — Coursera
- **Generative AI with LLMs** — Coursera
- **End-to-End Multimodal AI: Fine-Tuning, Fusion & MLOps** — Coursera

---

## References

1. **UniMIE:** Fei et al., *Communications Medicine*, 2025.
2. **Zero-DCE:** Guo et al., *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
3. **U-Net:** Ronneberger et al., *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015.
4. **MediVLM:** MediVLM: A Vision Language Model for Radiology Report Generation from Medical Images.
5. **Agentic MKG:** Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge.
6. **CheXpert:** Irvin et al., *AAAI Conference on Artificial Intelligence*, 2019.
7. **Score-CAM:** Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks.

---

## Project Status

- Image Enhancement, Disease Detection, Explainability (Score-CAM), MediVLM report generation, and RadCheck RAG grounding modules have each been developed and evaluated individually.
- Current focus: integrating the Image Enhancement module into the complete end-to-end pipeline and finalizing the shared evaluation protocol across all Disease Detection techniques.
