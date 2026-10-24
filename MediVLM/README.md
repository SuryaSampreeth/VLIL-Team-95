# MediVLM: Report Generation

**Objective:** Generate a radiology report from a chest X-ray using region-aware visual understanding combined with medical language representation.

The MediVLM model code is not in this repository. The notebooks clone the public implementation from [sonai-commits/MediVLM](https://github.com/sonai-commits/MediVLM), and the end-to-end pipeline (`Complete Pipeline/pipeline_core.py`) pins it to commit `f5a6e1d`. The architecture below is taken from that code.

## Architecture

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

## Training Setup

- **Loss:** 1.0 × cross-entropy over the report tokens + 0.7 × InfoNCE.
- **Optimizer:** AdamW, LR = 2 × 10⁻⁵, weight decay 1 × 10⁻⁴, 500 warm-up steps then cosine decay (floor 5%), gradient clipping 1.0, mixed precision.
- **Frozen:** the detector, CLIP, ClinicalBERT, and GPT-2's position embeddings and first 8 blocks.
- **Trained:** the position MLP, projection heads, fusion block, prefix projection, and GPT-2's last 4 blocks, final layer norm and LM head (tied to the token embeddings). That is 72,272,896 of 606,206,176 parameters.
- **Checkpoint selection:** `best.ckpt` is the epoch with the highest validation BLEU-4.

## Data

- **Source:** a Kaggle export of MIMIC-CXR (`mimic_cxr_aug_train.csv`, `mimic_cxr_aug_validate.csv`, images under `official_data_iccv_final/`), unzipped from Google Drive. Each training notebook builds `annotations.json` from the CSVs: 64,586 training rows. The validation CSV is used for both `val` and `test`, so test scores are not independent of validation.
- **Images:** only the first image listed for each report is used. Rows whose image is missing from the zip are skipped; in the 1-epoch run, 45,224 of 64,586 training rows and 338 of 500 validation rows remain.
- **Reports:** reports are lower-cased, de-identification placeholders become `xxxx`, and each report is cut to 4 sentences. The report text in the CSV is a stringified Python list (`['findings: ...', ...]`), and the model learns to write in that format.

The MediVLM paper trains on the full MIMIC-CXR split (369K / 3.0K / 5.2K images) for 30–50 epochs at batch size 32. The runs in this repo are much smaller.

## Experiments in This Repo

| Notebook | Training data | Epochs | Status |
|---|---|---|---|
| `MediVLM_MIMIC_Sample_Train.ipynb` | First 20,000 training rows with images, batch 32 | 3 | Run. Its `best.ckpt` (epoch 1) is the `mimic_sample.ckpt` used by the pipeline: the checkpoint's stored metrics match epoch 1. |
| `MediVLM_MIMIC_1_Epoch.ipynb` | All 45,224 training rows with images, batch 32 | 1 | Run |
| `MediVLM_MIMIC_Full_Training.ipynb` | Full `annotations.json`, every component unfrozen, batch 8 | 20 | Config only. Never run (no saved outputs). |
| `qwen_mimic_sample.ipynb` | All training rows with images, batch 2 | 1 | Experimental decoder. No saved outputs. |

The Qwen experiment replaces GPT-2 with Qwen2.5-3B-Instruct, loaded in 4-bit NF4 with LoRA (r = 8, α = 16, on `q_proj` / `v_proj`). It also replaces ClinicalBERT with DistilBERT (`distilbert-base-uncased`), allows up to 150 tokens, and trains with its own loop.

## Results

The MediVLM trainer computes BLEU-1–4 (NLTK corpus BLEU), METEOR (NLTK) and ROUGE-L (`rouge-score` F1). The test split is the validation split in every run.

| Run | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L |
|---|---|---|---|---|---|---|
| 20k sample, epoch 1, val (`mimic_sample.ckpt`) | 0.3752 | 0.2510 | 0.1820 | 0.1316 | 0.3792 | 0.2624 |
| 20k sample, epoch 3, test | 0.3631 | 0.2401 | 0.1707 | 0.1228 | 0.3583 | 0.2535 |
| 45,224 rows, 1 epoch, test | 0.3554 | 0.2340 | 0.1658 | 0.1192 | 0.3500 | 0.2464 |

The Qwen notebook has no saved outputs and no evaluation step. TEAM_95_PR2 (slide 25) reports BLEU-1 0.2111, BLEU-2 0.1444, BLEU-3 0.1061, BLEU-4 0.0784, METEOR 0.3456, ROUGE-L 0.2237 and BERTScore 0.7864 for it, but these numbers cannot be reproduced from this repo.

The MediVLM repo also implements BERTScore, RadGraph-F1, RaTEScore and a TF-IDF severity score (`scripts/evaluate.py`), but no notebook here runs them.

## Inference in the ReXTrust Pipeline

`generate_report` in `Complete Pipeline/pipeline_core.py`:

- **Input:** the enhanced 256×256 grayscale X-ray, converted to RGB and resized to 224×224. The disease-detection labels are **not** passed to MediVLM; `report_vs_detection` compares the report with the detector afterwards.
- **Loading:** the model config is read from the checkpoint. The Hugging Face sub-models are built from their configs only, because every weight comes from `mimic_sample.ckpt`. The state dict is loaded strictly, with CLIP keys remapped for transformers ≥ 5.
- **Generation:** 4 beams and up to 150 new tokens, because the checkpoint's 77-token limit cuts reports off mid-sentence.
- **Cleaning:** `clean_report` keeps only the first report from the stringified list and splits it into FINDINGS and IMPRESSION. It removes comparison clauses that refer to a prior study, and drops sentences containing `xxxx`, change-only sentences ("no change", "stable") and sentences cut off by the token limit. The cleaned report goes to RadCheck RAG, and the raw decoder output is kept alongside it.

## Files in This Folder

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
