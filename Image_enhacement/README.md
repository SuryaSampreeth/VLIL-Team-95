
# ReXTrust: Chest X-ray Image Enhancement

**Conditional, Attention-Based Enhancement of Chest Radiographs: A Comparative Study of Classical, Learned, and Hybrid Methods**

**Technologies:** Python · PyTorch · OpenCV · NumPy · scikit-image · Jupyter Notebook

**Project:** ReXTrust — VLIL Team 95  
**Institution:** Amrita Vishwa Vidyapeetham, Department of Computer Science and Engineering

---

## Table of Contents

1. [Abstract](#abstract)
2. [Project Overview](#project-overview)
3. [Problem Statement](#problem-statement)
4. [Objectives](#objectives)
5. [Dataset](#dataset)
6. [Enhancement Methods](#enhancement-methods)
7. [Proposed Method](#proposed-method-attention-u-net--clahe)
8. [Training Configuration](#training-configuration)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Experimental Results](#experimental-results)
11. [Discussion](#discussion)
12. [Limitations and Data Leakage Audit](#limitations-and-data-leakage-audit)
13. [Repository Structure](#repository-structure)
14. [Installation](#installation)
15. [Execution](#execution)
16. [Reproducibility](#reproducibility)
17. [Future Work](#future-work)
18. [Responsible Use](#responsible-use)
19. [Team and Acknowledgements](#team-and-acknowledgements)
20. [References](#references)

---

## Abstract

Chest X-ray images can suffer from low contrast, noise, and uneven illumination, potentially reducing the visibility of anatomical structures and affecting downstream medical-image analysis.

This module of **ReXTrust** investigates classical, unsupervised, diffusion-based, supervised, attention-based, and hybrid image enhancement techniques using the CheXpert chest radiograph dataset.

The evaluated approaches include CLAHE, Zero-DCE, UniMIE, U-Net V3, Attention U-Net with CBAM, Retinexformer-Lite, and combinations of neural-network enhancement with CLAHE.

The proposed **Attention U-Net + CLAHE** pipeline combines learned image restoration with local contrast enhancement. The Attention U-Net uses residual blocks, a Convolutional Block Attention Module (CBAM), and attention-gated skip connections to refine image features.

Enhancement quality is assessed using BRISQUE, Shannon entropy, Laplacian-variance sharpness, Contrast Improvement Index (CII), and Enhancement Measure Evaluation (EME).

In the reported evaluation of 23 selected low-quality images, Attention U-Net + CLAHE achieved an average rank of **2.2 across eight methods**, the highest mean entropy of **7.92**, and the highest mean CII of **1.90**.

These findings are preliminary. The evaluation subset was selected using two of the metrics later used for evaluation, and the experiments have identified potential training-evaluation overlap. The reported results therefore require confirmation on a fixed, independent test set.

## Project Overview

ReXTrust is an explainable chest X-ray analysis project that investigates image enhancement as an initial stage in a larger medical-image analysis pipeline.

This repository focuses on improving image visibility while attempting to preserve fine anatomical structures and medical devices that may be important for downstream disease detection.

The image enhancement module investigates three key challenges:

- Improving local contrast and visibility.
- Reducing degradation without excessively smoothing important structures.
- Evaluating image quality using multiple complementary metrics.

### Role in the ReXTrust Pipeline

| Stage | Component | Scope |
|---|---|---|
| 1 | Image quality assessment | BRISQUE and entropy-based screening |
| 2 | Image enhancement | Attention U-Net + CLAHE and comparative methods |
| 3 | Disease detection | Separate `DiseaseDetection/` module |
| 4 | Per-disease saliency | Score-CAM pipeline stage |
| 5 | Report generation and verification | RadCheck pipeline stage |

This repository primarily implements and evaluates the first two stages.

## Problem Statement

Chest X-ray images may contain low contrast, noise, and uneven illumination. Direct contrast enhancement can amplify noise, while learned restoration networks may blur edges or modify subtle image details.

The objective is to develop and compare enhancement methods that balance:

- Image clarity.
- Local contrast.
- Sharpness.
- Information content.
- Preservation of fine structures.

The project also investigates whether combining neural-network restoration with CLAHE improves image quality compared with either technique alone.

## Objectives

- Implement and compare classical and deep-learning-based enhancement techniques.
- Develop an attention-based U-Net using residual blocks, CBAM, and attention gates.
- Investigate hybrid enhancement pipelines that combine learned restoration and CLAHE.
- Evaluate image quality using five complementary metrics.
- Compare methods using average metric rankings.
- Investigate preservation of support-device edges.
- Examine the computational cost of different architectures.
- Identify data leakage and evaluation biases before drawing conclusions.

## Dataset

The project uses the **CheXpert** chest radiograph dataset.

The notebook loads image metadata from `train.csv` and optionally references `valid.csv`.

### Dataset Configuration

```python
BASE_PATH = "/kaggle/input/datasets/ashery/chexpert"

TRAIN_CSV = f"{BASE_PATH}/train.csv"
VALID_CSV = f"{BASE_PATH}/valid.csv"
```

Update `BASE_PATH` according to the location of the dataset in your environment.

### Image Preprocessing

The documented preprocessing workflow includes:

1. Loading image paths and metadata.
2. Converting radiographs to grayscale.
3. Resizing images to 256 × 256 pixels.
4. Replicating grayscale images across three channels for neural-network input.
5. Applying enhancement and degradation operations as required by each experiment.

### Experimental Subsets

| Subset | Number of Images | Purpose |
|---|---:|---|
| Random sample | 50 | Initial image-quality screening |
| Flagged low-quality subset | 23 | Multi-image comparison |
| Single-image comparison | 1 | Detailed comparison of ten configurations |
| Support-device example | 1 | Fine-edge preservation assessment |

The 23-image subset was selected using BRISQUE and entropy thresholds. This introduces potential selection bias because these metrics were subsequently used to evaluate enhancement quality.

### Training Data

| Model | Training Images | Training / Validation Split |
|---|---:|---|
| Zero-DCE | 2,000 | Training pool |
| U-Net V3 | 2,000 | 1,900 / 100 |
| Attention U-Net | 500 | 400 / 100 |
| Retinexformer-Lite | 1,999 | 1,799 / 200 |

These counts describe the documented experiments and should be verified against the notebook configuration used for a final reproduction.

## Enhancement Methods

The project evaluates the following methods and configurations.

| Method | Category | Main Approach |
|---|---|---|
| Original | Reference | Unprocessed chest X-ray |
| CLAHE | Classical | Local histogram equalization |
| UniMIE | Diffusion-based | Diffusion-model image enhancement |
| Zero-DCE | Unsupervised deep learning | Pixel-wise enhancement curve estimation |
| U-Net V3 | Supervised deep learning | Encoder-decoder restoration network |
| U-Net V3 + CLAHE | Hybrid | Neural restoration followed by CLAHE |
| Attention U-Net | Attention-based deep learning | Residual blocks, CBAM, and attention gates |
| Attention U-Net + CLAHE | Proposed hybrid | Attention-based restoration followed by CLAHE |
| Retinexformer-Lite | Transformer-inspired | Illumination-guided attention |
| Retinexformer-Lite + CLAHE | Hybrid | Illumination-aware restoration followed by CLAHE |

### 1. CLAHE

Contrast Limited Adaptive Histogram Equalization enhances local contrast by equalizing intensity distributions within image tiles while limiting excessive contrast amplification.

**Advantages:**
- Does not require model training.
- Computationally efficient.
- Improves local contrast and visibility.

**Limitations:**
- May amplify noise.
- Can produce excessive local contrast.
- Does not perform learned image restoration.

### 2. UniMIE

UniMIE is used for diffusion-based enhancement experiments with the guided-diffusion implementation and a pretrained 256 × 256 unconditional diffusion checkpoint.

Diffusion-based enhancement can produce visually different outputs, but the generative process may alter fine image details.

The documented inference time is approximately 35–40 seconds per image on a Tesla T4 GPU.

### 3. Zero-DCE

Zero-DCE estimates pixel-wise enhancement curves using a lightweight convolutional network.

The implementation trains without paired clean-target images and uses loss components associated with:

- Spatial consistency.
- Exposure control.
- Color constancy.
- Illumination smoothness.

The experiment uses 2,000 CheXpert images. Its exposure and regularization settings were adjusted to reduce excessive brightening and preserve image information.

### 4. U-Net V3

U-Net V3 uses an encoder-decoder architecture with skip connections.

The model learns to map synthetically degraded inputs toward CLAHE-generated target images.

The training loss combines:

- L1 reconstruction loss.
- SSIM-based structural loss.
- Laplacian edge loss.

Because CLAHE generates the target images, the model learns to approximate that target-generation procedure. These targets are not verified clinically ideal images.

### 5. U-Net V3 + CLAHE

This hybrid method first applies the trained U-Net V3 model and then applies CLAHE to improve local contrast.

The objective is to reduce some noise before increasing contrast.

### 6. Attention U-Net with CBAM

The attention-based U-Net combines residual feature extraction with channel and spatial attention.

Its main components are:

- Four encoder stages.
- Residual blocks.
- CBAM at the bottleneck.
- Attention gates on skip connections.
- Four decoder stages.
- A final convolution and sigmoid output.

This design attempts to select relevant feature channels and spatial regions while reconstructing the enhanced image.

### 7. Retinexformer-Lite

Retinexformer-Lite is a simplified Retinexformer-inspired architecture that incorporates Illumination-Guided Multi-head Self-Attention (IG-MSA).

An illumination estimator predicts a light map, which is used to modulate feature processing so that bright and dark regions can be treated differently.

This implementation is an adaptation and should not be considered an exact reproduction of the original Retinexformer architecture.

## Proposed Method: Attention U-Net + CLAHE

The proposed hybrid method combines learned image restoration and classical contrast enhancement.

### Architecture

| Component | Configuration |
|---|---|
| Input | Grayscale chest X-ray replicated to three channels |
| Input resolution | 256 × 256 |
| Encoder | Four residual feature-extraction stages |
| Encoder channels | 32, 64, 128, 256 |
| Bottleneck | 512 channels with CBAM |
| Decoder | Four attention-gated reconstruction stages |
| Output | Three-channel enhanced image |
| Post-processing | CLAHE |
| Parameters | 8,235,421 |
| Model size | Approximately 31.42 MB in float32 |

### Processing Pipeline

```text
Input Chest X-ray
        |
        v
Grayscale Conversion and Resizing
        |
        v
Residual Encoder
        |
        v
CBAM Bottleneck
        |
        v
Attention-Gated Decoder
        |
        v
Reconstructed Image
        |
        v
CLAHE Post-Processing
        |
        v
Final Enhanced Image
        |
        v
Image Quality Evaluation
```

### Component Roles

| Component | Function |
|---|---|
| Residual blocks | Learn feature corrections and support optimization |
| Channel attention | Reweight feature channels |
| Spatial attention | Emphasize relevant spatial regions |
| Attention gates | Filter skip-connection features using decoder information |
| CLAHE | Improve local contrast in the reconstructed image |

The network uses residual blocks to extract features, CBAM to refine bottleneck representations, and attention gates to filter encoder features before they are combined with decoder features.

CLAHE is applied after neural restoration to improve local contrast.

## Training Configuration

The supervised models use synthetic degradation to generate training inputs.

### Training Pair Generation

```text
Original CheXpert Image
          |
          +--------------------------+
          |                          |
          v                          v
CLAHE Target Image             Synthetic Degradation
                               Gaussian Noise + Blur
          |                          |
          |                          v
          |                    Neural Network
          |                          |
          +------------+-------------+
                       |
                       v
                 Training Loss
```

### Synthetic Degradation

The documented configuration includes:

- Gaussian noise with standard deviation sampled from 8 to 25.
- Gaussian blur kernel sizes of 3 or 5.
- Blur standard deviation sampled from 0.3 to 1.2.
- CLAHE-generated targets with clip limit 3.0 and an 8 × 8 tile grid.

### Model Training Configuration

| Setting | U-Net V3 | Attention U-Net | Retinexformer-Lite |
|---|---|---|---|
| Input resolution | 256 × 256 | 256 × 256 | 256 × 256 |
| Optimizer | Adam | Adam | Adam |
| Learning rate | 3 × 10⁻⁴ | 3 × 10⁻⁴ | 3 × 10⁻⁴ |
| Loss | L1 + SSIM + Laplacian | L1 + SSIM + Laplacian | Same + illumination smoothness |
| Maximum epochs | 60 | 60 | 60 |
| Batch size | 8 | 8 | 8 |
| Training images | 1,900 | 400 | 1,799 |
| Validation images | 100 | 100 | 200 |
| Learning-rate scheduling | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |

The reported hardware is a Tesla T4 GPU with approximately 15.6 GB of memory.

## Evaluation Metrics

The project uses five complementary image-quality metrics.

| Metric | Preferred Direction | Purpose |
|---|---|---|
| BRISQUE | Lower | Estimates perceptual image quality |
| Shannon Entropy | Higher, with caution | Measures intensity information content |
| Sharpness | Higher, with caution | Measures high-frequency image detail |
| CII | Higher | Measures contrast improvement relative to the original |
| EME | Higher, with caution | Measures local contrast characteristics |

### BRISQUE

BRISQUE is a no-reference image-quality metric. Lower values conventionally indicate better perceptual quality.

However, it was developed around natural-image statistics, and its reliability on medical radiographs is limited.

### Shannon Entropy

Entropy measures the information content of the image intensity distribution.

Higher entropy does not necessarily mean better image quality because noise can also increase entropy.

### Sharpness

Sharpness is estimated using the variance of the Laplacian.

Higher values can indicate stronger edges, but they can also result from noise amplification or artificial sharpening.

### Contrast Improvement Index

CII measures contrast improvement relative to the original image. The original image has a CII of 1.00 by construction.

### Enhancement Measure Evaluation

EME evaluates local contrast characteristics across image blocks.

A high EME value alone does not guarantee that anatomical structures are preserved.

### Average Rank

Each method is ranked for each metric, respecting whether higher or lower values are preferred. The rankings are aggregated to obtain an average rank.

Lower average rank indicates better aggregate ranking under the selected metrics.

## Experimental Results

### Multi-Image Evaluation

The following table reports the documented results for 23 selected low-quality images across eight methods.

| Rank | Method | BRISQUE ↓ | Entropy ↑ | Sharpness ↑ | CII ↑ | EME ↑ | Average Rank |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Attention U-Net + CLAHE** | 13.35 | **7.92** | 2082.26 | **1.90** | 11.10 | **2.2** |
| 2 | U-Net V3 + CLAHE | 17.51 | 7.91 | 1859.65 | 1.76 | 10.76 | 3.2 |
| 3 | CLAHE | 7.01 | 7.84 | **3609.61** | 1.71 | 8.65 | 3.8 |
| 4 | Attention U-Net | 13.22 | 7.86 | 1295.71 | 1.37 | 7.90 | 4.6 |
| 5 | Original | 5.62 | 7.85 | 534.33 | 1.00 | 9.54 | 5.2 |
| 6 | Zero-DCE | 19.59 | 6.54 | 1766.44 | 1.27 | **12.16** | 5.4 |
| 7 | U-Net V3 | 17.96 | 7.86 | 1190.87 | 1.29 | 7.71 | 5.6 |
| 8 | UniMIE | **-0.81** | 7.85 | 725.65 | 0.93 | 5.44 | 6.0 |

Retinexformer-Lite and its CLAHE hybrid were added after this multi-image run and were evaluated separately on a single image.

### Results Interpretation

- Attention U-Net + CLAHE achieved the best average rank of 2.2.
- The proposed method achieved the highest mean entropy and CII.
- CLAHE achieved the highest sharpness score, which may partly reflect stronger edges or noise amplification.
- Zero-DCE achieved the highest EME but also the lowest entropy.
- UniMIE achieved the lowest reported BRISQUE score, but its CII was below 1.0.
- The original images outperformed several enhancement methods on BRISQUE, demonstrating that no single metric is sufficient to determine enhancement quality.

### Single-Image Comparison

The notebook reports the following results for `view2_lateral.jpg`.

| Rank | Method | BRISQUE ↓ | Entropy ↑ | Sharpness ↑ | CII ↑ | EME ↑ | Average Rank |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Attention U-Net + CLAHE** | 1.50 | 7.37 | 1621.70 | 1.79 | 8.46 | 1.8 |
| 2 | U-Net V3 + CLAHE | 22.89 | 7.29 | 821.18 | 1.37 | 7.57 | 4.4 |
| 3 | Retinexformer-Lite + CLAHE | 21.62 | 7.33 | 593.43 | 1.39 | 7.55 | 4.4 |
| 4 | CLAHE | 6.72 | 6.85 | 2212.86 | 1.28 | 5.73 | 4.8 |
| 5 | Zero-DCE | 14.74 | 5.43 | 876.06 | 1.16 | 21.41 | 5.0 |
| 6 | Attention U-Net | 8.50 | 7.05 | 777.61 | 1.24 | 5.98 | 5.2 |
| 7 | U-Net V3 | 33.58 | 7.12 | 491.36 | 1.07 | 6.23 | 6.8 |
| 8 | UniMIE | 1.01 | 7.00 | 366.94 | 0.88 | 4.34 | 7.0 |
| 9 | Original | 19.22 | 6.79 | 285.95 | 1.00 | 18.38 | 7.2 |
| 10 | Retinexformer-Lite | 28.28 | 6.93 | 297.79 | 1.01 | 5.66 | 8.4 |

The single-image results are illustrative and should not be treated as evidence of general performance.

### Statistical Analysis

The notebook compares Attention U-Net + CLAHE against U-Net V3 + CLAHE using a paired Wilcoxon signed-rank test over 23 image pairs.

| Metric | Reported p-value |
|---|---:|
| BRISQUE | 0.0003 |
| Entropy | 0.0001 |
| Sharpness | 0.0049 |
| CII | < 0.0001 |
| EME | 0.0001 |

These p-values are uncorrected for multiple comparisons and inherit the dataset-selection and potential leakage issues discussed below. They should not be interpreted as proof of clinical benefit.

### Artifact Preservation

The project evaluates edge retention on one CheXpert image labelled with support devices.

| Method | Fine-Edge Retention |
|---|---:|
| CLAHE | 98.6% |
| Attention U-Net | 99.1% |
| Attention U-Net + CLAHE | 98.9% |

This measurement estimates how many original edges remain detectable after enhancement. It does not penalize newly introduced edges and does not establish semantic preservation of every anatomical structure or medical device.

## Discussion

The recorded experiments suggest that combining neural restoration with CLAHE can improve the aggregate ranking of image-quality metrics.

Attention U-Net + CLAHE outperformed Attention U-Net alone and CLAHE alone in average rank on the selected subset. U-Net V3 + CLAHE also achieved a better average rank than U-Net V3 alone.

However, the results do not show that the proposed method is superior for every metric. CLAHE produced the highest sharpness, Zero-DCE produced the highest EME, and UniMIE produced the lowest reported BRISQUE value.

The main findings are:

1. Hybrid enhancement can provide a useful balance between learned restoration and local contrast enhancement.
2. Attention mechanisms may help feature selection and reconstruction, but controlled comparisons are needed.
3. Image-quality metrics measure different properties and can disagree.
4. Edge retention is a useful preliminary check but is not a complete measure of clinical information preservation.
5. Evaluation leakage and subset selection must be addressed before drawing strong conclusions.

## Limitations and Data Leakage Audit

The reported experiments have several limitations that must be considered when interpreting their results.

### Evaluation Overlap

| Model | Training Configuration | Reported Overlap Status |
|---|---|---|
| U-Net V3 | 1,900 training images and 100 validation images | 18 of 23 evaluation images overlap with the training pool |
| Attention U-Net | 500-image pool, split into 400 training and 100 validation images | Potential overlap with all 23 evaluation images |
| Retinexformer-Lite | 1,799 training and 200 validation images | Notebook reports a leakage guard |
| Zero-DCE | 2,000 training images | Overlap not fully verified |
| CLAHE | No model training | Not applicable |
| UniMIE | Pretrained diffusion checkpoint | No project-specific training on the evaluation subset |

The Attention U-Net training pool and the evaluation subset were generated using related sampling procedures and the same random seed. This creates a substantial risk that the evaluation images were included in training or validation.

The Retinexformer-Lite notebook includes a leakage guard, but all model path lists should be independently checked before claiming that the entire evaluation is leakage-free.

### Selection Bias

The 23 evaluation images were selected using BRISQUE and entropy thresholds, and these metrics were then used in the evaluation.

This creates a selection bias because the evaluation set was chosen partly on the basis of the same quantities used to assess enhancement.

### Other Limitations

- The multi-image evaluation contains only 23 selected images.
- The dataset includes frontal and lateral views.
- No clinically ideal reference images are available for direct comparison.
- BRISQUE may not reliably represent medical-image quality.
- Higher sharpness and entropy may reflect noise or artifacts.
- Supervised models learn from CLAHE-generated targets rather than verified ideal images.
- The edge-preservation experiment uses a single support-device image.
- Statistical comparisons are not corrected for multiple testing.
- Retinexformer-Lite is absent from the eight-method multi-image comparison.
- Downstream disease-detection performance and radiologist assessments have not been established.

### Recommended Final Evaluation Protocol

1. Define a fixed, independent test set before training the enhancement models.
2. Split data at the patient level where patient identifiers are available.
3. Exclude every test patient from training and validation.
4. Check image-path and patient-level overlap for all learned models.
5. Select evaluation images independently of the metrics used to compare methods.
6. Evaluate all methods on the same fixed test set.
7. Apply appropriate multiple-comparison corrections when conducting statistical tests.
8. Investigate downstream disease-detection performance with and without enhancement.
9. Obtain expert assessment before making claims about clinical usefulness.

## Repository Structure

```text
Image_enhancement/
|
|-- results/
|   |-- artifact_highlighted_full.png
|   |-- comparision_table (2).png
|   |-- five_way_comparison_v3.png
|   |-- four_way_comparison_clean.png
|   |-- multi_image_eval_barchart.png
|   |-- quality_distribution.png
|   |-- single_image_comparison.png
|   |-- texture_blurring_verification.png
|   |-- unet_xray_v3.pt
|   |-- zerodce_training_loss.png
|   |-- zerodce_xray.pt
|
|-- README.md
|-- final_image_enhacement_3000.ipynb
|-- final-image-enhancement-with-attention.ipynb
|-- final-image-enhancement.ipynb
|-- final_image_enhancement_23_images.ipynb
|-- image_enhancement_stage_1.ipynb
|-- image_enhancement_unimie.ipynb
|-- unet_xray_v3.pt
|-- zerodce_xray.pt
```

### Main Files

| File | Purpose |
|---|---|
| `final_image_enhacement_3000.ipynb` | Consolidated experiments and comparative evaluation |
| `final-image-enhancement-with-attention.ipynb` | Attention U-Net and hybrid enhancement experiments |
| `final-image-enhancement.ipynb` | Enhancement pipeline experiments |
| `final_image_enhancement_23_images.ipynb` | Multi-image and texture-preservation evaluation |
| `image_enhancement_stage_1.ipynb` | Initial enhancement workflow |
| `image_enhancement_unimie.ipynb` | Diffusion-based enhancement experiments |
| `unet_xray_v3.pt` | Saved U-Net V3 weights |
| `zerodce_xray.pt` | Saved Zero-DCE weights |
| `results/` | Saved evaluation outputs, plots, and model assets |

Some files may exist both in the project root and inside `results/`. Update this listing if the repository structure changes.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version.
- Access to the CheXpert dataset under its usage terms.
- Jupyter Notebook or Kaggle Notebooks.
- A CUDA-capable GPU is recommended for training and diffusion experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/SuryaSampreeth/VLIL-Team-95.git
cd VLIL-Team-95/Image_enhancement
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Core Dependencies

```bash
python -m pip install --upgrade pip

python -m pip install numpy pandas scipy scikit-image scikit-learn matplotlib seaborn opencv-python-headless Pillow tqdm brisque piq jupyter
```

Install PyTorch and Torchvision using the official installation selector:

https://pytorch.org/get-started/locally/

### 4. Optional UniMIE Dependencies

The diffusion experiments may require the guided-diffusion implementation and its checkpoint.

```bash
pip install git+https://github.com/openai/guided-diffusion.git

git clone https://github.com/Fayeben/UniMIE.git
```

The documented experiment uses a pretrained 256 × 256 diffusion checkpoint of approximately 2 GB.

Follow the checkpoint's official distribution and access instructions rather than assuming that it is included in this repository.

### 5. Configure the Dataset

Update `BASE_PATH` in the notebook to the correct CheXpert dataset directory.

The notebook resolves image paths from the metadata, so the dataset directory structure must match the expected paths.

**Note:** The repository does not include a verified, fully pinned dependency lockfile. Package versions may need adjustment for the selected Python, PyTorch, and CUDA versions.

## Execution

1. Clone the repository and install the required dependencies.
2. Open `final_image_enhacement_3000.ipynb` in Kaggle or Jupyter Notebook.
3. Attach the CheXpert dataset and configure `BASE_PATH`.
4. Run the setup and dataset-loading cells.
5. Execute the quality-assessment and enhancement cells in order.
6. Load saved model checkpoints where available to avoid unnecessary retraining.
7. Run the comparative evaluation cells to calculate metrics and generate result tables.
8. Review the statistical results and leakage checks before interpreting model performance.

Some experiments depend on variables, functions, and models created in earlier cells. Execute the notebook in sequence.

The documented 23-image comparison takes approximately 15 minutes on a Tesla T4 GPU, largely due to UniMIE inference. Actual execution time depends on the environment and configuration.

## Reproducibility

| Setting | Documented Value |
|---|---|
| Training seed | 42 |
| Evaluation seed | 123 |
| Sampling seed | 42 |
| Input resolution | 256 × 256 |
| Batch size | 8 for supervised models |
| Hardware | Tesla T4, approximately 15.6 GB |
| PyTorch | 2.10.0+cu128 |
| NumPy | 2.4.6 |

Random seeds improve repeatability but do not guarantee identical results across hardware and library versions.

For reproducible experiments, record the dataset version, patient split, checkpoint, model configuration, package versions, evaluation subset, and metric implementation.

## Future Work

- Re-evaluate all learned models on a fixed, leakage-free test set.
- Investigate patient-level separation across training, validation, and testing.
- Measure the impact of enhancement on downstream disease-detection performance.
- Conduct expert review of enhanced radiographs.
- Evaluate frontal and lateral radiographs separately.
- Investigate medical-domain quality metrics alongside BRISQUE.
- Improve support-device and anatomical-structure preservation assessment.
- Replace fixed quality-screening thresholds with validated quality-estimation methods.
- Add pinned dependencies and experiment tracking.
- Integrate the enhancement module into the complete ReXTrust pipeline and assess its effect on downstream results.

## Responsible Use

This project is intended for **research and educational purposes only**.

Enhanced images may contain artifacts or altered details. They must not replace the original radiograph or professional clinical interpretation.

The experiments do not establish clinical safety, diagnostic accuracy, or regulatory approval. Any clinical application would require independent validation, appropriate clinical oversight, and compliance with applicable regulations.

Use the CheXpert dataset according to its access and usage terms. Do not commit patient-identifying data, credentials, or access tokens to the repository.

## Team and Acknowledgements

**Team:** VLIL Team 95  
**Institution:** Amrita Vishwa Vidyapeetham, Department of Computer Science and Engineering

The project builds on the CheXpert dataset and open-source research in image enhancement, convolutional neural networks, attention mechanisms, and diffusion models.

We acknowledge the researchers and developers behind PyTorch, OpenCV, scikit-image, PIQ, guided-diffusion, and the methods discussed in this repository.

## References

1. Irvin, J. et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI, 2019.
2. Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." Graphics Gems IV, 1994.
3. Guo, C. et al. "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement." CVPR, 2020.
4. Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.
5. Oktay, O. et al. "Attention U-Net: Learning Where to Look for the Pancreas." MIDL, 2018.
6. Woo, S. et al. "CBAM: Convolutional Block Attention Module." ECCV, 2018.
7. He, K. et al. "Deep Residual Learning for Image Recognition." CVPR, 2016.
8. Cai, Y. et al. "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement." ICCV, 2023.
9. Dhariwal, P. and Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS, 2021.
10. Mittal, A., Moorthy, A. K., and Bovik, A. C. "No-Reference Image Quality Assessment in the Spatial Domain." IEEE Transactions on Image Processing, 2012.
11. Wang, Z. et al. "Image Quality Assessment: From Error Visibility to Structural Similarity." IEEE Transactions on Image Processing, 2004.
12. Agaian, S. S., Panetta, K., and Grigoryan, A. M. "Transform-Based Image Enhancement Algorithms with Performance Measure." IEEE Transactions on Image Processing, 2001.
13. Wilcoxon, F. "Individual Comparisons by Ranking Methods." Biometrics Bulletin, 1945.
14. UniMIE implementation: https://github.com/Fayeben/UniMIE

---

**ReXTrust — Chest X-ray Image Enhancement Module**

*Research and educational use only. Update this README whenever the implementation, checkpoints, evaluation protocol, or reported results change.*

