
# ReXTrust — Chest X-ray Disease Detection

<p align="center">
  <strong>A Deep Learning Approach to Multi-Label Chest X-ray Classification</strong>
</p>

<p align="center">
  DenseNet121 · CNN–Transformer · CheXGCN
</p>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Experimental Results](#experimental-results)
- [Evaluation Metrics](#evaluation-metrics)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Execution](#execution)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Responsible Use](#responsible-use)

---

## Overview

**ReXTrust** is a deep-learning-based chest X-ray disease detection project that investigates different neural network architectures for multi-label classification of thoracic findings.

The project compares conventional convolutional neural networks, Transformer-based feature modelling, and graph-based label reasoning to investigate their effectiveness in identifying abnormalities from chest radiographs.

The Stage 2 experiments focus on three target findings:

- Cardiomegaly
- Edema
- Pleural Effusion

The project emphasizes controlled experimentation, patient-level data partitioning, validation-based threshold selection, and systematic evaluation using classification metrics.

## Problem Statement

Chest X-ray interpretation is an important part of thoracic disease assessment. Automated image analysis can assist research into identifying disease-associated patterns, but reliable classification requires careful model design and evaluation.

This project investigates whether different deep-learning architectures can improve multi-label chest X-ray classification and how image feature extraction, spatial attention, and relationships between disease labels influence predictive performance.

## Objectives

- Develop a reproducible chest X-ray classification pipeline.
- Implement and compare CNN-based, Transformer-based, and graph-based approaches.
- Investigate the effect of different training strategies.
- Prevent patient-level data leakage between training, validation, and test sets.
- Tune classification thresholds using validation data.
- Evaluate model performance using F1 score and ROC-AUC.
- Analyze classification errors using confusion matrices and ROC curves.
- Estimate metric uncertainty using bootstrap confidence intervals.

## Dataset

The experiments use the **CheXpert chest radiograph dataset**, which contains chest X-ray images and associated radiology labels.

### Target Findings

| Finding | Description |
|---|---|
| Cardiomegaly | A finding associated with an enlarged cardiac silhouette |
| Edema | A finding associated with pulmonary edema |
| Pleural Effusion | A finding associated with fluid accumulation in the pleural space |

### Data Preprocessing

The experimental pipeline includes:

1. Loading image metadata from `train.csv`.
2. Handling uncertain labels using the U-Zeros policy.
3. Replacing missing target labels with zero.
4. Filtering images to retain frontal-view radiographs.
5. Extracting patient identifiers from image paths.
6. Constructing a sample containing disease-positive examples and examples negative for all three target labels.
7. Splitting the dataset at the patient level.
8. Resizing images to `224 × 224`.
9. Applying ImageNet normalization for the pretrained DenseNet121 backbone.

The Stage 2 notebook reports 191,027 frontal images after label cleaning and a selected experimental sample of 71,811 images.

The final split contains:

| Partition | Number of Images |
|---|---:|
| Training | 50,403 |
| Validation | 10,653 |
| Test | 10,755 |
| **Total** | **71,811** |

The selected sample is not fully balanced across all possible label combinations. Consequently, performance should be interpreted in the context of the sampling strategy and label distribution.

## Methodology

The project follows a common experimental pipeline for comparing the four model configurations.

```text
CheXpert Dataset
       |
       v
Metadata Loading and Label Cleaning
       |
       v
Frontal Image Filtering
       |
       v
Experimental Sample Construction
       |
       v
Patient-Level Train / Validation / Test Split
       |
       v
Image Preprocessing and Data Loaders
       |
       +------------------+------------------+------------------+
       |                  |                  |                  |
       v                  v                  v                  v
   Baseline CNN      Matched CNN       CNN-Transformer       CheXGCN
       |                  |                  |                  |
       +------------------+------------------+------------------+
                                  |
                                  v
                     Validation Threshold Selection
                                  |
                                  v
                      Held-Out Test Evaluation
                                  |
                                  v
                    Comparative Performance Analysis
```

### Experimental Workflow

1. Load and preprocess the dataset.
2. Construct patient-level data partitions.
3. Create shared data loaders.
4. Train each model using its designated training configuration.
5. Select the best checkpoint using validation performance.
6. Tune per-disease classification thresholds on validation predictions.
7. Evaluate the selected models on the held-out test set.
8. Compare per-disease F1 scores and ROC-AUC values.
9. Examine confusion matrices, ROC curves, and bootstrap confidence intervals.

## Model Architectures

### 1. Baseline CNN — DenseNet121

The baseline model uses an ImageNet-pretrained DenseNet121 backbone to extract visual features from chest X-ray images.

The extracted features are passed through global average pooling and a linear classification layer that produces three output logits.

**Training configuration:**
- Loss: Binary Cross-Entropy with Logits
- Optimizer: Adam
- Classification: Multi-label prediction

This model serves as the baseline for evaluating the alternative architectures and training strategies.

### 2. Matched-Recipe CNN

The matched CNN retains the DenseNet121-based classification architecture while using a different training configuration.

**Training configuration:**
- Loss: Focal loss
- Dropout: `0.3`
- Learning-rate scheduling: `ReduceLROnPlateau`

Comparing this model with the baseline helps investigate the effect of changing the training recipe while keeping the core architecture fixed.

### 3. Hybrid CNN–Transformer

The CNN–Transformer model combines convolutional feature extraction with Transformer-based spatial feature modelling.

DenseNet121 first extracts a spatial feature map. The feature map is converted into a sequence of tokens, and a learnable classification token is introduced. A Transformer encoder processes the sequence to model relationships between spatial locations.

The resulting classification representation is passed to the output layer to predict the three target findings.

This architecture investigates whether modelling spatial relationships improves disease classification compared with a conventional CNN.

### 4. CheXGCN — Graph-Based Label Reasoning

CheXGCN combines image features with graph-based modelling of relationships between disease labels.

A graph represents relationships between the target findings. Graph convolutional layers refine the label representations, and image-to-label compatibility scores are used to generate classification predictions.

In this implementation, the disease co-occurrence graph is constructed from training labels only. The label embeddings are randomly initialized rather than pretrained from external text embeddings.

This approach investigates whether explicitly modelling relationships between target labels can improve multi-label classification.

### Comparison Strategy

The baseline CNN uses a distinct training recipe, whereas the matched CNN, CNN–Transformer, and CheXGCN use the matched training configuration.

Therefore:

- **Baseline CNN vs. Matched CNN:** examines changes associated with the training recipe.
- **Matched CNN vs. CNN–Transformer vs. CheXGCN:** compares alternative architectures under the matched training configuration.

## Data Leakage Prevention

Patient-level data leakage is a significant concern in medical image classification because multiple images may belong to the same patient.

If images from the same patient appear in both training and test partitions, the reported test performance may overestimate the model's ability to generalize to unseen patients.

This project uses patient-level splitting to reduce this risk.

### Leakage Prevention Measures

- Extract patient identifiers before partitioning the dataset.
- Assign patients to training, validation, or test partitions.
- Verify that patient identifiers do not overlap between partitions.
- Reuse the same data partitions across all model configurations.
- Construct the CheXGCN co-occurrence graph using training labels only.
- Tune classification thresholds using validation predictions.
- Reserve the test partition for final evaluation.

The notebook reports no patient overlap between the training, validation, and test partitions.

## Experimental Results

The following results are reported by the Stage 2 notebook using validation-tuned classification thresholds and held-out test predictions.

### Test F1 Score

| Target Finding | Baseline CNN | Matched CNN | CNN–Transformer | CheXGCN |
|---|---:|---:|---:|---:|
| Cardiomegaly | **0.6640** | 0.6514 | 0.6652 | 0.6603 |
| Edema | 0.6987 | **0.6987** | 0.6856 | 0.6928 |
| Pleural Effusion | 0.7722 | 0.7664 | 0.7701 | **0.7727** |

### Test ROC-AUC

| Target Finding | Baseline CNN | Matched CNN | CNN–Transformer | CheXGCN |
|---|---:|---:|---:|---:|
| Cardiomegaly | **0.8546** | 0.8470 | 0.8527 | 0.8503 |
| Edema | 0.8080 | **0.8122** | 0.8004 | 0.8052 |
| Pleural Effusion | **0.8620** | 0.8548 | 0.8576 | 0.8614 |

**Note:** Bold values indicate the highest reported score in each row. Values are rounded to four decimal places.

### Results Analysis

- **Cardiomegaly:** The CNN–Transformer achieves the highest F1 score of `0.6652`, while the baseline CNN achieves the highest ROC-AUC of `0.8546`.
- **Edema:** The baseline and matched CNN report an F1 score of `0.6987`. The matched CNN achieves the highest ROC-AUC of `0.8122`.
- **Pleural Effusion:** CheXGCN achieves the highest F1 score of `0.7727`, while the baseline CNN achieves the highest ROC-AUC of `0.8620`.

The results indicate that the evaluated architectures have broadly comparable performance on the three target findings. No single architecture achieves the highest value across every disease and metric.

The notebook also uses 1,000 bootstrap resamples to estimate confidence intervals for selected test metrics. These intervals are important when interpreting small differences between models.

All reported values should be interpreted in the context of the notebook's experimental sample and evaluation protocol.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision | Measures the proportion of predicted positive cases that are actual positives |
| Recall / Sensitivity | Measures the proportion of actual positive cases detected |
| F1 Score | Balances precision and recall using their harmonic mean |
| ROC-AUC | Measures the model's ability to rank positive cases above negative cases across thresholds |
| Confusion Matrix | Summarizes true positives, false positives, true negatives, and false negatives |
| Bootstrap Confidence Interval | Estimates uncertainty in performance metrics through repeated resampling |

Per-disease thresholds are selected using validation predictions. The selected thresholds are then applied to test predictions for threshold-dependent metrics such as F1 score.

ROC-AUC is calculated using prediction probabilities and does not require a single classification threshold.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| PyTorch | Deep-learning model development and training |
| Torchvision | Image transformations and pretrained DenseNet121 |
| NumPy | Numerical computation |
| pandas | Dataset metadata processing |
| OpenCV | Image-processing support |
| scikit-learn | Evaluation metrics and analysis |
| Matplotlib | Visualisation of results |
| PyTorch Geometric | Graph neural network implementation |
| Jupyter Notebook | Experimental development and execution |
| Kaggle Notebooks | Dataset access and GPU-enabled experimentation |

## Repository Structure

The `DiseaseDetection/` directory contains the following notebooks and model-related assets.

```text
DiseaseDetection/
│
├── Models/
│
├── dataset-loading.ipynb
├── diseasedetectiondensenet121.ipynb
├── final-rextrust-stage2-scaled-6k-umasked-chexgcn.ipynb
├── fork-of-diseasedetectiondensenet121.ipynb
├── fork-of-rextrust-stage2-scaled-20k.ipynb
├── rextrust-stage2-scaled-1k.ipynb
├── rextrust-stage2-scaled-6k.ipynb
└── rextrust-stage2-scaled-20k.ipynb
```

### Main Components

- **`dataset-loading.ipynb`** — dataset loading and preparation workflow.
- **`diseasedetectiondensenet121.ipynb`** — DenseNet121-based disease detection experiments.
- **`rextrust-stage2-scaled-1k.ipynb`** — scaled ReXTrust experiment.
- **`rextrust-stage2-scaled-6k.ipynb`** — scaled ReXTrust experiment.
- **`rextrust-stage2-scaled-20k.ipynb`** — main Stage 2 model comparison workflow described in this README.
- **`final-rextrust-stage2-scaled-6k-umasked-chexgcn.ipynb`** — CheXGCN-related experiment.
- **`Models/`** — model-related assets.

Notebook contents and dataset paths may differ between experimental versions. Use the intended notebook when reproducing a particular set of results.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed libraries.
- Access to the CheXpert dataset under its applicable access and usage terms.
- Jupyter Notebook or Kaggle Notebooks.
- A CUDA-capable GPU is recommended for training.

### 1. Clone the Repository

```bash
git clone https://github.com/SuryaSampreeth/VLIL-Team-95.git
cd VLIL-Team-95/DiseaseDetection
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install General Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib opencv-python scikit-learn jupyter
```

Install PyTorch and Torchvision using the official installation selector:

https://pytorch.org/get-started/locally/

For CheXGCN, install a compatible PyTorch Geometric build using its official installation guide:

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

**Note:** The repository does not currently provide a verified, fully pinned dependency lockfile. Exact package versions should be selected according to the Python, CUDA, and PyTorch versions in your environment.

### 4. Configure the Dataset Path

The main notebook uses a Kaggle-style dataset path similar to:

```python
DATA_ROOT = "/kaggle/input/datasets/ashery/chexpert/"
```

Update `DATA_ROOT` to point to the dataset location in your environment. Ensure that the expected metadata file, `train.csv`, and the corresponding image files are accessible.

## Execution

### Using Jupyter Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open the intended experiment notebook and execute its cells in order.

For the Stage 2 comparison, start with:

```text
rextrust-stage2-scaled-20k.ipynb
```

### Using Kaggle Notebooks

1. Open the notebook in Kaggle.
2. Attach the required dataset.
3. Enable a GPU accelerator if available.
4. Verify the dataset path.
5. Execute the notebook cells in order.
6. Review the final metric tables, confusion matrices, ROC curves, and bootstrap confidence intervals.

Training all four configurations can require substantial GPU time and memory. If you change the sample size, number of epochs, or model configuration, document those changes and evaluate the resulting experiment separately.

## Reproducibility

The main notebook sets random seeds for Python, NumPy, and PyTorch using `SEED = 42`.

The documented configuration includes:

- Image size: `224 × 224`
- Batch size: `32`
- Maximum training epochs: `20`
- Early-stopping patience: `4`

Random seeds improve repeatability but do not guarantee identical results across hardware, CUDA versions, library versions, and data-loading configurations.

For reproducible experiments, record:

- Dataset version and sampling configuration
- Patient-level split details
- Python, PyTorch, Torchvision, CUDA, and PyTorch Geometric versions
- Model architecture and training configuration
- Checkpoint-selection criteria
- Validation thresholds
- Test metrics and bootstrap confidence intervals

## Limitations

- The current experiment focuses on three target findings rather than all available CheXpert labels.
- Sampling and label-processing choices affect the target prevalence.
- The results are based on an internal held-out test partition; external validation is not established by these experiments.
- The CheXGCN implementation uses randomly initialized label embeddings rather than pretrained semantic embeddings.
- Differences between model metrics are relatively small for several comparisons.
- Bootstrap confidence intervals should be considered when interpreting these differences.
- Model predictions and visual explanations can be incorrect and should not be interpreted as proof of clinical reasoning.
- Clinical readiness, prospective clinical utility, and regulatory approval have not been established.

## Future Enhancements

- Extend the experiments to additional thoracic findings.
- Evaluate generalization using independent datasets and institutions.
- Investigate probability calibration and clinically relevant operating thresholds.
- Compare alternative graph construction methods and pretrained label embeddings.
- Analyze false positives and false negatives with appropriate expert review.
- Improve experiment tracking and dependency versioning.
- Evaluate subgroup performance where appropriate data and governance are available.
- Investigate explainability methods while validating their reliability.

## Responsible Use

This project is intended for **research and educational purposes only**. It is not a medical device and must not be used as a substitute for professional medical assessment.

Model predictions may be incorrect and may not generalize across hospitals, populations, scanners, or imaging protocols. Any potential clinical application would require external validation, appropriate clinical oversight, privacy and security review, and applicable regulatory approval.

Use the dataset according to its access conditions and usage terms. Do not commit patient data, credentials, access tokens, or other sensitive information to the repository.

## Acknowledgements

- The authors and maintainers of the CheXpert dataset.
- The researchers whose work on DenseNet, Transformers, and graph neural networks informs these experiments.
- The open-source communities behind PyTorch, Torchvision, NumPy, pandas, scikit-learn, OpenCV, and PyTorch Geometric.

---

**ReXTrust — Stage 2: Chest X-ray Disease Detection**

*Update this documentation whenever the dataset, implementation, experimental configuration, or reported results change.*
