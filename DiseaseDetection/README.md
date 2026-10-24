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
