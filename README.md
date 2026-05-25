
## Project Overview

This project addresses the multi-class classification of dermoscopic skin lesion images into seven diagnostic categories using the HAM10000 dataset. The central methodological challenge is severe class imbalance: the most frequent class (*Melanocytic Nevi*) accounts for roughly 67% of samples, while rare but clinically critical classes such as *Melanoma* and *Dermatofibroma* represent fewer than 10% combined.

The project implements a full deep learning pipeline — from exploratory data analysis and preprocessing justification, through baseline comparisons, to a custom residual CNN and a DenseNet121 transfer learning model — with every methodological decision explicitly motivated rather than applied by convention.

---

## Repository Structure

```
skin-lesion-classification-advanced/
│
├── Negin_Machine_Learning_Advanced_Project.ipynb   # Main notebook: full pipeline and analysis
├── utils.py                                         # Reusable helper functions (imported by notebook)
└── README.md                                        # This file
```

### `utils.py` — Module Reference

All reusable logic is extracted from the notebook into `utils.py` and imported at the top of the notebook. This separation keeps the notebook focused on analysis and decisions rather than implementation details.

| Function | Purpose |
|---|---|
| `plot_model_history(history)` | Plots training/validation accuracy, loss, and val macro-F1 curves |
| `extract_color_hist_features(images, bins=16)` | Extracts density-normalized RGB histogram features for ML baselines |
| `evaluate_predictions(name, y_true, y_pred)` | Returns a dict with Accuracy, Balanced Accuracy, Macro F1, and Weighted F1 |
| `make_metric_table(results)` | Converts a list of metric dicts into a clean, rounded comparison DataFrame |
| `plot_labeled_confusion_matrix(cm, class_names, ...)` | Plots a confusion matrix with explicit disease name labels on both axes |
| `combine_training_histories(*histories)` | Merges multiple Keras History objects for continuous curve plotting |

---

## The Seven Diagnostic Classes

| Code | Disease | Type |
|---|---|---|
| `nv` | Melanocytic Nevi | Benign |
| `mel` | Melanoma | Malignant |
| `bkl` | Benign Keratosis-like Lesions | Benign |
| `bcc` | Basal Cell Carcinoma | Malignant |
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma | Pre-malignant |
| `vasc` | Vascular Lesions | Benign |
| `df` | Dermatofibroma | Benign |

---

## Pipeline

### 1. Exploratory Data Analysis

- Distribution of classes, patient age, anatomical localization, and diagnosis method
- Side-by-side comparison of visually similar class pairs (e.g. Melanoma vs. Nevi) to illustrate why the classification task is difficult
- RGB channel pixel-value histograms per class, used to empirically justify the normalization step

### 2. Preprocessing — With Explicit Justification

**Resizing to 128 × 96 pixels**  
The original images average around 450 × 600 pixels. They are resized to 128 × 96 to preserve the original 4:3 aspect ratio (preventing geometric distortion of lesion morphology) while keeping memory and compute manageable for training on a single GPU/CPU.

**Data Augmentation**  
Only mild, clinically safe transformations are applied: horizontal and vertical flips, and small rotations (≤ 20°). Aggressive color jitter and large-angle rotations are deliberately avoided because dermoscopic colour information (e.g., blue-white veil, pigmentation patterns) and lesion shape are both used in clinical diagnosis and must not be distorted.

**Normalization**  
Per-channel pixel-value distributions are inspected before normalization. All three channels have means well above zero and are not centred, which would slow gradient descent and cause instability. Images are normalized to [0, 1] before the from-scratch CNN and with the DenseNet ImageNet statistics for transfer learning.

### 3. Baseline Comparisons

Two baselines are included to provide a quantitative lower bound and to justify the necessity of a deep CNN:

- **Majority-class dummy predictor** — always predicts *Melanocytic Nevi*; shows the floor that any real model must surpass
- **Classical ML baseline** — Logistic Regression trained on 48-dimensional RGB colour histogram features (extracted via `utils.py`); demonstrates what a non-spatial, hand-crafted feature approach achieves

### 4. Class Imbalance Management

- **During training:** Inverse-frequency class weights are computed with `sklearn.utils.class_weight.compute_class_weight` and passed directly to the cross-entropy loss function. Square-root weighting is used for the from-scratch CNN to avoid over-correcting on extreme minority classes.
- **During evaluation:** Standard accuracy is supplemented with **Balanced Accuracy**, **Macro F1-Score**, and **Weighted F1-Score** throughout all experiments. A custom `MacroF1Callback` tracks validation macro-F1 at the end of each epoch so that learning-rate scheduling and early stopping both monitor this imbalance-aware metric.

### 5. Models

| Model | Description |
|---|---|
| **Dummy Baseline** | Majority-class predictor |
| **LR + RGB Histograms** | Logistic Regression on colour histogram features |
| **Residual CNN (from scratch)** | Custom architecture with separable convolutions and residual connections, ~4M parameters, trained with AdamW and cosine annealing |
| **DenseNet121 (transfer learning)** | Pre-trained on ImageNet; fine-tuned in two stages (frozen backbone → full model) with `densenet_preprocess_input` normalization |

### 6. Evaluation

All four models are compared on the same held-out test set using the `evaluate_predictions` / `make_metric_table` utilities. Error analysis uses `plot_labeled_confusion_matrix` with explicit clinical disease names on both axes, and a per-class misclassification bar chart to identify which classes drive the most errors.

---

## Key Results

| Model | Balanced Accuracy | Macro F1 |
|---|---|---|
| Majority-class dummy | ~0.14 | ~0.07 |
| LR + RGB histograms | ~0.27 | ~0.18 |
| Residual CNN (scratch) | substantially above baselines | ✓ |
| DenseNet121 (transfer) | best overall | ✓ highest minority-class recall |

The transfer learning model achieves the highest sensitivity on clinically critical minority classes such as Melanoma, confirming that the added complexity of a deep CNN is empirically justified over the classical baselines.

---

## Requirements

```
tensorflow >= 2.10
scikit-learn >= 1.2
pandas
numpy
matplotlib
seaborn
Pillow
```

Install all dependencies with:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn Pillow
```

---

## Dataset

The HAM10000 dataset must be downloaded separately from Kaggle. Update the data path at the top of the notebook before running.

[Download HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
