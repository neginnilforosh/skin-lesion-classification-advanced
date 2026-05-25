
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

All reusable plotting and evaluation logic is extracted into `utils.py` and imported at the top of the notebook with:

```python
from utils import (
    extract_color_hist_features,
    plot_model_history,
    plot_labeled_confusion_matrix,
    evaluate_predictions,
    make_metric_table,
)
```

| Function | Purpose |
|---|---|
| `plot_model_history(history)` | Plots training/validation accuracy, loss, and val macro-F1 curves in one call |
| `extract_color_hist_features(images, bins=16)` | Extracts density-normalized RGB histogram features used by the ML baseline |
| `evaluate_predictions(name, y_true, y_pred)` | Returns a dict with Accuracy, Balanced Accuracy, Macro F1, and Weighted F1 |
| `make_metric_table(results)` | Converts a list of metric dicts into a clean, rounded comparison DataFrame |
| `plot_labeled_confusion_matrix(cm, class_names, ...)` | Plots a confusion matrix with explicit disease name labels on both axes |
| `combine_training_histories(*histories)` | Merges multiple Keras History objects for continuous multi-stage training curves |

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

- Distribution of classes, patient age, anatomical localization, and diagnosis confirmation method
- Side-by-side comparison of visually similar class pairs (Melanoma vs. Nevi; Actinic Keratoses vs. Benign Keratosis; Basal Cell Carcinoma vs. Benign Keratosis) to illustrate why the classification task is difficult even for image-based models
- RGB channel pixel-value histograms on a random sample of 1000 images, used to empirically justify the normalization step

### 2. Preprocessing — With Explicit Justification

**Resizing to 128 × 96 pixels**  
The original images are 450 × 600 pixels. They are resized to 128 × 96 for four reasons: this preserves the original 4:3 aspect ratio (preventing geometric distortion of lesion morphology); it retains more spatial and texture detail than the earlier 100 × 75 setup; it remains computationally manageable; and it satisfies the minimum input size required by DenseNet121.

**Data Augmentation**  
Only mild, clinically safe spatial transformations are applied: horizontal and vertical flips and small rotations (≤ 20°). Aggressive colour jitter and large-angle rotations are deliberately avoided because dermoscopic colour information (e.g., blue-white veil, pigmentation patterns) and lesion shape are both used in clinical diagnosis and must not be distorted.

**Normalization**  
Per-channel pixel-value distributions are inspected before applying normalization. All three channels have means well above zero and are not centred. Pixel values are scaled to [0, 1] for the from-scratch CNN (rescaling layer inside the model) and with ImageNet channel statistics for DenseNet121 (`densenet_preprocess_input`).

### 3. Baseline Comparisons

Two baselines are included to provide a quantitative lower bound and to justify the necessity of a deep CNN:

- **Majority-class dummy predictor** — always predicts *Melanocytic Nevi*; establishes the floor that any real model must surpass
- **Classical ML baseline** — Logistic Regression (balanced class weights, `StandardScaler`) trained on 48-dimensional RGB colour histogram features extracted via `utils.py`; demonstrates what a non-spatial, hand-crafted feature approach achieves

### 4. Class Imbalance Management

**During training:** Inverse-frequency class weights are computed with `sklearn.utils.class_weight.compute_class_weight`. Square-root weighting is applied to the from-scratch CNN to avoid over-correcting on extreme minority classes. Both full and square-root weight dictionaries are computed and made available for the two CNN models.

**During evaluation:** A custom `MacroF1Callback` (defined in the notebook) computes validation macro-F1 at the end of every epoch and writes it into the Keras logs, making it available to `ReduceLROnPlateau` and `EarlyStopping`. Standard accuracy is reported alongside **Balanced Accuracy**, **Macro F1**, and **Weighted F1** for all four models.

### 5. Models

| Model | Description |
|---|---|
| **Dummy Baseline** | Majority-class predictor |
| **LR + RGB Histograms** | Logistic Regression on colour histogram features (balanced class weights) |
| **Residual CNN (from scratch)** | Custom architecture with separable convolutions and residual connections, trained with AdamW + label smoothing + square-root class weights |
| **DenseNet121 (transfer learning)** | Pre-trained on ImageNet; fine-tuned in two stages (frozen backbone → full model) using `densenet_preprocess_input` and square-root class weights |

### 6. Evaluation

All four models are compared on the same held-out test set (2003 samples) using the `evaluate_predictions` / `make_metric_table` utilities from `utils.py`. Error analysis uses `plot_labeled_confusion_matrix` with explicit clinical disease names (not numerical indices) on both axes, alongside per-class misclassification analysis.

---

## Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Majority-class dummy | 0.6695 | 0.1429 | 0.1146 | 0.5370 |
| RGB histogram + Logistic Regression | 0.4673 | 0.4786 | 0.3335 | 0.5296 |
| Residual CNN (from scratch) | 0.6970 | 0.4844 | 0.4394 | 0.6940 |
| **DenseNet121 (transfer learning)** | **0.7519** | **0.5798** | **0.5716** | **0.7516** |

The dummy baseline reveals the ceiling achievable without any learning: a model that only predicts the majority class reaches 66.9% accuracy but 11.5% macro F1, confirming that accuracy alone is misleading under class imbalance. The classical ML baseline improves substantially on balanced accuracy (47.9%) but remains well below the CNNs. The final DenseNet121 model achieves the highest performance across all four metrics, including the best sensitivity on clinically critical minority classes such as Melanoma and Basal Cell Carcinoma.

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
kagglehub
```

Install with:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn Pillow kagglehub
```

The dataset is downloaded automatically at runtime via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
```
