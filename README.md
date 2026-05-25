

```markdown
# Skin Lesion Classification on HAM10000 Dataset
### Applied Machine Learning - Advanced Course Final Project

**Student Name:** Negin Nillforoosh  
**Matricola:** 1164045  

---

## Project Overview
This repository contains the advanced deep learning project focused on classifying skin lesions using the **HAM10000 dataset**. The project addresses severe class imbalance in medical imaging and follows a rigorous engineering pipeline, comparing multiple modeling strategies from simple baselines to deep transfer learning.

## Repository Structure
The project is strictly organized to adhere to professional software engineering and reproducible research standards:

```text
├── Negin_Machine_Learning_Advanced_Project.ipynb  # Main experimentation & analysis notebook
├── utils.py                                       # Modularized helper functions (plotting, metrics, feature extraction)
└── README.md                                      # Project documentation and summary

```

* **`Negin_Machine_Learning_Advanced_Project.ipynb`**: Contains the core pipeline, including exploratory data analysis (EDA), training routines, and evaluation.
* **`utils.py`**: A dedicated Python module that houses reusable functions for drawing training histories, computing advanced evaluation metrics, plotting labeled confusion matrices, and extracting classical image features.

---

## Implemented Enhancements (Second Submission)

In this final version, the project has been substantially upgraded to address previous methodological limitations:

### 1. Rigorous Methodological Justifications

* **Image Resizing:** Changed resolution to $128 \times 96$ pixels, keeping the original $4:3$ aspect ratio to prevent geometric distortion of lesions while balancing computational efficiency.
* **Data Augmentation:** Applied strict, mild spatial transformations (horizontal/vertical flips, minor rotations). Explicitly avoided aggressive color jittering because color channels carry critical clinical diagnostic criteria for skin malignancies.
* **Normalization:** Evaluated and justified based on the pixel-value distributions across the RGB channels to stabilize gradient descent.

### 2. Comprehensive Baselines

To quantitatively justify the necessity of a complex Convolutional Neural Network (CNN), two clear baselines were introduced:

* **Trivial Baseline:** A majority-class dummy predictor acting as the absolute lower bound.
* **Classic ML Baseline:** A Logistic Regression model trained on global RGB color histogram features extracted via `utils.py`.

### 3. Advanced Class Imbalance Management

* Integrated automated `class_weight` calculation directly into the multiclass cross-entropy loss function to penalize minority class misclassifications more heavily during optimization.
* Shifted the evaluation paradigm from standard Accuracy to imbalance-aware metrics: **Balanced Accuracy**, **Macro F1-Score**, and **Weighted F1-Score**.

### 4. Code Quality & Readability

* **Modular Design:** Extracted repetitive visual and evaluation logic into the external `utils.py` script, leaving the main notebook clean, readable, and analytical.
* **Plot Readability:** Updated the confusion matrix and error analysis charts to display explicit clinical disease labels (e.g., *Melanoma, Basal cell carcinoma*) instead of raw numerical class indices.

---

## Key Results Summary

* **Baselines:** The trivial and classic ML models struggled heavily due to the dataset's high skewness.
* **CNN from Scratch:** A custom residual architecture demonstrated solid pattern learning, far outperforming the baselines.
* **Final Model (DenseNet121):** Utilizing transfer learning yielded the highest performance, significantly improving the sensitivity (recall) on critical minority classes like Melanoma.

```

```
