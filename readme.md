
# Hand Gesture Classification Project

## Overview

This project implements a **hand gesture recognition system** using machine learning. It uses **MediaPipe Hands** to extract 21 hand landmarks from video frames and classifies gestures with different ML models such as **RandomForest, SVM, and KNN**. The project evaluates model performance and saves both trained models and confusion matrices for analysis.

---

## Features

* Extract **21 hand landmarks** (x, y, z) per hand using MediaPipe.
* Flatten landmarks into **63 features** for ML models.
* Train and evaluate multiple models:

  * **RandomForestClassifier**
  * **SVM (Support Vector Machine)**
  * **K-Nearest Neighbors (KNN)**
* Compute evaluation metrics:

  * Accuracy, Precision, Recall, F1-Score
* Generate and **save confusion matrices** for each model.
* Save trained **model pipelines** for future use.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Merna-Hany12/Hand-Gesture-Classification-model.git```

2. Create a virtual environment and activate it (optional but recommended):

```bash
python -m venv env
# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Example requirements.txt**:

```
# ----------------------------
# Hand Tracking & Computer Vision
# ----------------------------
mediapipe==0.10.32
opencv-python==4.11.0

# ----------------------------
# Data Handling
# ----------------------------
pandas==2.0.3
numpy==1.26.4

# ----------------------------
# Machine Learning
# ----------------------------
scikit-learn==1.1.3
joblib==1.2.0

# ----------------------------
# Visualization
# ----------------------------
matplotlib==3.9.4
seaborn==0.12.2
```
---

## Usage
1. **Prepare your dataset**:
   * Video or images of hand gestures.
   * Extract landmarks using MediaPipe Hands.
   * Flatten landmarks to `[x1, y1, z1, ..., x21, y21, z21]`.

2. **Train models and evaluate**:
 
Open the notebook hand_gesture.ipynb in Jupyter Notebook and run the cells sequentially.

The notebook will:
* Preprocess the landmarks: recenter using the wrist and normalize based on finger length.
* Encode labels: convert gesture names into numeric labels.
* Train all models in your `models` dictionary (**RandomForest, SVM, GradientBoosting**).
* Compute metrics.
* Save **trained pipelines** in `../models/`.
* Save **confusion matrices** in `../models/confusion_matrices/`.

3. **Predict new gestures**:

```python
import joblib

pipeline = joblib.load("../models/RandomForest_pipeline.pkl")
y_pred = pipeline.predict(new_hand_landmarks)
```

---

## Folder Structure

```
project/
│
├─ models/                        # Saved trained pipelines
│   ├─ RandomForest.pkl
│   ├─ SVM.pkl
│   ├─ KNN.pkl
│
├─ models/confusion_matrices/     # Saved confusion matrix images
│   ├─ RandomForest_confusion_matrix.png
│   ├─ SVM_confusion_matrix.png
│   ├─ KNN_confusion_matrix.png
│
├─ data/                          # Input videos/images dataset
├─ output/                        # Generated videos with landmark overlay & predictions
│   ├─ output_video.mp4
├─ hand_gesture.ipynb             # Notebook for preprocessing, training, evaluation & video output
├─ requirements.txt               # Project dependencies
└─ README.md
```

---

## Results

* Each model prints a **classification report** with accuracy, precision, recall, and F1-score.
* Confusion matrices are visualized and saved for analysis, helping to identify misclassified gestures.
* Trained models can be reused for predicting new hand gestures.
Confusion Matrices

RandomForest:
![RandomForest Confusion Matrix](models/confusion_matrices/RandoForest_confusion_matrix.png)  

SVM:
![SVM Confusion Matrix](models/confusion_matrices/SVM_confusion_matrix.png)  


KNN:
![GradientBoosting Confusion Matrix](models/confusion_matrices/KNN_confusion_matrix.png)  

Model Comparison
| Model        | Accuracy | Precision | Recall | F1-Score |
| ------------ | -------- | --------- | ------ | -------- |
| RandomForest | 0.9759   | 0.9761    | 0.9759 | 0.9759   |
| SVM          | 0.9823   | 0.9826    | 0.9823 | 0.9823   |
| KNN          | 0.9566   | 0.9573    | 0.9566 | 0.9566   |

## License

This project is released under the **MIT License**.

