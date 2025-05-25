# 🌱 Soil Classification using Deep Learning

![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning-based solution for **Soil Classification**, developed as part of the **Kaggle "True Soil Classification" Challenge** hosted by **ANNAM AI**.

This repository includes two challenges:
- 🔹 **Challenge 1:** Multiclass Soil Image Classification
- 🔹 **Challenge 2:** Anomaly Detection via Autoencoders

---

## 💡 Challenge 1: Multiclass Soil Classification

**Objective:** Classify images into one of four soil categories:
- Alluvial
- Clay
- Red
- Black

### 🧠 Model Architecture
- **Model Used:** Transfer Learning with MobileNetV2 (pre-trained on ImageNet)
- **Layers Fine-Tuned:** Only dense layers for task-specific features
- **Techniques Used:**
  - Image Normalization
  - Dropout (0.3) and L2 Regularization to prevent overfitting
  - ReLU Activation in hidden layers
  - Softmax Output for 4-class prediction
  - EarlyStopping to avoid overtraining

### 🔧 Training Details
- Optimizer: `Adam`
- Loss Function: `Sparse Categorical Crossentropy`
- Metric: `Accuracy`
- Validation Strategy: 80-20 train-validation split

### 📊 Performance
- **F1-Score:** `0.9655` on training data
- Visualization: Confusion matrix, Accuracy & Loss curves
- Final Output: `submission.csv` with predicted soil types

---

## 💡 Challenge 2: Anomaly Detection via Autoencoders

**Objective:** Binary classification — determine if an image is a soil type or not

### 🧠 Model Architecture
- **Method:** Unsupervised Deep Learning using Autoencoders
- **Approach:**
  - Trained Autoencoder to reconstruct known soil images
  - Predicted whether test images are "soil" based on reconstruction error
- **Techniques:**
  - Encoder-Decoder structure with Sigmoid Activation
  - Mean Squared Error for pixel-level comparison
  - Threshold optimization to distinguish anomalies

### ⚠️ Challenges
- Single-class training data made supervised learning infeasible
- Autoencoder had difficulty generalizing to unseen images
- Some misclassifications (e.g., frogs or boats detected as soil due to similar textures)

---

## ✅ Final Outcome

Despite the complexity of Challenge 2, both tasks were tackled with creative problem-solving and deep learning techniques.

| Challenge | Task Type           | Technique                     | Final Score / Remarks              |
|----------:|---------------------|-------------------------------|------------------------------------|
| 1         | Multiclass Classification | Transfer Learning (MobileNetV2) | 🏆 F1 Score: `0.9655`               |
| 2         | Binary Classification      | Autoencoder Anomaly Detection    |   🏆 F1 Score: `0.7228`  |

---

## 📦 Installation & Setup

1. Clone the repo
   ```bash
   git clone https://github.com/Abhipsit16/SoilClassification_annam.git
2. Install Dependencies
```bash
 pip install -r requirements.txt
