# Accent Detection using Machine Learning and Deep Learning

## Repository Description

This repository contains the complete codebase, experiments, and documentation for an English accent recognition system built using classical machine learning and deep learning techniques. The project leverages the Speech Accent Archive dataset and custom-collected accent samples to classify English accents based on acoustic features such as MFCCs and spectrograms. The repository emphasizes reproducibility, comparative model evaluation, and a robust inference pipeline for real-world usage.

---

## README

### 📌 Project Overview

Accents play a significant role in speech recognition performance and inclusivity. This project focuses on identifying and classifying English accents using a structured machine learning pipeline. Starting from traditional feature-based models and progressing to convolutional neural networks, the system evaluates multiple representations of speech signals to determine the most effective approach for accent recognition.

A key contribution of this project is the augmentation of the Speech Accent Archive with additional Italian and Sicilian accent samples, improving representation for underrepresented dialects.

---

### 🗂 Dataset

* **Primary Dataset**: Speech Accent Archive (Weinberger, 2013)
* **Source**: Kaggle – Speech Accent Archive
* **Description**:

  * 2,000+ English speech samples
  * Speakers from 177 countries and 214 native languages
  * Standardized English paragraph read by all speakers

Custom field recordings are added to enhance accent diversity, particularly for Sicilian accents.

---

### ⚙️ Methodology

The project follows an end-to-end pipeline:

1. **Data Acquisition & Preprocessing**

   * Audio resampling to 16 kHz
   * Mono conversion and silence handling
   * Fixed-length audio segmentation

2. **Dataset Cleaning & Balancing**

   * Label filtering and encoding
   * Class balancing using RandomOverSampler

3. **Feature Extraction**

   * MFCCs
   * Raw Spectrograms
   * Mel Spectrograms
   * 2D Spectrograms for CNNs

4. **Model Training & Evaluation**

   * Logistic Regression (baseline)
   * Shallow MLP
   * Deep MLP with dropout
   * Convolutional Neural Network (CNN)

5. **Model Selection & Persistence**

   * Best-performing model saved with metadata

6. **Inference Utilities**

   * Single-audio prediction
   * Batch prediction and CSV export

---

### 🧠 Models & Tools

* **Frameworks**: PyTorch, torchaudio, scikit-learn
* **Optimization**: Adam optimizer, CrossEntropyLoss
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

### 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run preprocessing and training notebooks/scripts
4. Use inference utilities to predict accents on new audio files

---

### 📊 Outputs

* Trained model weights (`accent_classifier_best.pth`)
* Model metadata (`accent_classifier_metadata.json`)
* Evaluation reports and plots
* CSV files with batch prediction results

---

### 🔁 Reproducibility

The repository includes metadata, fixed random seeds, and clearly documented steps to ensure experiments can be reproduced and extended.

---

### 📈 Future Work

* Expand dataset with more underrepresented accents
* Explore pretrained speech models and transfer learning
* Integrate prosodic features
* Deploy as a real-time web or mobile application

---

### 👥 Authors

Jeff Anderson
Jeff Mobley
Megha Narendra Simha
Sai Mani Ritish

DS-620 Machine Learning & Deep Learning
City University of Seattle
