# Driver Drowsiness Detection System

## Overview

Driver drowsiness is a major cause of road accidents worldwide. This project aims to detect early signs of fatigue in drivers using computer vision and deep learning techniques.

The system analyzes facial features—specifically eye closure and yawning—to determine the driver’s alertness level and provide a safety assessment.

---

## Problem Statement

To develop a system that can classify driver states based on visual cues and determine whether the driver is alert or fatigued.

The model classifies images into:

* Open Eyes
* Closed Eyes
* Yawning
* No Yawning

These predictions are combined using decision logic to estimate the driver’s fatigue level.

---

## Approach

### 1. Data Preparation

* Dataset divided into:

  * 70% Training
  * 15% Validation
  * 15% Testing
* Ensured proper class distribution across all sets

---

### 2. Exploratory Data Analysis

* Verified class balance
* Visualized sample images
* Checked image quality, size, and brightness
* Removed corrupted or inconsistent data

---

### 3. Data Preprocessing

* Resized images to **224 × 224**
* Normalized pixel values (0–255 → 0–1)
* Applied data augmentation:

  * Rotation
  * Zoom
  * Horizontal flip
  * Brightness variation

---

### 4. Model Development

Two approaches were explored:

#### 🔹 Custom CNN

* Built from scratch
* Used convolution and pooling layers
* Helped establish baseline performance

#### 🔹 Transfer Learning (MobileNetV2)

* Pretrained on ImageNet
* Faster convergence and better accuracy
* Selected as the final approach

---

### 5. Final Model Architecture

Instead of a single multi-class model, the system uses **two specialized binary classifiers**:

*  **Eye Model**

  * Classifies: Open vs Closed

* **Mouth Model**

  * Classifies: Yawn vs No Yawn

This modular approach improves performance and reduces misclassification between unrelated features.

---

### 6. Fatigue Detection Logic

The final fatigue level is determined using decision fusion:

| Eye State | Mouth State | Fatigue Level     |
| --------- | ----------- | ----------------- |
| Open      | No Yawn     | ✅ Alert           |
| Open      | Yawn        | ⚠️ Mild Fatigue   |
| Closed    | Any         | 🚨 Severe Fatigue |

> Eye closure is given higher priority as it is a stronger indicator of driver fatigue compared to yawning.

---

## Model Performance

* Achieved strong accuracy on test data
* Performs well on clearly distinguishable features
* Minor confusion observed in edge cases (e.g., subtle yawning expressions)

---

## Visualizations

The project includes:

* Confusion Matrix
* Class Distribution Graph
* Sample Input Images
* Model Performance Analysis

---

## Limitations

* Performance may degrade with:

  * Low-light conditions
  * Blurry or low-resolution images
  * Partial or obstructed faces

* Some ambiguity in borderline cases (e.g., slight mouth opening vs yawning)

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib / Seaborn
* Streamlit

---

## Streamlit Application

An interactive web interface built using Streamlit allows users to:

* Upload an image
* Select image type:

  * Eye Image (Open/Closed)
  * Full Face (Yawn/No Yawn)
* View real-time predictions and fatigue assessment

---

## Project Structure

```
Driver-Drowsiness-Detection/
│
├── app.py
├── eye_model.h5
├── mouth_model.h5
├── Driver Drowsiness Detection.ipynb
├── requirements.txt
├── README.md
├── sample_images.png
├── confusion_matrix.png
└── ...
```

---

## 📌 Conclusion

This project demonstrates how deep learning can be applied to improve road safety by detecting driver fatigue using visual indicators. By combining eye and mouth analysis, the system provides a practical and effective approach to drowsiness detection.




