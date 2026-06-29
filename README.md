# 🩺 Diabetic Retinopathy Detection System

> An AI-powered tool that detects **Diabetic Retinopathy (DR)** from retinal images using Machine Learning. Upload an OCT or Fundus image and get an instant prediction through a clean Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-green?logo=lightgbm&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-orange)

---

## 📑 Table of Contents

- Overview
- Project Structure
- Prerequisites
- Installation
- Dataset Download
- Training the Model
- Running the Web App
- How to Use
- Running the Notebook
- Algorithm & Pipeline
- Tech Stack
- Disclaimer

---

## 🔬 Overview

Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes and is a leading cause of blindness.

This project uses:

- **LightGBM** classifier trained on retinal images
- **VGG16** (optional) for deep feature extraction
- **Streamlit** for a clean web interface
- **OpenCV + scikit-learn** for preprocessing and ML

---

## 📁 Project Structure

```text
Diabetic-Retinopathy-Detection-System/
├── app.py
├── train_model.py
├── dataset_analysis.py
├── organize_data.py
├── draw_methodology.py
├── Diabetic_Retinopathy.ipynb
├── methodology_diagram.png
├── test_image.jpg
├── models/
│   ├── lgbm_model.pkl
│   ├── scaler.pkl
│   └── has_tf.pkl
├── data/
│   ├── OCT/
│   └── Fundus/
├── downloads/
├── FIXES_APPLIED.md
└── README.md
```

---

## ✅ Prerequisites

- Python 3.10+
- pip
- Git

Check version:

```bash
python3 --version
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/Yashwanth-gundagoni/Diabetic-Retinopathy-Detection-System.git
cd Diabetic-Retinopathy-Detection-System
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Windows:

```cmd
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib opencv-python scikit-learn lightgbm seaborn joblib tqdm streamlit Pillow
```

(Optional)

```bash
pip install tensorflow
```

---

## 📦 Dataset Download

Recommended datasets:

- APTOS 2019 Blindness Detection
- EyePACS
- Kermany OCT
- IDRiD
- MESSIDOR-2

Download from Kaggle and organize into the required folders before training.

---

## 🤖 Training

```bash
python train_model.py
```

---

## 🌐 Run the Web App

```bash
streamlit run app.py
```

Open:

http://localhost:8501

---

## 🖥️ Usage

- Upload an OCT/Fundus image
- View prediction
- Confidence score
- Probability distribution
- Technical details

---

## 📓 Jupyter Notebook

```bash
pip install jupyter
jupyter notebook Diabetic_Retinopathy.ipynb
```

---

## 🔁 Pipeline

```text
Input Image
      ↓
Resize (128×128)
      ↓
Normalize
      ↓
Feature Extraction
 ├─ VGG16
 └─ Pixel Features
      ↓
StandardScaler
      ↓
Train/Test Split
      ↓
LightGBM
      ↓
Prediction
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LightGBM
- OpenCV
- scikit-learn
- TensorFlow/Keras
- Matplotlib
- Seaborn
- Pillow
- Joblib

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only** and must not be used as a substitute for professional medical diagnosis.

---

## 📝 License

Educational Use.

---

## 👨‍💻 Author

**Yashwanth-gundagoni**  
Major Project · 2024–2025
