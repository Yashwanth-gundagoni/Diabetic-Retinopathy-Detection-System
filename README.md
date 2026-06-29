# 🩺 Diabetic Retinopathy Detection System

> An AI-powered tool that detects **Diabetic Retinopathy (DR)** from retinal images using Machine Learning. Upload an OCT or Fundus image and get an instant prediction through a clean Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-green?logo=lightgbm&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-orange)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Dataset Download](#-dataset-download)
- [Training the Model](#-training-the-model)
- [Running the Web App](#-running-the-web-app)
- [How to Use](#%EF%B8%8F-how-to-use)
- [Running the Notebook](#-running-the-jupyter-notebook)
- [Algorithm & Pipeline](#-algorithm--pipeline)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 🔬 Overview

Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes and is a leading cause of blindness. This project uses:

- **LightGBM** classifier trained on retinal images
- **VGG16** (optional, via TensorFlow) for deep feature extraction
- **Streamlit** for a user-friendly web interface
- **OpenCV + scikit-learn** for image preprocessing and ML utilities

---

## 📁 Project Structure

```
vaishu-major-project/
├── app.py                        # Streamlit Web UI
├── train_model.py                # Model training script
├── dataset_analysis.py           # Dataset analysis utilities
├── organize_data.py              # Data organization helper
├── draw_methodology.py           # Methodology diagram generator
├── Diabetic_Retinopathy.ipynb    # Jupyter Notebook (step-by-step)
├── methodology_diagram.png       # Pipeline diagram
├── test_image.jpg                # Sample test image
├── models/
│   ├── lgbm_model.pkl            # Trained LightGBM model (generated)
│   ├── scaler.pkl                # Feature scaler (generated)
│   └── has_tf.pkl                # TensorFlow availability flag (generated)
├── data/
│   ├── OCT/                      # OCT retinal scan images
│   └── Fundus/                   # Fundus retinal images
├── downloads/                    # Raw downloaded datasets
├── FIXES_APPLIED.md              # Changelog of fixes
└── README.md                     # This file
```

---

## ✅ Prerequisites

Before you begin, ensure you have the following installed on your machine:

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | [Download here](https://www.python.org/downloads/) |
| pip | latest | Comes with Python |
| Git | any | [Download here](https://git-scm.com/downloads) |

**Check your Python version:**
```bash
python3 --version
```

---

## 🚀 Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/srivardhan-kondu/vaishu-major-project.git
cd vaishu-major-project
```

### Step 2 — Create a Virtual Environment

A virtual environment keeps your project dependencies isolated from the rest of your system.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> 💡 You should see `(.venv)` at the start of your terminal prompt once activated.

### Step 3 — Install Required Packages

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib opencv-python scikit-learn lightgbm seaborn joblib tqdm streamlit Pillow
```

#### All packages explained:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical array operations |
| `pandas` | Data handling and analysis |
| `matplotlib` | Plotting and visualization |
| `opencv-python` | Image reading and preprocessing |
| `scikit-learn` | ML utilities (scaler, train/test split, metrics) |
| `lightgbm` | Gradient boosting classifier |
| `seaborn` | Statistical data visualization |
| `joblib` | Saving and loading ML model files |
| `tqdm` | Progress bars during training |
| `streamlit` | Web UI framework |
| `Pillow` | Image handling in Python |

### Step 4 — (Optional) Install TensorFlow for Better Accuracy

TensorFlow enables **VGG16 deep feature extraction**, which significantly improves model accuracy.

```bash
pip install tensorflow
```

> ⚠️ TensorFlow requires ~500 MB of disk space. If you skip this, the model will use pixel-based features as a fallback — it still works, just with lower accuracy.

---

## 📦 Dataset Download

You need a retinal image dataset to train the model. Below are the recommended datasets:

### Option A — APTOS 2019 Blindness Detection (Recommended for Beginners)

1. Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
2. Sign in to Kaggle (free account required)
3. Accept the competition rules
4. Download `train_images.zip` and `train.csv`
5. Extract images into the following structure:

```
downloads/dr_small/retino/
├── train/
│   ├── DR/       ← Images with Diabetic Retinopathy
│   └── No_DR/    ← Healthy retinal images
└── valid/
    ├── DR/
    └── No_DR/
```

### Option B — Kaggle CLI (Faster)

Install the Kaggle CLI and download directly:

```bash
pip install kaggle
# Place your kaggle.json API key at ~/.kaggle/kaggle.json
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d downloads/
```

### Other Recommended Datasets

| Dataset | Type | Images | Link |
|---------|------|--------|------|
| **APTOS 2019** | Fundus | 5,590 | [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |
| **Kermany OCT** | OCT | 84,000+ | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
| **EyePACS** | Fundus | 88,000+ | [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection) |
| **MESSIDOR-2** | Fundus | 1,748 | [Research site](https://www.adcis.net/en/third-party/messidor2/) |
| **IDRiD** | Fundus | 516 | [Research site](https://idrid.grand-challenge.org/) |

---

## 🤖 Training the Model

Once your dataset is in place, run the training script:

```bash
# Make sure your virtual environment is active
source .venv/bin/activate     # macOS/Linux
# or
.venv\Scripts\activate.bat    # Windows

# Train
python train_model.py
```

**Expected output:**
```
Loading train/No_DR …
  Found 2000 images
Loading train/DR …
  Found 2000 images
...
Training LightGBM …

✅ Accuracy: 87.45%

Classification Report:
              precision    recall  f1-score ...

✅ Model saved to ~/DR/models/
   lgbm_model.pkl  — trained classifier
   scaler.pkl      — feature scaler
   has_tf.pkl      — TensorFlow flag
```

> ⏱️ Training time: ~2–10 minutes depending on dataset size and whether TensorFlow is available.

---

## 🌐 Running the Web App

```bash
streamlit run app.py
```

Open your browser and go to: **http://localhost:8501**

---

## 🖥️ How to Use

1. Open **http://localhost:8501** in your browser
2. Click **"Browse files"** or drag & drop a retinal image (`.jpg`, `.png`, `.bmp`, `.tif`)
3. The app displays:
   - Your uploaded image
   - **Prediction** — `Healthy (No DR)` or `Diabetic Retinopathy Detected`
   - **Confidence score** (percentage)
   - **Probability breakdown** for each class
4. Click **"Technical Details"** to see feature extraction info

---

## 📓 Running the Jupyter Notebook

If you prefer a step-by-step notebook workflow:

```bash
# Activate environment
source .venv/bin/activate

# Install Jupyter if not already installed
pip install jupyter

# Open in Jupyter
jupyter notebook Diabetic_Retinopathy.ipynb

# OR open in VS Code
code Diabetic_Retinopathy.ipynb
```

Run all cells **sequentially from top to bottom** (Cell → Run All).

---

## 🔁 Algorithm & Pipeline

```
Input Images (OCT + Fundus)
        ↓
   Resize to 128×128 px
        ↓
   Normalize pixels (0–1)
        ↓
   Feature Extraction
   ├── VGG16 deep features (if TensorFlow available)
   └── Pixel flattening fallback
        ↓
   StandardScaler normalization
        ↓
   Train/Test Split (80/20)
        ↓
   LightGBM Classifier
   (150 estimators, lr=0.05, max_depth=10)
        ↓
   Prediction + Confidence Score
```

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10+** | Core language |
| **Streamlit** | Web UI framework |
| **LightGBM** | Gradient boosting classifier |
| **OpenCV** | Image loading and preprocessing |
| **scikit-learn** | ML utilities (scaler, metrics, splits) |
| **TensorFlow/Keras** | VGG16 feature extraction (optional) |
| **Seaborn / Matplotlib** | Data visualization |
| **Pillow** | Image handling in Python |
| **joblib** | Model serialization |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist for clinical decisions.

---

## 📝 License

This project is open-source for educational use.

---

## 👩‍💻 Author

**Vaishu** · Major Project · 2024–2025
