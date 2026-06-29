"""
Train the Diabetic Retinopathy detection model and save artifacts.
Run this once before starting the Streamlit app.

Usage:
    python train_model.py
"""

import os
import warnings
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "DR", "downloads", "dr_small", "retino")
MODEL_DIR = os.path.join(HOME, "DR", "models")
IMG_SIZE = 128


def load_images_from_folder(folder, label, img_size=IMG_SIZE):
    """Load and preprocess images from a directory tree."""
    images, labels = [], []
    for root, _, files in os.walk(folder):
        for img_name in files:
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
                img_path = os.path.join(root, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (img_size, img_size))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"  Skipping {img_path}: {e}")
    return np.array(images), np.array(labels)


def train():
    # ── Load data (DR = 1, No_DR = 0) from train + valid splits ───────
    all_images, all_labels = [], []

    for split in ["train", "valid"]:
        for class_name, label in [("No_DR", 0), ("DR", 1)]:
            folder = os.path.join(DATA_DIR, split, class_name)
            if not os.path.isdir(folder):
                print(f"  ⚠️ Folder not found: {folder}")
                continue
            print(f"Loading {split}/{class_name} …")
            imgs, lbls = load_images_from_folder(folder, label)
            print(f"  Found {len(imgs)} images")
            if len(imgs) > 0:
                all_images.append(imgs)
                all_labels.append(lbls)

    if len(all_images) == 0:
        print(f"\n⚠️  No images found in {DATA_DIR}")
        print("   Expected structure: retino/{train,valid}/{DR,No_DR}/")
        return

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    images = images / 255.0

    print(f"\nTotal images: {images.shape[0]}")

    # ── Feature extraction ─────────────────────────────────────────────
    HAS_TF = False
    try:
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model as KerasModel

        HAS_TF = True
    except ImportError:
        pass

    if HAS_TF:
        print("Extracting VGG16 features (may take a few minutes) …")
        base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        feat_model = KerasModel(inputs=base.input, outputs=base.layers[-2].output)
        features = feat_model.predict(images, verbose=1)
        features = features.reshape(features.shape[0], -1)
    else:
        print("TensorFlow unavailable – using pixel features …")
        features = images.reshape(images.shape[0], -1)

    print(f"Feature shape: {features.shape}")

    # ── Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Train LightGBM ─────────────────────────────────────────────────
    print("\nTraining LightGBM …")
    lgbm = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=31,
        random_state=42,
        force_col_wise=True,
        verbosity=-1,
    )
    lgbm.fit(X_train, y_train)

    preds = lgbm.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # ── Save artifacts ─────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lgbm, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(HAS_TF, os.path.join(MODEL_DIR, "has_tf.pkl"))

    print(f"\n✅ Model saved to {MODEL_DIR}/")
    print("   lgbm_model.pkl  — trained classifier")
    print("   scaler.pkl      — feature scaler")
    print("   has_tf.pkl      — TensorFlow flag")


if __name__ == "__main__":
    train()
