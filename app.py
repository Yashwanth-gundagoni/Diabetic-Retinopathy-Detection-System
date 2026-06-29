"""
Diabetic Retinopathy Detection — Streamlit Web UI
Upload a retinal image and get an instant prediction.

Usage:
    streamlit run app.py
"""

import os
import numpy as np
import cv2
import joblib
import streamlit as st
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
IMG_SIZE = 128

LABEL_MAP = {
    0: "Healthy (No DR)",
    1: "Diabetic Retinopathy Detected",
}

LABEL_COLORS = {
    0: "green",
    1: "red",
}

LABEL_ICONS = {
    0: "✅",
    1: "⚠️",
}


# ── Helper functions ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load saved model artifacts (cached so it only runs once)."""
    model_path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    tf_flag_path = os.path.join(MODEL_DIR, "has_tf.pkl")

    if not os.path.exists(model_path):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    has_tf = joblib.load(tf_flag_path) if os.path.exists(tf_flag_path) else False
    return model, scaler, has_tf


@st.cache_resource
def load_vgg16():
    """Load VGG16 feature extractor (only if TensorFlow is available)."""
    try:
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model as KerasModel

        base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        feat_model = KerasModel(inputs=base.input, outputs=base.layers[-2].output)
        return feat_model
    except ImportError:
        return None


def preprocess_image(pil_image):
    """Convert a PIL image to a preprocessed numpy array."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def extract_features(img_array, has_tf, vgg_model):
    """Extract features using VGG16 or raw pixel flattening."""
    if has_tf and vgg_model is not None:
        feats = vgg_model.predict(img_array, verbose=0)
        return feats.reshape(feats.shape[0], -1)
    return img_array.reshape(img_array.shape[0], -1)


# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR Detection",
    page_icon="🩺",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title { text-align: center; color: #1E3A5F; }
    .subtitle  { text-align: center; color: #555; margin-bottom: 2rem; }
    .result-box {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        font-size: 1.3rem; font-weight: 600; margin-top: 1rem;
    }
    .result-healthy  { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .result-dr       { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>🩺 Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Upload a retinal image (OCT or Fundus) to check for signs of Diabetic Retinopathy</p>",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This application uses a **LightGBM** machine-learning model "
        "trained on OCT and Fundus retinal images to predict Diabetic Retinopathy."
    )
    st.divider()
    st.subheader("How to use")
    st.markdown(
        """
        1. **Upload** a retinal image (.jpg, .png, .bmp, .tif)
        2. Wait for the model to **analyze** the image
        3. View the **prediction** and confidence score
        """
    )
    st.divider()
    st.subheader("Supported formats")
    st.code("JPG · JPEG · PNG · BMP · TIF")
    st.divider()
    st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")

# ── Load model ─────────────────────────────────────────────────────────
model, scaler, has_tf = load_model()
vgg_model = load_vgg16() if has_tf else None

if model is None:
    st.error(
        "**Model not found!** Please train the model first by running:\n\n"
        "```bash\npython train_model.py\n```"
    )
    st.stop()

# ── Upload section ─────────────────────────────────────────────────────
st.divider()
uploaded_file = st.file_uploader(
    "📁 Upload a retinal image",
    type=["jpg", "jpeg", "png", "bmp", "tif"],
    help="Drag and drop or click to browse. Max 200 MB.",
)

if uploaded_file is not None:
    # Display the uploaded image
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(pil_image, use_container_width=True)

    # Run prediction
    with st.spinner("🔍 Analyzing image …"):
        img_array = preprocess_image(pil_image)
        features = extract_features(img_array, has_tf, vgg_model)
        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100

    # Display results
    with col2:
        st.subheader("📊 Results")

        label_text = LABEL_MAP.get(prediction, "Unknown")
        icon = LABEL_ICONS.get(prediction, "❓")
        css_class = "result-healthy" if prediction == 0 else "result-dr"

        st.markdown(
            f"<div class='result-box {css_class}'>{icon} {label_text}</div>",
            unsafe_allow_html=True,
        )

        st.metric(label="Confidence", value=f"{confidence:.1f}%")

        # Probability breakdown
        st.write("**Class Probabilities:**")
        for idx, prob in enumerate(probabilities):
            label = LABEL_MAP.get(idx, f"Class {idx}")
            st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

    # Additional details in expander
    with st.expander("🔬 Technical Details"):
        st.write(f"- **Image size:** {pil_image.size[0]} × {pil_image.size[1]} px")
        st.write(f"- **Resized to:** {IMG_SIZE} × {IMG_SIZE} px")
        st.write(f"- **Feature extraction:** {'VGG16 (Deep Learning)' if has_tf else 'Pixel-based'}")
        st.write(f"- **Feature vector length:** {features.shape[1]}")
        st.write(f"- **Classifier:** LightGBM")
        st.write(f"- **Raw prediction:** Class {prediction}")
        st.write(f"- **Probabilities:** {[f'{p:.4f}' for p in probabilities]}")
else:
    # Placeholder when no image is uploaded
    st.info("👆 Upload a retinal image above to get started.")

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · LightGBM · OpenCV · scikit-learn")
