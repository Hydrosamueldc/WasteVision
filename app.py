"""Waste image classification interface. Run with: streamlit run app.py."""
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from utils import ROOT, load_checkpoint, predict_image

MODEL_PATH = ROOT / "models/wastevision_stage3.pth"

st.set_page_config(page_title="WasteVision — Waste Image Classifier", layout="centered")
st.title("WasteVision — Waste Image Classifier")
st.write("Classify an image as cardboard, glass, metal, paper, plastic, or trash.")
st.caption(
    "Predictions are experimental and may be wrong, especially for images that differ "
    "from the training dataset. Model scores are not guarantees of correctness."
)


@st.cache_resource(max_entries=1)
def cached_checkpoint(path, device):
    """Keep only the model resource cached; never cache uploaded images."""
    torch.set_num_threads(4)
    return load_checkpoint(path, device)


if not MODEL_PATH.is_file():
    st.error("Model checkpoint missing: models/wastevision_stage3.pth. Restore the trained checkpoint.")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    try:
        with Image.open(uploaded) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("This file could not be opened as an image. Please choose a valid JPG or PNG image.")
        st.stop()

    st.image(image, caption="Uploaded image", alt="Uploaded waste image", width=400)
    if st.button("Classify Image", type="primary"):
        try:
            with st.spinner("Classifying image..."):
                result = predict_image(image, model_path=MODEL_PATH,
                                       checkpoint_loader=cached_checkpoint)
        except Exception as error:
            # Keep failures readable in the interface rather than showing a traceback.
            st.error(f"Prediction failed: {error}")
        else:
            st.subheader(f"Predicted category: {result['predicted_class']}")
            st.metric("Model score", f"{result['score']:.2%}")
            st.write("Top three predictions")
            for item in result["top_three"]:
                st.write(f"{item['class'].capitalize()}: {item['score']:.2%}")
