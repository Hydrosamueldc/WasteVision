"""Run with: streamlit run app.py"""
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from utils import ROOT, CLASSES, load_checkpoint, preprocessing

st.set_page_config(page_title="WasteVision", page_icon="♻️")
st.title("WasteVision")
st.write("Upload one waste item to predict its category.")


@st.cache_resource
def load_model():
    torch.set_num_threads(4)
    model, _ = load_checkpoint(ROOT / "models/wastevision.pth", torch.device("cpu"))
    return model


if not (ROOT / "models/wastevision.pth").exists():
    st.info("Train the model first: python train.py")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded is not None:
    try:
        with Image.open(uploaded) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("This file could not be opened as an image. Please choose another file.")
        st.stop()
    st.image(image, caption="Uploaded image", alt="Waste item uploaded for classification", width="stretch")
    model = load_model()
    with torch.inference_mode():
        probabilities = model(preprocessing()(image).unsqueeze(0)).softmax(dim=1)[0]
    scores, indices = probabilities.topk(3)
    st.subheader(f"Prediction: {CLASSES[indices[0].item()]}")
    st.write(f"Confidence: {scores[0].item():.1%}")
    if st.checkbox("Show top 3 predictions"):
        for score, index in zip(scores.tolist(), indices.tolist()):
            st.write(f"{CLASSES[index].capitalize()}: {score:.1%}")
    st.caption("Confidence is a model score, not a guarantee. This model always chooses one of six categories.")
