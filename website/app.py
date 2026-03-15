import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# Page settings
st.set_page_config(
    page_title="EndoNet",
    
    layout="wide",
)

st.title("EndoNet")
st.caption("Research prototype for automated laparoscopic lesion analysis")
st.subheader("Automated Detection of Endometriosis Lesions")

st.markdown(
"""
Upload a laparoscopic image to identify potential endometriosis lesions.
"""
)

# Sidebar
st.sidebar.title("About EndoNet")

st.sidebar.markdown(
"""
Deep learning framework for detection of endometriosis lesions.

Detected Classes:
- Peritoneum
- Ovary
- TIE
- Uterus
"""
)

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    0.9,
    0.25
)

MODEL_PATH = "models/best.pt"

# Load model if available
model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    st.warning("Model not found. Running in demo mode.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Laparoscopic Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Image")
        st.image(image, use_column_width=True)

    if model:

        results = model(image_np, conf=confidence_threshold)
        result_img = results[0].plot()

        with col2:
            st.markdown("### Detected Lesions")
            st.image(result_img, use_column_width=True)

        st.markdown("### Detection Results")

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]

                st.write(f"**{class_name}**")
                st.progress(confidence)

        else:

            st.info("No lesions detected.")

    else:

        with col2:
            st.markdown("### Detected Lesions")
            st.info("Model not loaded yet.")
