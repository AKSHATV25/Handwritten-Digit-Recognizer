import pickle

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]


model, scaler = load_model()

st.title("✏️ Handwritten Digit Recognizer")
st.write(
    "Draw a single digit (0–9) in the box below, then click **Predict**. "
    "The model was trained on the scikit-learn digits dataset (8×8 grayscale images)."
)

col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("**Instructions**")
    st.markdown(
        "- Draw one digit, centered\n"
        "- Use a thick stroke\n"
        "- Click Predict to see the result\n"
        "- Click the trash icon on the canvas to clear"
    )
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA").convert("L")

        if np.array(img).max() == 0:
            st.warning("Please draw a digit first.")
        else:
            img_resized = img.resize((8, 8), Image.LANCZOS)
            arr = np.array(img_resized).astype("float64")

            # Scale pixel intensities to the 0-16 range used by the sklearn digits dataset
            arr = (arr / 255.0) * 16.0
            arr = arr.reshape(1, -1)

            arr_scaled = scaler.transform(arr)
            pred = model.predict(arr_scaled)[0]
            proba = model.predict_proba(arr_scaled)[0]

            st.success(f"### Predicted digit: **{pred}**")
            st.write(f"Confidence: {proba[pred] * 100:.1f}%")

            st.bar_chart({"probability": proba})

st.divider()
st.caption(
    "Model: MLPClassifier (scikit-learn) trained on the digits dataset · "
    "Built with Streamlit"
)
