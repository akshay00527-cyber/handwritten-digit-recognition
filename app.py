
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Digit AI | Handwritten Recognition",
    page_icon="🔢",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #182848, #243b55);
    color: white;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 30px;
}
.result-card {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
.result-digit {
    font-size: 90px;
    font-weight: 900;
    line-height: 1.1;
}
.result-label {
    font-size: 20px;
    font-weight: 600;
}
.info-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 15px;
}
h2, h3 {
    color: white !important;
}
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🔢 Digit AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Handwritten Digit Recognition Using Deep Learning</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
<center>
Draw a single digit between <b>0 and 9</b>.
Our neural network will analyze your handwriting and predict the digit.
</center>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

model = load_model()

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.1, 1])

with left:
    st.markdown("### ✍️ Draw Your Digit")

    canvas = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="digit_canvas",
        return_image_data=True
    )

    predict_button = st.button(
        "🚀 Predict Digit",
        type="primary"
    )

    clear_button = st.button("🔄 Clear Drawing")

    if clear_button:
        st.rerun()

with right:
    st.markdown("### 🤖 AI Prediction")

    if predict_button:

        if canvas.image_data is None:
            st.warning("Please draw a digit first.")

        else:
            image = Image.fromarray(
                canvas.image_data.astype("uint8")
            ).convert("L")

            image = ImageOps.invert(image)
            pixels = np.array(image)

            coords = np.argwhere(pixels > 40)

            if len(coords) == 0:
                st.warning("Please draw a digit first.")

            else:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                cropped = image.crop((
                    x_min, y_min,
                    x_max + 1, y_max + 1
                ))

                # Check horizontal digit groups
                cropped_pixels = np.array(cropped) > 40
                occupied = np.any(cropped_pixels, axis=0)

                groups = 0
                inside = False

                for value in occupied:
                    if value and not inside:
                        groups += 1
                        inside = True
                    elif not value:
                        inside = False

                # Approximate multiple-digit detection
                width = cropped.width
                height = cropped.height

                multiple_digits = (
                    groups >= 2 or width > height * 1.35
                )

                if multiple_digits:

                    st.markdown("""
                    <div class="result-card">
                        <div class="result-label">⚠️ Cannot Determine</div>
                        <h3>Please Draw Only ONE Digit</h3>
                        <p>Supported digits: 0 to 9</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    # Resize and center
                    cropped.thumbnail((20, 20))

                    final_image = Image.new(
                        "L", (28, 28), 0
                    )

                    x_offset = (28 - cropped.width) // 2
                    y_offset = (28 - cropped.height) // 2

                    final_image.paste(
                        cropped,
                        (x_offset, y_offset)
                    )

                    # Normalize
                    image_array = np.array(
                        final_image,
                        dtype="float32"
                    ) / 255.0

                    image_array = image_array.reshape(
                        1, 28, 28
                    )

                    # Prediction
                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    digit = int(np.argmax(predictions))
                    confidence = float(predictions[digit])

                    # Display result
                    if confidence < 0.70:

                        st.markdown("""
                        <div class="result-card">
                            <div class="result-label">⚠️ Cannot Determine</div>
                            <h3>Draw More Clearly</h3>
                        </div>
                        """, unsafe_allow_html=True)

                    else:

                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">Predicted Digit</div>
                            <div class="result-digit">{digit}</div>
                            <p>Confidence: {confidence * 100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")

                    # Confidence scores
                    st.markdown("### 📊 Prediction Scores")

                    for i, score in enumerate(predictions):
                        st.progress(
                            float(score),
                            text=f"Digit {i}: {score * 100:.2f}%"
                        )

                    st.markdown("### 🖼️ Processed Image")

                    st.image(
                        final_image,
                        caption="28 × 28 image sent to the model",
                        width=150
                    )

    else:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">Your Prediction Will Appear Here</div>
            <div class="result-digit">?</div>
            <p>Draw a digit and click Predict</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("")
st.markdown("""
<center style="color:#94a3b8">
Built with Python • TensorFlow • Streamlit • MNIST
</center>
""", unsafe_allow_html=True)
