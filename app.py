import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.set_page_config(
    page_title="Vegetable Image Classifier",
    page_icon="🥕",
    layout="centered",
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


model_path = "finetuned_mobilenet.keras"
try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Class names ---
class_names = ['Indian market', 'Onion', 'Potato', 'Tomato']

# --- Preprocessing ---
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

# --- UI Title ---
st.title("🥕 Vegetable Image Classifier")
st.caption("Upload an image of **indianmarket**,**Tomato**, **Onion**, or **Potato** and get a prediction!")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess & Predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Log to console
    st.write("Prediction raw output:", prediction)
    st.write("Predicted class:", predicted_class)
    st.write("Confidence score:", confidence)

    # Show Prediction
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.progress(confidence)

    if confidence > 0.9:
        st.success("✅ High Confidence!")
    elif confidence > 0.6:
        st.warning("⚠️ Moderate Confidence")
    else:
        st.error("❌ Low Confidence — Try a clearer image.")
