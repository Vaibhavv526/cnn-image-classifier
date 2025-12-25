import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cnn_model.h5")

st.title("CNN-based Image Classification System")
st.write("Upload a handwritten digit image (0–9)")

uploaded_file = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    st.image(uploaded_file, caption="Uploaded Image", width=150)
    st.success(f"Predicted Digit: {predicted_class}")
