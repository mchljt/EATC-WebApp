import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = load_model("cnn_model.h5")

# Map class indices
class_indices = {'FAKE': 0, 'REAL': 1}
labels = {v: k for k, v in class_indices.items()}

# Image size (matching model's input)
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Streamlit UI
st.title("🕵️‍♂️ Deepfake Image Detector")
st.write("Upload an image and this app will tell you whether it is likely a **deepfake** or **real**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.convert('RGB')              # ensure 3 channels
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    predicted_class = int(np.round(prediction))
    confidence = prediction if predicted_class == 1 else 1 - prediction

    # Output
    st.markdown("---")
    st.subheader("🔍 Prediction:")
    st.write(f"**Class:** {labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    st.write(f"Raw prediction: {prediction}")
    st.write(f"**Predicted class:** {predicted_class} | **Raw score:** {prediction:.4f} | **Mapped label:** {labels[predicted_class]}")


