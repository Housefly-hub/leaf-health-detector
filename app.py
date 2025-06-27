import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_model.h5", compile=False)
    with open("classes.txt", "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    return model, class_names

model, class_names = load_model_and_labels()

# UI
st.set_page_config(page_title="Leaf Health Detector", layout="centered")
st.title("🌿 Leaf Health Detector")
st.write("Upload an image of a plant leaf to check its health status.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess: Resize & Normalize
    resized_image = image.resize((224, 224))
    img_array = np.asarray(resized_image).astype(np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_class = class_names[predicted_index]

    # Display Result
    st.subheader("Prediction:")
    if "healthy" in predicted_class.lower():
        st.success(f"✅ {predicted_class} ({confidence:.2f}%)")
    else:
        st.error(f"⚠️ {predicted_class} ({confidence:.2f}%)")

    # Optional confidence breakdown
    with st.expander("🔍 View all class confidences"):
        for i, prob in enumerate(predictions[0]):
            st.write(f"{class_names[i]}: {prob*100:.2f}%")
