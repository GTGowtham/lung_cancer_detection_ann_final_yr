import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import requests

# Dynamic model loading
MODEL_URL = "https://link-to-your-model-file/lung_cancer_model.h5"  # Replace with the actual URL of your .h5 file
MODEL_PATH = "lung_cancer_model.h5"

def download_model():
    """Download the model if not already present."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        st.success("Model downloaded successfully!")

# Ensure the model is available
download_model()

# Load the model
model = load_model(MODEL_PATH)

# Define a function for preprocessing the uploaded image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')  # Convert to grayscale
    # Resize to the model's expected input size
    image = image.resize((256, 256))
    # Normalize the image
    image_array = np.array(image) / 255.0
    # Add channel and batch dimensions
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
    return image_array

# Define class labels
class_labels = ['Benign', 'Malignant', 'Normal']

# Streamlit app
st.set_page_config(page_title="Lung Cancer Detection", layout="centered", page_icon="🫁")

# Header and introduction
st.title("🫁 Lung Cancer Detection System")
st.markdown(
    """
    Welcome to the Lung Cancer Detection System!  
    Upload a **CT scan image** to classify it into one of the following categories:
    - **Benign**: Early signs of abnormalities.
    - **Normal**: No significant signs of cancer.
    - **Malignant**: High likelihood of cancer.

    The model has been trained to provide a **98% accuracy** in predictions.
    """
)

# Upload an image file
uploaded_file = st.file_uploader(
    "Upload a CT scan image (JPG or PNG)", 
    type=["jpg", "png"],
    help="Make sure the image is clear and properly formatted."
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan", use_column_width=True)
    st.write("---")  # Divider

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence_scores = prediction[0]

    # Display the prediction and confidence
    st.subheader(f"Prediction: **{predicted_class}**")
    
    st.markdown("### Confidence Scores:")
    for i, label in enumerate(class_labels):
        st.write(f"- **{label}**: {confidence_scores[i]:.2f}")
    
    # Add a progress bar for visualization
    st.progress(int(max(confidence_scores) * 100))

    # Show additional tips based on prediction
    if predicted_class == "Malignant":
        st.error("⚠️ High likelihood of cancer detected. Please consult a medical professional immediately.")
    elif predicted_class == "Benign":
        st.warning("⚠️ Early signs detected. Consider scheduling a follow-up with your doctor.")
    else:
        st.success("✅ No significant signs of cancer detected. Maintain regular check-ups.")

    st.write("---")  # Divider
    st.info("Note: This tool is for preliminary analysis only. Always consult a medical professional for an accurate diagnosis.")

else:
    st.info("Please upload a CT scan image to proceed.")

# Footer
st.markdown(
    """
    ---
    **Developed by Gowtham** | Powered by TensorFlow and Streamlit  
    """
)
