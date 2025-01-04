import streamlit as st
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize database
DATABASE_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    # Create the users table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            diagnosis_date TEXT NOT NULL,
            diagnosis_image BLOB,
            prediction_result TEXT
        )
    ''')
    # Ensure the prediction_result column exists (for backward compatibility)
    try:
        c.execute("ALTER TABLE users ADD COLUMN prediction_result TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

# Save user registration details in the database
def save_user(name, email, password, diagnosis_date, image_path):
    with open(image_path, 'rb') as file:
        image_blob = file.read()

    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO users (name, email, password, diagnosis_date, diagnosis_image) 
        VALUES (?, ?, ?, ?, ?)
    ''', (name, email, password, diagnosis_date, image_blob))
    conn.commit()
    conn.close()

# Update prediction result in the database
def update_prediction(email, prediction_result):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''
        UPDATE users SET prediction_result = ? WHERE email = ?
    ''', (prediction_result, email))
    conn.commit()
    conn.close()

# Preprocess image for prediction
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((256, 256))  # Resize to the model's expected input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
    return image_array

# Load the trained model
model = load_model('lung_cancer_model.h5')

# Class labels
class_labels = ['Benign', 'Malignant', 'Normal']

# Initialize database
init_db()

# Streamlit app
st.set_page_config(page_title="Lung Cancer Detection", layout="centered", page_icon="🫁")

# Registration Page
if 'registered' not in st.session_state:
    st.title("User Registration")

    with st.form("registration_form"):
        name = st.text_input("Name", max_chars=50)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        diagnosis_date = st.date_input("Diagnosis Date")
        uploaded_image = st.file_uploader("Upload Diagnosis Image (JPG or PNG)", type=["jpg", "png"])

        submitted = st.form_submit_button("Register")

        if submitted:
            if name and email and password and diagnosis_date and uploaded_image:
                # Save the uploaded image temporarily
                temp_image_path = os.path.join("temp_image.jpg")
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

                try:
                    save_user(name, email, password, diagnosis_date.strftime('%Y-%m-%d'), temp_image_path)
                    st.success("Registration successful! You can now proceed to the prediction.")
                    st.session_state['registered'] = email
                except sqlite3.IntegrityError:
                    st.error("Email already exists. Please use a different email.")
                os.remove(temp_image_path)
            else:
                st.error("Please fill all fields and upload an image.")

# Prediction Page
if 'registered' in st.session_state:
    st.title("🫁 Lung Cancer Detection System")
    st.markdown("""
        Welcome to the Lung Cancer Detection System!  
        Upload a **CT scan image** to classify it into one of the following categories:
        - **Benign**: Early signs of abnormalities
        - **Normal**: No significant signs of cancer
        - **Malignant**: High likelihood of cancer

        The model has been trained to provide a **98% accuracy** in predictions.
    """)

    uploaded_file = st.file_uploader(
        "Upload a CT scan image (JPG or PNG)", 
        type=["jpg", "png"], 
        help="Make sure the image is clear and properly formatted."
    )

    if uploaded_file is not None:
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
        st.subheader(f"### Prediction: **{predicted_class}**")
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

        # Update the prediction result in the database
        update_prediction(st.session_state['registered'], predicted_class)
    else:
        st.info("Please upload a CT scan image to proceed.")
