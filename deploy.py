import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Function to preprocess the uploaded image and make predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.preprocessing.image import img_to_array, load_img

def predict_image(image, model):
    img = load_img(image, target_size=(150, 150))  # Resize the image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    prediction = model.predict(img_array)
    return prediction

    # Flatten the image array
    flattened_img_array = img_array.reshape(1, -1)

    result = model.predict(flattened_img_array)
    return result[0][0]


# Load the model
model = load_model(r"C:\Users\Lalji Bodar\OneDrive\Documents\Bunny\Projects\Bone-Classifier\model.h5")

# Streamlit app
st.title('Bone Fracture Detection')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = predict_image(uploaded_file, model)

    # Display prediction result
    if prediction > 0.5:
        st.write('No Fractures Here, Clear Bones! 🦴')
    else:
        st.write('Fracture Detected, Caution Advised! 🚨')
