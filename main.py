import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import os

# Load your model
try:
    loaded_model = load_model('catanddog.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title('Cats and Dogs Classification Using CNN')

classes = ['Cat', 'Dog']  # Define your classes

# Image upload selection (must be before image display/processing)
genre = st.radio("How You Want To Upload Your Image", ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
elif genre == 'Browse Photos':
    ImagePath = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])  # Specify allowed types
else:
    ImagePath = None  # No image selected yet


# Dictionary to store example image paths (replace with your actual paths)
example_images = {
    "Cat": "cat.png",  
    "Dog": "dog.png",  
    "Flower": "flower.png",  
}

# Function to make a prediction (reusable)
def predict_image(image_path):
    try:
        img = Image.open(image_path)  # Open the image here for consistency
        loaded_single_image = tf.keras.utils.load_img(
            img,  # Pass the image object directly
            color_mode='rgb', target_size=(224, 224)
        )
        test_image = tf.keras.utils.img_to_array(loaded_single_image)
        test_image /= 255.0  # Normalize

        test_image = np.expand_dims(test_image, axis=0)

        logits = loaded_model(test_image)
        softmax = tf.nn.softmax(logits)

        predict_output = tf.argmax(logits, -1).numpy()
        predicted_class = classes[predict_output[0]]
        probability = softmax.numpy()[0][predict_output[0]] * 100

        return predicted_class, probability, img  # Return the image object
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None


# Example images display and prediction (using st.button)
for image_name, image_path in example_images.items():
    try:
        img = Image.open(image_path)

        # Use st.button with the image inside it
        if st.button(image=img, width=250, use_column_width=False, key=image_name):  # Key is important here
            predicted_class, probability, displayed_img = predict_image(image_path)
            if predicted_class and probability:
                st.header(f"Prediction: {predicted_class}")
                st.header(f"Probability: {probability:.4f}%")
                st.image(displayed_img, width=250)

    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
    except UnidentifiedImageError:
        st.error(f"Invalid image format: {image_path}")
    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")

# Uploaded image processing
if ImagePath is not None and genre == 'Browse Photos':  # Check if a file was uploaded
    try:
        img = Image.open(ImagePath)
        st.image(img, width=250)  # Display the uploaded image

        if st.button('Predict Uploaded'):  # Button for uploaded image prediction
            predicted_class, probability, displayed_img = predict_image(ImagePath)
            if predicted_class and probability:
                st.header(f"Prediction: {predicted_class}")
                st.header(f"Probability: {probability:.4f}%")
                st.image(displayed_img, width=250)

    except UnidentifiedImageError:
        st.error("Invalid image format. Please upload a JPG, PNG, or JPEG image.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
