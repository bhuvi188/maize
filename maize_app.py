#Library imports
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model
import streamlit as st

model = load_model('3maize.h5')
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

#Setting Title of App
st.title("Maize Disease Detection")
st.markdown("Upload an image of the plant")

image = st.file_uploader("Choose an image...", type=['png', 'jpg' , 'jpeg'])
submit = st.button('Predict')
#On predict button click
if submit:


    if image is not None:
        plt.figure(figsize=(15, 15))
        predicted_class, confidence = predict(model, image)

        plt.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%")
