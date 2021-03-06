#Library imports
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import streamlit as st
class_names=['Blight', 'Common_Rust', 'Gray_Leaf_Spot']
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
st.markdown("Upload an image of Maize plant leaf which has disease")

images = st.file_uploader("Choose an image...", type=['png', 'jpg' , 'jpeg'])
submit = st.button('Predict')


#On predict button click
if submit:

    if images is not None:
        
        images = Image.open(images)
        images = images.resize((256, 256))
        images=images.convert('RGB')
        st.write(images.mode)
        st.write(images.size)

# show the image
        st.image(images)
        
        predicted_class, confidence = predict(model, images)
        
     
        st.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        
       
