#Library imports
import numpy as np
import cv2
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
st.markdown("Upload an image of the plant")

images = st.file_uploader("Choose an image...", type=['png', 'jpg' , 'jpeg'])
img = image.load_img('crr.jpg',target_size = (256,256))

submit = st.button('Predict')
#On predict button click
if submit:


    if image is not None:
#        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
   #     opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(img)
      #  st.write(opencv_image.shape)
       # opencv_image = cv2.resize(opencv_image, (256,256))
        #Converting image to 4 Dimension
        
        predicted_class, confidence = predict(model, img)
        st.write(type(img))
        st.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%")
