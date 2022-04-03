#Library imports
import numpy as np
import cv2
from keras.models import load_model
import streamlit as st

model = load_model('3maize.h5')

CLASS_NAMES = ['Blight', 'Common_Rust', 'Grey_leaf_Spot']

#Setting Title of App
st.title("Maize Disease Detection")
st.markdown("Upload an image of the plant")

image = st.file_uploader("Choose an image...", type=['png', 'jpg' , 'jpeg'])
submit = st.button('Predict')
#On predict button click
if submit:


    if image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Converting image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Predicting
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str(result))
