import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import streamlit as st
class_names=['Blight', 'Common_Rust', 'Gray_Leaf_Spot']
model = load_model('3maize.h5')
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    result = class_names[np.argmax(predictions)]
    YY=np.array(predictions, dtype=np.float32)
    st.title('Probabilities')
    st.write('Blight')
    st.write(YY[0][0])
    st.write('Common_Rust')
    st.write(YY[0][1])
    st.write('Gray_Leaf_Spot')
    st.write(YY[0][2])
    st.warning('Predicted : '+result)
    if result is 'Common_Rust':
        st.info('Common rust produces rust-colored to dark brown, elongated pustules on both leaf surfaces. The pustules contain rust spores (urediniospores) that are cinnamon brown in color. Pustules darken as they age. Leaves, as well as sheaths, can be infected. Under severe conditions leaf chlorosis and death may occur. Common rust can be differentiated from Southern rust by the brown pustules occurring on both top and bottom leaf surfaces with common rust.')
    elif result is 'Blight':
        st.info('Long, elliptical lesions are often described as cigar-shaped; they are typically large (up to 10 cm or more), greyish-green or tan in colour. The disease develops first on the lower leaves of the canopy and progresses upward on the plant through the growing season. Under severe infection, leaves of a susceptible hybrid can become blighted or ‘burned’ giving the appearance of late-season frost or freeze injury to the plant. The disease is often confused with Stewart’s wilt.')
    else:
        st.info('Gray leaf spot, caused by the fungus Cercospora zeae-maydis, occurs virtually every growing season. If conditions favor disease development, economic losses can occur. Symptoms first appear on lower leaves about two to three weeks before tasseling. The leaf lesions are long (up to 2 inches), narrow, rectangular, and light tan colored. Later, the lesions can turn gray. They are usually delimited by leaf veins but can join together and kill entire leaves. ')
#Setting Title of App
image = Image.open('index.jpg')
st.image(image,use_column_width=True)
st.title("Maize Disease Detection")
st.markdown("Upload only images of Maize plant leaf which has a disease for better results")

images = st.file_uploader("Choose an image...", type=['png', 'jpg' , 'jpeg'])
submit = st.button('Predict')


#On predict button click
if submit:

    if images is not None:
        
        images = Image.open(images)
        images = images.resize((256, 256))
        images=images.convert('RGB')
        #st.write(images.mode)
        st.write(images.size)

# show the image
        st.image(images)
        
        predicted_class= predict(images)

