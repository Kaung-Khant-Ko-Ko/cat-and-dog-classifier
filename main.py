import streamlit as st
from PIL import Image
from tensorflow.keras import models
import numpy as np
from tensorflow.keras import preprocessing

st.set_page_config(page_title="Cat and Dog Classifier", page_icon="👥")

header = st.container()
desc = st.container()
upload_image = st.container()
image_classifier = st.container()
cont1 = st.empty()
cont2 = st.container()
cont3 = st.container()
copyright_container = st.container()

clicked = False

with header:
    st.title("Cat and Dog Classification Project")


@st.cache
def load_model():
    return models.load_model("classifier_with_transfer_learning.h5")


@st.cache
def predict(image_array):
    dictionary = {0: "ကြောင်", 1: "ခွေး"}
    x = image_array
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    images = np.vstack([x])
    label_class = model.predict(images)
    y_class = np.round(label_class)
    y_label = y_class[0, 0]
    return dictionary[y_label]


with desc:
    st.markdown(
        "This is a cat and dog classifier trained as a machine learning model using Transfer Learning which was "
        "trained with "
        "7,003 train images and 1,002 validation images. The classifier is tested with 2,023 testing images and has a "
        "predicted result of 98.37% accuracy, 98.37% precision score, "
        "98.37% recall score and 0.9837 f1 score. Libraries such as tensorflow, sklearn, numpy, matplotlib, "
        "streamlit are used. "
        "The whole project source code can be found here.")

with upload_image:
    uploaded_image = st.file_uploader("Choose an Image File", type=["jpg", "png", "svg"])

with image_classifier:
    if uploaded_image is not None:
        left_column, right_column = st.columns(2)
        with left_column:
            image = Image.open(uploaded_image)
            st.image(image)

        column1, column2, column3, column4, column5 = st.columns([1.5, 2, 0.5, 4.2, 0.1])
        with column2:
            if st.button("Predict"):
                clicked = True

        with right_column:
            if clicked:
                st.image(image)

        with column4:
            if clicked:
                model = load_model()
                image = image.resize((224, 224))
                img_array = preprocessing.image.img_to_array(image)
                label = predict(img_array)
                st.write("Model မှခန့်မှန်းလိုက်သောအဖြေမှာ", label, "ဖြစ်ပါသည်။")

with copyright_container:
    left_column, middle_column, right_column = st.columns(3)
    if uploaded_image is not None:
        with right_column:
            st.info("Copyright@TeamHybrid")
