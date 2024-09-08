import tensorflow as tf
from tensorflow.keras import layers, Model
import streamlit as st
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from PIL import Image

st.set_page_config(page_title="Rock Paper Scissors", layout="wide")


#
# x = layers.Conv2D(16, 3, activation="relu")(img_input)
# x = layers.MaxPooling2D(2)(x)
#
# x = layers.Conv2D(32, 3, activation="relu")(x)
# x = layers.MaxPooling2D(2)(x)
#
# x = layers.Conv2D(64, 3, activation="relu")(x)
# x = layers.MaxPooling2D(2)(x)
#
# x = layers.Flatten()(x)
# x = layers.Dense(512, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
@st.cache(allow_output_mutation=True)
def load_images():
    rock = Image.open("rock.png")
    paper = Image.open("paper.png")
    scissors = Image.open("scissors.png")
    return (rock, paper, scissors)


@st.cache(allow_output_mutation=True)
def load_model():
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False)
    for layer in pre_trained_model.layers:
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(3, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)
    model.load_weights("my_model.hdf5")
    model.compile()
    return model

@st.cache(allow_output_mutation=True)
def load_model_2():
    img_input = layers.Input(shape=(150, 150, 3))

    x = tf.image.rgb_to_grayscale(img_input)

    x = layers.Conv2D(32, 3, activation="relu", kernel_regularizer="l2")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.LeakyReLU(0.01)(x)

    x = layers.Conv2D(64, 3, activation="relu", kernel_regularizer="l2")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.LeakyReLU(0.01)(x)

    x = layers.Conv2D(128, 3, activation="relu", kernel_regularizer="l2")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.LeakyReLU(0.01)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu", activity_regularizer="l2")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(512, activation="relu", activity_regularizer="l2")(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(3, activation="softmax")(x)

    model = Model(img_input, output)
    model.load_weights("my_model_2.hdf5")
    model.compile()
    return model

images = load_images()
my_model = load_model()
my_model_2 = load_model_2()
current_model = my_model

st.title("Rock Paper Scissors")
st.header("Play rock paper scissors using your camera")

radio = st.radio("Current model", ["Inception_v3","CNN"], horizontal=True)
if radio == "Inception_v3":
    current_model = my_model
elif radio == "CNN":
    current_model = my_model_2

col1, col2 = st.columns([1.3,1], gap="large")

with col1:
    picture = st.camera_input("Take a picture, keep your hand close to the camera")
if picture is not None:
    bytes_data = picture.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels=3)
    normalization_layer = layers.Rescaling(1. / 255)(img_tensor)
    resizing_layer = layers.Resizing(150, 150)(normalization_layer)

    data = tf.reshape(resizing_layer, shape=(1, 150, 150, 3))

    prediction = current_model.predict(data)
    with col2:
        pred = np.argmax(prediction)
        move = (pred + 1) % 3

        st.write(f"Your move (Confidence:{prediction[0][pred]:.3f})")
        st.image(images[pred])
        st.write("My move")
        st.image(images[move])


