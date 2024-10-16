import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model('mnist-cnn.keras')

st.title("Predicción de Modelo Keras")

# Opción para cargar imagen desde archivo
uploaded_file = st.file_uploader("Elige una imagen...", type="jpg")

# Opción para capturar imagen desde la cámara
camera_input = st.camera_input("Captura una imagen")

# Procesar la imagen desde el archivo cargado
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagen cargada.', use_column_width=True)
elif camera_input is not None:  # Procesar la imagen desde la cámara
    img = Image.open(camera_input)
    st.image(img, caption='Imagen capturada.', use_column_width=True)
else:
    img = None

if img is not None and st.button("Predecir"):
    img = img.resize((64, 64))  # Cambia el tamaño según tu modelo
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    if(predicted_class==1):
        st.write(f"Predicción: El neumatico esta en buenas condiciones para ser usado.")
    else:
        st.write(f"Predicción: El neumatico NO esta en buenas condiciones.")
    

