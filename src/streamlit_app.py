import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model('mnist-cnn.keras')

st.title("Analizador de estado de neumaticos")
uploaded_file = st.file_uploader("Elige una imagen...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagen cargada.', use_column_width=True)

    if st.button("Predecir"):
        img = img.resize((64, 64))  # Cambia el tamaño según tu modelo
        img_array = np.array(img) / 255.0  # Normalizar
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        if (predicted_class == 1):
            st.write(f"Predicción: El neumatico esta en buenas condiciones para ser usado")
        else:
            st.write(f"Predicción: El neumatico NO DEBE ser usado")
        
