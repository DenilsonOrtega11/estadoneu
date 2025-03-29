import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

# Función para cargar el modelo desde un archivo en memoria
def cargar_modelo(model_file):
    try:
        model = tf.keras.models.load_model(BytesIO(model_file.read()))
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

st.title("Analizador de estado de neumáticos")

# Opción para cargar un modelo propio
uploaded_model = st.file_uploader("Sube tu modelo (.h5 o .keras)", type=["h5", "keras"])

# Cargar el modelo predeterminado si no se sube ninguno
if uploaded_model is not None:
    model = cargar_modelo(uploaded_model)
    if model is None:
        st.stop()  # Detener la ejecución si no se pudo cargar el modelo
else:
    # Si no se sube ningún modelo, se carga el modelo predeterminado
    try:
        model = tf.keras.models.load_model('mnist-cnn.keras')
    except Exception as e:
        st.error(f"Error al cargar el modelo predeterminado: {str(e)}")
        st.stop()

# Opción para cargar imagen desde archivo
uploaded_file = st.file_uploader("Elige una imagen...", type="jpg")

# Opción para capturar imagen desde la cámara
camera_input = st.camera_input("Captura una imagen")

# Procesar la imagen desde el archivo cargado
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Imagen cargada.', use_column_width=True)
    except Exception as e:
        st.error(f"Error al cargar la imagen: {str(e)}")
        img = None
elif camera_input is not None:  # Procesar la imagen desde la cámara
    try:
        img = Image.open(camera_input)
        st.image(img, caption='Imagen capturada.', use_column_width=True)
    except Exception as e:
        st.error(f"Error al capturar la imagen: {str(e)}")
        img = None
else:
    img = None

# Predicción si se tiene una imagen
if img is not None and st.button("Predecir"):
    try:
        img = img.resize((64, 64))  # Cambia el tamaño según tu modelo
        img_array = np.array(img) / 255.0  # Normalizar
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        
        # Aquí puedes personalizar el mensaje según la salida de tu modelo
        if predicted_class == 1:
            st.write(f"Predicción: El neumático está en buenas condiciones para ser usado.")
        else:
            st.write(f"Predicción: El neumático NO está en buenas condiciones.")
    except Exception as e:
        st.error(f"Error al hacer la predicción: {str(e)}")
