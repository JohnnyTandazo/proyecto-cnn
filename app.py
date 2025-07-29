import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

st.title("Clasificador de prendas")

# Cargar diccionario de clases
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
indices_to_class = {v: k for k, v in class_indices.items()}

# Cargar modelo SavedModel
model = tf.saved_model.load("modelo_prendas_savedmodel")
infer = model.signatures["serving_default"]

# Nombre de la salida
output_tensor_name = 'output_0'

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Imagen subida", use_container_width=True)

    preds = infer(tf.constant(img_array))[output_tensor_name].numpy()
    pred_index = np.argmax(preds)
    pred_class = indices_to_class[pred_index]
    confidence = preds[0][pred_index]

    st.write(f"**Predicción:** {pred_class}")
    st.write(f"**Confianza:** {confidence:.2f}")
