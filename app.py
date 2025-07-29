import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

st.title("Clasificador de prendas")

# Cargar modelo
model = load_model("modelo_prendas.h5")

# Cargar el diccionario índice->clase
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Invertirlo para que sea índice (como string) -> clase
indices_to_class = {str(v): k for k, v in class_indices.items()}


# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer imagen y preprocesar
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Añadir batch dimension

    # Mostrar imagen subida
    st.image(img, caption="Imagen subida", use_column_width=True)

    # Predicción
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = indices_to_class[str(pred_index)]
    confidence = preds[0][pred_index]

    st.write(f"**Predicción:** {pred_class}")
    st.write(f"**Confianza:** {confidence:.2f}")
