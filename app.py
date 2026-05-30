import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(page_title="RosaCervix AI", page_icon="🌸")

# Estilo visual
st.markdown("<h1 style='text-align: center; color: #D53F8C;'>🌸 RosaCervix AI 🌸</h1>", unsafe_allow_html=True)

# Carga del modelo (protegida)
@st.cache_resource
def cargar_modelo():
    if os.path.exists("mejor.pt"):
        return YOLO("mejor.pt")
    return None

model = cargar_modelo()

# Formulario
nombre = st.text_input("Nombre de la paciente:")
archivo_imagen = st.file_uploader("Cargar imagen:", type=["jpg", "png"])

if archivo_imagen:
    img = Image.open(archivo_imagen)
    st.image(img, width=300)
    
    if st.button("Iniciar Análisis"):
        if model:
            with st.spinner("Analizando..."):
                res = model(img)
                # Resultado real
                st.success("✅ ANÁLISIS COMPLETADO")
        else:
            st.error("Error: El modelo no se encontró en GitHub.")
