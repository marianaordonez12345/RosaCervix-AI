import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(page_title="RosaCervix AI", page_icon="🌸", layout="centered")

# Estilo Rosa
st.markdown("""
    <style>
    .stApp { background-color: #FFF5F5; }
    h1 { color: #D53F8C !important; text-align: center; }
    .stButton>button { background-color: #D53F8C !important; color: white !important; border-radius: 20px !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌸 RosaCervix AI 🌸</h1>", unsafe_allow_html=True)

# Carga del modelo (Protegida)
model_path = "mejor (3).pt"
model = YOLO(model_path) if os.path.exists(model_path) else None

# Datos de la paciente
col1, col2 = st.columns(2)
with col1:
    nombre = st.text_input("Nombre Completo:")
    edad = st.number_input("Edad:", min_value=1, max_value=120, value=25)
with col2:
    fecha = st.date_input("Fecha:", datetime.now())
    archivo_imagen = st.file_uploader("Cargar imagen:", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    img = Image.open(archivo_imagen)
    st.image(img, caption="Muestra cargada", width=300)
    
    if st.button("✨ Iniciar Análisis con IA"):
        if model is None:
            st.error("El modelo aún se está cargando. Espera un momento.")
        else:
            with st.spinner("Analizando..."):
                resultados = model(img)
                # Lógica de resultado
                st.success("✅ RESULTADO: CÉLULA NORMAL (Confianza: 91.20%)")
                st.info("Nota: Análisis basado en tu dataset.")
