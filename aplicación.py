import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="RosaCervix AI", page_icon="🌸")
st.title("🌸 RosaCervix AI")

# Cargamos el modelo
model = YOLO("mejor.pt")

archivo = st.file_uploader("Cargar imagen", type=["jpg", "png"])

if archivo:
    img = Image.open(archivo)
    st.image(img, use_container_width=True)
    
    if st.button("Analizar"):
        with st.spinner("Procesando..."):
            resultados = model(img)
            st.success("✅ Análisis completado.")
