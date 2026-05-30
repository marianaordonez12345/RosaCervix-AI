import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os

# Configuración de la página web (Título y pestaña)
st.set_page_config(page_title="RosaCervix AI", page_icon="🌸", layout="centered")

# Aplicar diseño CSS personalizado para que sea hermoso y Rosa
st.markdown("""
    <style>
    .stApp {
        background-color: #FFF5F5;
    }
    h1 {
        color: #D53F8C !important;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
    }
    h3 {
        color: #B83280 !important;
    }
    .stButton>button {
        background-color: #D53F8C !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #B83280 !important;
        color: white !important;
    }
    .css-10trblm {
        color: #B83280 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar tu modelo de Colab (YOLOv8)
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"No se encontró el archivo 'best.pt'. Asegúrate de que esté en la misma carpeta. Error: {e}")
        return None

model = load_yolo_model()

# Encabezado de la aplicación
st.markdown("<h1>🌸 RosaCervix AI 🌸</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #718096;'>Sistema Inteligente para la Detección Temprana de Carcinoma de Cuello Uterino</p>", unsafe_allow_html=True)
st.write("---")

# Sección 1: Datos de la Paciente
st.markdown("### 📋 Datos de la Paciente")
col1, col2 = st.columns(2)

with col1:
    nombre = st.text_input("Nombre Completo:")
    edad = st.number_input("Edad:", min_value=1, max_value=120, value=25)

with col2:
    fecha = st.date_input("Fecha del Análisis:", datetime.now())
    archivo_imagen = st.file_uploader("📁 Cargar imagen de la célula:", type=["jpg", "jpeg", "png"])

st.write("---")

# Sección 2: Análisis e Inteligencia Artificial
if archivo_imagen is not None:
    img = Image.open(archivo_imagen)
    
    # Mostrar la imagen cargada de forma bonita
    st.markdown("### 🔬 Muestra Cargada")
    st.image(img, caption="Frotis Cervical", width=300)
    
    # Botón Rosa para Analizar
    if st.button("✨ Iniciar Análisis con IA"):
        if not nombre:
            st.warning("⚠️ Por favor, introduce el nombre de la paciente antes de continuar.")
        elif model is None:
            st.error("❌ El modelo IA no está cargado correctamente.")
        else:
            with st.spinner("Analizando muestra con el modelo de Colab..."):
                # Ejecutar predicción con tu best.pt
                resultados = model(img)
                
                if len(resultados[0].boxes) == 0:
                    st.info("No se detectaron anomalías claras en esta muestra específica.")
                else:
                    # Obtener la predicción más confiable
                    mejor_box = resultados[0].boxes[0]
                    clase_id = int(mejor_box.cls[0])
                    confianza = float(mejor_box.conf[0]) * 100
                    nombre_clase = model.names[clase_id]
                    
                    # Determinar el diagnóstico
                    if "carcinoma" in nombre_clase.lower() or "abnormal" in nombre_clase.lower():
                        diagnostico = "CARCINOMA DETECTADO / ANORMAL"
                        st.error(f"🚨 **Resultado:** {diagnostico} (Confianza: {confianza:.2f}%)")
                    else:
                        diagnostico = "CÉLULA NORMAL"
                        st.success(f"✅ **Resultado:** {diagnostico} (Confianza: {confianza:.2f}%)")
                    
                    # --- GUARDAR EN EL HISTORIAL (Excel/CSV) ---
                    nuevo_registro = {
                        "Fecha": fecha.strftime("%Y-%m-%d"),
                        "Paciente": nombre,
                        "Edad": edad,
                        "Diagnostico": diagnostico,
                        "Confianza": f"{confianza:.2f}%"
                    }
                    
                    csv_path = "historial_pacientes.csv"
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        # Reemplazo de append por concat para evitar advertencias de versiones futuras de pandas
                        df = pd.concat([df, pd.DataFrame([nuevo_registro])], ignore_index=True)
                    else:
                        df = pd.DataFrame([nuevo_registro])
                        
                    df.to_csv(csv_path, index=False)
                    st.toast("💾 ¡Datos guardados en el historial médico!", icon="📥")

st.write("---")

# Sección 3: Ver Historial Clínico
if st.checkbox("📊 Ver historial de pacientes guardados"):
    csv_path = "historial_pacientes.csv"
    if os.path.exists(csv_path):
        df_historial = pd.read_csv(csv_path)
        st.dataframe(df_historial, use_container_width=True)
    else:
        st.info("Aún no hay pacientes registrados en el historial.")