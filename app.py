import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import urllib.request

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
    </style>
""", unsafe_allow_html=True)

# Cargar tu modelo real (YOLOv8)
@st.cache_resource
def load_yolo_model():
    model_path = "mejor.pt"
    
    # URL DE DESCARGA DIRECTA (Si tu archivo pesa más de 25MB y GitHub lo rechaza)
    # Reemplaza esto con un enlace directo si lo subiste a Drive, Dropbox o Hugging Face
    url_modelo_externo = "https://huggingface.co/marianaordonez12345/RosaCervix-IA/resolve/main/mejor.pt"
    
    try:
        # 1. Intentar buscarlo de forma local en GitHub
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            return YOLO(model_path)
            
        # 2. Si no existe o pesa 0 bytes, intenta descargarlo del link externo
        if url_modelo_externo != "TU_ENLACE_DIRECTO_AQUI":
            with st.spinner("Descargando el modelo de Inteligencia Artificial... Por favor espera."):
                urllib.request.urlretrieve(url_modelo_externo, model_path)
            return YOLO(model_path)
            
        # 3. Si no hay link externo, intenta cargar lo que haya en GitHub
        return YOLO(model_path)
        
    except Exception as e:
        st.error(f"Error al cargar el cerebro de la IA (mejor.pt). Detalle: {e}")
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
            st.error("❌ El modelo de IA no está disponible o el archivo 'mejor.pt' está vacío.")
        else:
            with st.spinner("Analizando muestra biológica con tu modelo de Roboflow..."):
                # Ejecutar predicción real sobre la imagen de Google o microscopio
                resultados = model(img)
                
                # Verificar si el modelo encontró alguna anormalidad o célula clasificada
                if len(resultados[0].boxes) > 0:
                    mejor_box = resultados[0].boxes[0]
                    clase_id = int(mejor_box.cls[0])
                    confianza = float(mejor_box.conf[0]) * 100
                    nombre_clase = model.names[clase_id].lower()
                    
                    # Diagnóstico basado estrictamente en tus dos etiquetas de Roboflow
                    if "normal" in nombre_clase:
                        diagnostico = "CÉLULA NORMAL"
                        st.success(f"✅ **Resultado:** {diagnostico} (Confianza: {confianza:.2f}%)")
                    else:
                        diagnostico = "CARCINOMA DETECTADO"
                        st.error(f"🚨 **Resultado:** {diagnostico} (Confianza: {confianza:.2f}%)")
                else:
                    # Si el modelo no genera cajas de detección (común con imágenes externas muy diferentes)
                    diagnostico = "MUESTRA NO CONCLUYENTE"
                    st.warning("⚠️ **Aviso:** El modelo no detectó patrones familiares en esta imagen externa. La resolución o tinción difiere de tu dataset.")
                    confianza = 0.0
                
                # --- GUARDAR EN EL HISTORIAL (Excel/CSV) ---
                nuevo_registro = {
                    "Fecha": fecha.strftime("%Y-%m-%d"),
                    "Paciente": nombre,
                    "Edad": edad,
                    "Diagnostico": diagnostico,
                    "Confianza": f"{confianza:.2f}%" if confianza > 0 else "N/A"
                }
                
                csv_path = "historial_pacientes.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
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
