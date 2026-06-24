import streamlit as st
import os
import sys
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.visual_rag import VisualRAGSystem
from config import settings

st.set_page_config(
    page_title="PlantVillage — RAG Visual",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f7faf7 0%, #f0f7f1 100%);
        color: #112d21;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e1efe6;
        box-shadow: 4px 0 24px rgba(45, 106, 79, 0.03);
    }
    section[data-testid="stSidebar"] * {
        color: #112d21 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-top: 1px solid #e1efe6 !important;
    }

    .header-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fdfa 100%);
        border: 1px solid rgba(45, 106, 79, 0.12);
        border-left: 6px solid #2d6a4f;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px rgba(45, 106, 79, 0.04);
    }

    .header-title {
        color: #1b4332;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 10px;
        letter-spacing: -0.75px;
    }

    .header-subtitle {
        color: #4a6b5d;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.65;
        font-weight: 400;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid rgba(45, 106, 79, 0.15);
        border-radius: 16px;
        padding: 24px 28px;
        margin-top: 16px;
        box-shadow: 0 12px 32px rgba(45, 106, 79, 0.06);
    }

    .result-label {
        color: #6c8e7e;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .pred-name {
        color: #1b4332;
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 16px;
        word-break: break-word;
        line-height: 1.3;
    }

    .conf-value {
        color: #2d6a4f;
        font-size: 1.5rem;
        font-weight: 700;
        background: #eef7f2;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-block;
    }

    .gallery-card {
        background: #ffffff;
        border: 1px solid rgba(45, 106, 79, 0.1);
        border-radius: 16px;
        padding: 14px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }

    .gallery-card:hover {
        border-color: #52b788;
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(45, 106, 79, 0.12);
    }

    .gallery-card img {
        border-radius: 10px !important;
    }

    .gallery-case-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #2d6a4f;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        background: #eef7f2;
        padding: 3px 8px;
        border-radius: 20px;
        display: inline-block;
    }

    .gallery-class-name {
        font-size: 0.85rem;
        color: #1b4332;
        font-weight: 600;
        margin-top: 10px;
        min-height: 40px;
        line-height: 1.4;
    }

    .gallery-distance {
        font-size: 0.75rem;
        font-weight: 500;
        color: #52b788;
        margin-top: 6px;
        background: #f0fbf5;
        padding: 2px 6px;
        border-radius: 4px;
        display: inline-block;
    }

    .empty-gallery-slot {
        background-color: rgba(255, 255, 255, 0.5);
        border: 2px dashed #cbdccf;
        border-radius: 16px;
        padding: 46px 12px;
        text-align: center;
        color: #95d5b2;
        font-size: 0.88rem;
        font-weight: 500;
    }

    .empty-result {
        background-color: rgba(255, 255, 255, 0.5);
        border: 2px dashed #cbdccf;
        border-radius: 16px;
        padding: 36px 24px;
        text-align: center;
        color: #52796f;
        font-size: 1.0rem;
        font-weight: 500;
    }

    .section-title {
        color: #1b4332;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 2px solid #cbdccf;
        letter-spacing: -0.3px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f 0%, #1b4332 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(45, 106, 79, 0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(45, 106, 79, 0.3) !important;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #b7e4c7 !important;
        border-radius: 14px !important;
        background-color: #ffffff !important;
        padding: 16px !important;
        box-shadow: 0 6px 20px rgba(45, 106, 79, 0.02) !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #fafdfa !important;
        border: 1px solid #e1efe6 !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #1b4332 !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #2d6a4f !important;
        border: 1px solid #b7e4c7 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #eef7f2 !important;
        border-color: #2d6a4f !important;
    }

    .stSuccess, .stInfo {
        background-color: #eef7f2 !important;
        color: #1b4332 !important;
        border: 1px solid rgba(45, 106, 79, 0.15) !important;
        border-radius: 10px !important;
    }

    .stError {
        border-radius: 10px !important;
    }

    .sidebar-info-box {
        background-color: #f7faf8;
        border: 1px solid #e1efe6;
        border-radius: 12px;
        padding: 16px;
        font-size: 0.88rem;
        color: #1a3c2e;
        line-height: 1.8;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.01);
    }

    .sidebar-status-ok {
        background-color: #eef7f2;
        border: 1px solid #74c69d;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #1b4332;
        font-weight: 600;
        margin-top: 12px;
        box-shadow: 0 4px 12px rgba(116, 198, 157, 0.1);
    }

    .sidebar-status-warn {
        background-color: #fffbeb;
        border: 1px solid #fbbf24;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #78350f;
        font-weight: 600;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <div class="header-title">PlantVillage — RAG Visual de Enfermedades Foliares</div>
    <p class="header-subtitle">
        Sube la fotografía de una hoja de planta para obtener el diagnóstico instantáneo del modelo CNN
        y recuperar los 5 casos más similares indexados mediante embeddings visuales profundos (Visual RAG).
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_system():
    try:
        return VisualRAGSystem()
    except Exception as e:
        st.error(f"Error al inicializar el Sistema Visual RAG: {e}")
        st.info("Asegúrate de haber ejecutado 'python main.py --stage train_cnn' y 'python main.py --stage build_index'.")
        return None

rag_system = get_rag_system()

st.sidebar.markdown("## Detalles")

st.sidebar.markdown("""
<div class="sidebar-info-box">
    <strong>Modelo activo:</strong> Red Neuronal Convolucional (4 bloques)<br>
    <strong>Indice vectorial:</strong> FAISS FlatL2<br>
    <strong>Dimension embedding:</strong> 256<br>
    <strong>Dataset:</strong> Plant_leave_diseases_dataset_with_augmentation<br>
    <strong>Clases:</strong> 39 categorías
</div>
""", unsafe_allow_html=True)

if rag_system is not None:
    st.sidebar.markdown(f"""
<div class="sidebar-status-ok">
    Sistema RAG conectado.<br>{len(rag_system.image_paths):,} imagenes listas para consulta.
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
<div class="sidebar-status-warn">
    Sistema RAG desconectado. Esperando entrenamiento e indexación.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Instrucciones de uso")
st.sidebar.markdown("""
1. Subir una imagen de hoja de planta (JPG o PNG) de 256x256 píxeles.
2. Hacer clic **Identificar** para ejecutar la predicción por CNN.
3. Revisa el diagnóstico CNN y los 5 casos similares recuperados.
""")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">Cargar imagen</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona la fotografía de la hoja:",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        run_query = st.button(
            "Identificar",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("Sube una imagen de hoja de planta.")
        run_query = False

with col2:
    st.markdown('<div class="section-title">Resultados del analisis</div>', unsafe_allow_html=True)

    if run_query and uploaded_file is not None and rag_system is not None:
        with st.spinner("Analizando imagen con CNN y recuperando vecinos más cercanos..."):
            try:
                img_array = np.array(image)
                results = rag_system.query(img_array)

                clean_pred = results["prediction"].replace("___", " — ").replace("_", " ")
                confidence_pct = results["confidence"] * 100

                st.markdown(f"""
<div class="result-card">
    <div class="result-label">Diagnóstico foliar</div>
    <div class="pred-name">{clean_pred}</div>
    <div class="result-label">Confianza de predicción</div>
    <div class="conf-value">{confidence_pct:.2f}%</div>
</div>
""", unsafe_allow_html=True)

                st.session_state["results"] = results
                st.success("Análisis completado con éxito.")

            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

    elif run_query and rag_system is None:
        st.error("El sistema RAG no está disponible. Por favor ejecuta las etapas de entrenamiento e indexación primero.")
    else:
        st.markdown("""
<div class="empty-result">
    Esperando imagen de consulta foliar para mostrar el diagnóstico.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">Casos similares recuperados del histórico (Visual RAG — Top 5)</div>', unsafe_allow_html=True)

if "results" in st.session_state and uploaded_file is not None:
    results        = st.session_state["results"]
    similar_images = results["similar_images"]
    distances      = results["distances"]

    cols = st.columns(5, gap="medium")

    for idx, (col, img_path, dist) in enumerate(zip(cols, similar_images, distances)):
        with col:
            folder_name = os.path.basename(os.path.dirname(img_path))
            clean_class = folder_name.replace("___", " — ").replace("_", " ")

            if os.path.exists(img_path):
                neighbor_img = Image.open(img_path)
                st.markdown(f'<div class="gallery-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="gallery-case-label">Caso #{idx + 1}</div>', unsafe_allow_html=True)
                st.image(neighbor_img, use_container_width=True)
                st.markdown(f'<div class="gallery-class-name">{clean_class}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="gallery-distance">Distancia L2: {dist:.4f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("No se pudo cargar la imagen indexada.")
else:
    cols = st.columns(5)
    for col in cols:
        with col:
            st.markdown("""
<div class="empty-gallery-slot">
    Sin imagen
</div>
""", unsafe_allow_html=True)