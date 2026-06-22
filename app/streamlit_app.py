import streamlit as st
import os
import sys
from PIL import Image
import numpy as np

# Add base path for project module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.visual_rag import VisualRAGSystem
from config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantVillage — Diagnóstico Visual de Enfermedades",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — light botanical green theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global reset ────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ──────────────────────────────────────────────────── */
    .stApp {
        background-color: #f4f9f4;
        color: #1a3c2e;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #b7e4c7;
    }
    section[data-testid="stSidebar"] * {
        color: #1a3c2e !important;
    }

    /* ── Header card ─────────────────────────────────────────────────────── */
    .header-card {
        background-color: #ffffff;
        border: 1px solid #b7e4c7;
        border-left: 5px solid #2d6a4f;
        border-radius: 10px;
        padding: 28px 32px;
        margin-bottom: 28px;
    }

    .header-title {
        color: #1b4332;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #52796f;
        font-size: 1.0rem;
        max-width: 820px;
        line-height: 1.6;
    }

    /* ── Result card ─────────────────────────────────────────────────────── */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #b7e4c7;
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 14px;
    }

    .result-label {
        color: #52796f;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    .pred-name {
        color: #1b4332;
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 12px;
        word-break: break-word;
    }

    .conf-value {
        color: #2d6a4f;
        font-size: 1.35rem;
        font-weight: 600;
    }

    /* ── Gallery cards ───────────────────────────────────────────────────── */
    .gallery-card {
        background-color: #ffffff;
        border: 1px solid #d8f3dc;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .gallery-card:hover {
        border-color: #52b788;
        box-shadow: 0 4px 16px rgba(45, 106, 79, 0.15);
    }

    .gallery-case-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #2d6a4f;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .gallery-class-name {
        font-size: 0.8rem;
        color: #1b4332;
        font-weight: 500;
        margin-top: 6px;
        min-height: 36px;
    }

    .gallery-distance {
        font-size: 0.72rem;
        color: #74c69d;
        margin-top: 4px;
    }

    /* ── Empty placeholder ───────────────────────────────────────────────── */
    .empty-gallery-slot {
        background-color: #f4f9f4;
        border: 1px dashed #b7e4c7;
        border-radius: 10px;
        padding: 40px 10px;
        text-align: center;
        color: #95d5b2;
        font-size: 0.85rem;
    }

    /* ── Empty result placeholder ────────────────────────────────────────── */
    .empty-result {
        background-color: #f4f9f4;
        border: 1px dashed #b7e4c7;
        border-radius: 10px;
        padding: 28px 20px;
        text-align: center;
        color: #52796f;
        font-size: 0.95rem;
    }

    /* ── Section divider ─────────────────────────────────────────────────── */
    .section-title {
        color: #1b4332;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid #b7e4c7;
    }

    /* ── Streamlit component overrides ───────────────────────────────────── */
    .stButton > button {
        background-color: #2d6a4f;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1b4332;
        color: #ffffff;
    }

    div[data-testid="stFileUploader"] {
        border: 1px solid #b7e4c7;
        border-radius: 8px;
        background-color: #ffffff;
    }

    .stSuccess {
        background-color: #d8f3dc !important;
        color: #1b4332 !important;
        border-radius: 8px;
    }

    .stError {
        border-radius: 8px;
    }

    .stInfo {
        background-color: #d8f3dc !important;
        color: #1b4332 !important;
        border-radius: 8px;
    }

    /* ── Sidebar info box ────────────────────────────────────────────────── */
    .sidebar-info-box {
        background-color: #f4f9f4;
        border: 1px solid #b7e4c7;
        border-radius: 8px;
        padding: 14px 16px;
        font-size: 0.87rem;
        color: #1a3c2e;
        line-height: 1.7;
    }

    .sidebar-status-ok {
        background-color: #d8f3dc;
        border: 1px solid #74c69d;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.87rem;
        color: #1b4332;
        font-weight: 500;
        margin-top: 10px;
    }

    .sidebar-status-warn {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.87rem;
        color: #664d03;
        font-weight: 500;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="header-title">PlantVillage — Diagnóstico Visual de Enfermedades Foliares</div>
    <p class="header-subtitle">
        Sube la fotografía de una hoja de planta para obtener el diagnóstico instantáneo del modelo CNN
        y recuperar los casos más similares indexados mediante embeddings visuales profundos (Visual RAG).
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cache & load RAG system
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_rag_system():
    try:
        return VisualRAGSystem()
    except Exception as e:
        st.error(f"Error al inicializar el Sistema Visual RAG: {e}")
        st.info("Asegúrate de haber ejecutado 'python main.py --stage train_cnn' y 'python main.py --stage build_index'.")
        return None

rag_system = get_rag_system()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Panel MLOps y RAG")

st.sidebar.markdown("""
<div class="sidebar-info-box">
    <strong>Modelo activo:</strong> CNN personalizada (4 bloques Conv)<br>
    <strong>Indice vectorial:</strong> FAISS FlatL2<br>
    <strong>Dimension embedding:</strong> 256<br>
    <strong>Dataset:</strong> PlantVillage<br>
    <strong>Clases:</strong> 39 categorias (sanas y enfermas)
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
    Sistema RAG desconectado. Esperando entrenamiento e indexacion.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Instrucciones de uso")
st.sidebar.markdown("""
1. Sube una imagen de hoja de planta (JPG o PNG).
2. Pulsa **Identificar y Buscar** para ejecutar el analisis.
3. Revisa el diagnostico CNN y los 5 casos similares recuperados.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Main layout — two columns
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">Cargar imagen foliar</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona la fotografia de la hoja:",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        run_query = st.button(
            "Identificar y Buscar casos similares",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("Sube una imagen de hoja de planta para comenzar.")
        run_query = False

with col2:
    st.markdown('<div class="section-title">Resultados del analisis</div>', unsafe_allow_html=True)

    if run_query and uploaded_file is not None and rag_system is not None:
        with st.spinner("Analizando imagen con CNN y recuperando vecinos mas cercanos..."):
            try:
                img_array = np.array(image)
                results = rag_system.query(img_array)

                clean_pred = results["prediction"].replace("___", " — ").replace("_", " ")
                confidence_pct = results["confidence"] * 100

                st.markdown(f"""
<div class="result-card">
    <div class="result-label">Diagnostico foliar</div>
    <div class="pred-name">{clean_pred}</div>
    <div class="result-label">Confianza de prediccion</div>
    <div class="conf-value">{confidence_pct:.2f}%</div>
</div>
""", unsafe_allow_html=True)

                st.session_state["results"] = results
                st.success("Analisis completado con exito.")

            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

    elif run_query and rag_system is None:
        st.error("El sistema RAG no esta disponible. Por favor ejecuta las etapas de entrenamiento e indexacion primero.")
    else:
        st.markdown("""
<div class="empty-result">
    Esperando imagen de consulta foliar para mostrar el diagnostico.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Similar Cases Gallery (Visual RAG — Top-5 nearest neighbours)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Casos similares recuperados del historico (Visual RAG — Top 5)</div>', unsafe_allow_html=True)

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
                st.image(neighbor_img, use_column_width=True)
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
    Sin imagen vecina
</div>
""", unsafe_allow_html=True)
