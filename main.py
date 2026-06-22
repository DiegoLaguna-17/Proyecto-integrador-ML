import os
import argparse
import sys
import numpy as np
import joblib
import tensorflow as tf
import faiss
import json
from config import settings
from src.utils.config_utils import load_yaml_config
from src.utils.data_utils import extract_dataset, get_dataset_version, load_ml_data, get_cnn_datasets
from src.feature_extraction import process_dataset
from src.training.train_ml import train_ml_baseline
from src.training.train_cnn import train_cnn_pipeline
from src.retrieval.build_index import build_vector_index
from src.retrieval.visual_rag import VisualRAGSystem
from sklearn.metrics import accuracy_score, classification_report

def run_prepare_data():
    print("=========================================")
    print("ETAPA: PREPARE DATA")
    print("=========================================")
    zip_path = os.path.join(settings.DATA_DIR, "raw", "Plant_leaf_diseases_dataset_with_augmentation.zip")
    extract_to = settings.DATA_DIR
    
    # 1. Extraer dataset
    extract_dataset(zip_path, extract_to)
    
    # 2. Correr la extracción de características HSV
    if not os.path.exists(settings.FEATURES_CSV_PATH):
        process_dataset()
    else:
        print(f"[INFO] El archivo de características ya existe en: {settings.FEATURES_CSV_PATH}")

def run_train_ml():
    print("=========================================")
    print("ETAPA: TRAIN ML BASELINE")
    print("=========================================")
    train_ml_baseline()

def run_train_cnn(quick=False):
    print("=========================================")
    print(f"ETAPA: TRAIN CNN (quick={quick})")
    print("=========================================")
    train_cnn_pipeline(quick_train=quick)

def run_evaluate():
    print("=========================================")
    print("ETAPA: EVALUATE MODELS")
    print("=========================================")
    # 1. Evaluar Random Forest Baseline
    rf_model_path = os.path.join(settings.BASE_DIR, "models", "baseline_rf.pkl")
    if os.path.exists(rf_model_path):
        print("[INFO] Evaluando Random Forest Baseline...")
        rf_model = joblib.load(rf_model_path)
        try:
            _, X_val, _, y_val = load_ml_data()
            rf_preds = rf_model.predict(X_val)
            rf_acc = accuracy_score(y_val, rf_preds)
            print(f"--> [RESULTADO RF] exactitud (Accuracy): {rf_acc * 100:.2f}%")
        except Exception as e:
            print(f"[ERROR] No se pudo evaluar RF: {e}")
    else:
        print("[WARN] No se encontró el modelo Random Forest entrenado en models/baseline_rf.pkl")

    # 2. Evaluar CNN
    cnn_model_path = os.path.join(settings.BASE_DIR, "models", "cnn_model.keras")
    if os.path.exists(cnn_model_path):
        print("\n[INFO] Evaluando CNN...")
        cnn_model = tf.keras.models.load_model(cnn_model_path)
        config = load_yaml_config("config/train_config.yaml")
        cnn_params = config['cnn']
        try:
            _, val_dataset, class_names = get_cnn_datasets(
                data_dir=settings.RAW_DATA_DIR,
                img_size=tuple(cnn_params['img_size']),
                batch_size=cnn_params['batch_size'],
                val_split=cnn_params['validation_split'],
                seed=cnn_params['seed']
            )
            # Para evaluación rápida en CLI, tomar un lote
            val_loss, val_acc = cnn_model.evaluate(val_dataset, verbose=1)
            print(f"--> [RESULTADO CNN] exactitud (Accuracy): {val_acc * 100:.2f}% | Pérdida (Loss): {val_loss:.4f}")
        except Exception as e:
            print(f"[ERROR] No se pudo evaluar la CNN: {e}")
    else:
        print("[WARN] No se encontró el modelo CNN entrenado en models/cnn_model.keras")

def run_generate_embeddings(quick=False):
    print("=========================================")
    print(f"ETAPA: GENERATE EMBEDDINGS (quick={quick})")
    print("=========================================")
    build_vector_index(quick_run=quick)

def run_build_index():
    print("=========================================")
    print("ETAPA: BUILD INDEX")
    print("=========================================")
    # En nuestro diseño, build_vector_index genera embeddings e índice FAISS juntos.
    # Verificamos si ya existen los embeddings para simplemente construir el índice,
    # o si ejecutamos la construcción de índice FAISS a partir de los embeddings existentes.
    config = load_yaml_config("config/retrieval_config.yaml")
    embed_config = config['embeddings']
    embed_save_path = os.path.join(settings.BASE_DIR, embed_config['embeddings_path'])
    
    if os.path.exists(embed_save_path):
        print(f"[INFO] Cargando embeddings existentes desde: {embed_save_path}")
        embeddings = np.load(embed_save_path)
        dimension = embeddings.shape[1]
        
        print("[INFO] Construyendo índice FAISS FlatL2 a partir de los embeddings...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        index_save_path = os.path.join(settings.BASE_DIR, embed_config['index_path'])
        faiss.write_index(index, index_save_path)
        print(f"[INFO] Índice FAISS guardado correctamente en: {index_save_path} ({index.ntotal} vectores indexados).")
    else:
        print("[WARN] No se encontraron embeddings pregenerados. Ejecutando pipeline de extracción completo...")
        build_vector_index()

def run_retrieval_test():
    print("=========================================")
    print("ETAPA: RETRIEVAL TEST")
    print("=========================================")
    try:
        # Cargar el sistema RAG
        rag_system = VisualRAGSystem()
        
        # Buscar una imagen de muestra del dataset para probar
        if not rag_system.image_paths:
            print("[ERROR] No hay imágenes indexadas.")
            return
            
        sample_rel_path = rag_system.image_paths[0]
        sample_path = os.path.join(settings.BASE_DIR, sample_rel_path)
        print(f"[INFO] Imagen de prueba seleccionada: {sample_rel_path}")
        
        results = rag_system.query(sample_path)
        
        print("\n=========================================")
        print("RESULTADO DE LA BÚSQUEDA")
        print("=========================================")
        print(f"Predicción de Enfermedad: {results['prediction']}")
        print(f"Confianza: {results['confidence'] * 100:.2f}%")
        print("\nImágenes similares encontradas:")
        for idx, img_path in enumerate(results['similar_images'], 1):
            print(f"  [{idx}] {os.path.basename(img_path)} (Ruta: {img_path})")
        print("=========================================")
    except Exception as e:
        print(f"[ERROR] Falló la prueba de recuperación visual RAG: {e}")

def main():
    parser = argparse.ArgumentParser(description="Orquestador del pipeline MLOps para Clasificación y RAG de PlantVillage")
    parser.add_argument(
        "--stage", 
        type=str, 
        required=True,
        choices=[
            "prepare_data", 
            "train_ml", 
            "train_cnn", 
            "evaluate", 
            "generate_embeddings", 
            "build_index", 
            "retrieval_test", 
            "full_pipeline"
        ],
        help="Etapa del pipeline a ejecutar."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ejecuta las etapas de CNN y embeddings con un subconjunto mínimo para verificación rápida en CPU."
    )
    
    args = parser.parse_args()
    
    stage = args.stage
    quick = args.quick
    
    if stage == "prepare_data":
        run_prepare_data()
    elif stage == "train_ml":
        run_train_ml()
    elif stage == "train_cnn":
        run_train_cnn(quick=quick)
    elif stage == "evaluate":
        run_evaluate()
    elif stage == "generate_embeddings":
        run_generate_embeddings(quick=quick)
    elif stage == "build_index":
        run_build_index()
    elif stage == "retrieval_test":
        run_retrieval_test()
    elif stage == "full_pipeline":
        print("[INFO] Iniciando ejecución completa del pipeline...")
        run_prepare_data()
        run_train_ml()
        run_train_cnn(quick=quick)
        run_evaluate()
        run_generate_embeddings(quick=quick)
        run_build_index()
        run_retrieval_test()
        print("[INFO] ¡Pipeline completado con éxito!")

if __name__ == "__main__":
    main()
