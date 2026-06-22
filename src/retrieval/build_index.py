import os
import json
import numpy as np
import tensorflow as tf
import faiss
from config import settings
from src.utils.config_utils import load_yaml_config

def load_and_preprocess_img(path):
    """
    Función de utilidad para cargar y redimensionar imágenes mediante TensorFlow de forma eficiente.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img

def build_vector_index(config_path="config/retrieval_config.yaml", quick_run=False):
    """
    Genera embeddings para el dataset usando la capa intermedia de la CNN y crea el índice FAISS.
    """
    config = load_yaml_config(config_path)
    embed_config = config['embeddings']
    
    # 1. Cargar el modelo CNN entrenado
    model_path = os.path.join(settings.BASE_DIR, embed_config['model_path'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {model_path}. "
            "Por favor ejecuta primero la etapa train_cnn."
        )
        
    print(f"[INFO] Cargando modelo CNN desde: {model_path}")
    full_model = tf.keras.models.load_model(model_path)
    
    # 2. Construir el extractor de características recopilando las capas hasta 'embedding_latent'
    latent_layers = []
    for layer in full_model.layers:
        latent_layers.append(layer)
        if layer.name == "embedding_latent":
            break
    latent_model = tf.keras.Sequential(latent_layers)
    print("[INFO] Extractor de embeddings latentes construido correctamente.")
    
    # 3. Recopilar todas las rutas de imágenes en el dataset
    raw_data_dir = settings.RAW_DATA_DIR
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(
            f"El dataset no existe en: {raw_data_dir}. "
            "Por favor ejecuta primero la etapa prepare_data."
        )
        
    print(f"[INFO] Recopilando imágenes del dataset desde: {raw_data_dir}")
    image_paths = []
    
    # Recorrer subcarpetas
    for folder_name in sorted(os.listdir(raw_data_dir)):
        folder_path = os.path.join(raw_data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Guardar ruta relativa al directorio base para portabilidad
                full_path = os.path.join(folder_path, img_name)
                rel_path = os.path.relpath(full_path, settings.BASE_DIR)
                image_paths.append(rel_path)
                
    total_images = len(image_paths)
    print(f"[INFO] Se encontraron {total_images} imágenes en el dataset.")
    
    # Si es quick_run, limitar a un subconjunto muy pequeño de imágenes para validación rápida
    if quick_run:
        print("[INFO] Quick-run activado. Limitando extracción a 200 imágenes...")
        image_paths = image_paths[:200]
        total_images = len(image_paths)
        
    # 4. Crear dataset de TensorFlow optimizado para cargar las imágenes
    print("[INFO] Iniciando extracción de embeddings en lotes...")
    batch_size = embed_config.get('batch_size', 32)
    
    # Resolver rutas completas para la carga
    full_paths = [os.path.join(settings.BASE_DIR, p) for p in image_paths]
    
    path_dataset = tf.data.Dataset.from_tensor_slices(full_paths)
    image_dataset = path_dataset.map(load_and_preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    image_dataset = image_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Predecir/Extraer embeddings
    embeddings = latent_model.predict(image_dataset, verbose=1)
    print(f"[INFO] Extracción completada. Forma de la matriz de embeddings: {embeddings.shape}")
    
    # 5. Guardar embeddings e índices de ruta
    os.makedirs(os.path.dirname(os.path.join(settings.BASE_DIR, embed_config['embeddings_path'])), exist_ok=True)
    
    embed_save_path = os.path.join(settings.BASE_DIR, embed_config['embeddings_path'])
    np.save(embed_save_path, embeddings)
    print(f"[INFO] Embeddings guardados en: {embed_save_path}")
    
    paths_save_path = os.path.join(settings.BASE_DIR, embed_config['metadata_path'])
    with open(paths_save_path, 'w', encoding='utf-8') as f:
        json.dump(image_paths, f, indent=4)
    print(f"[INFO] Rutas de imágenes guardadas en: {paths_save_path}")
    
    # 6. Construir e inicializar el índice FAISS L2
    print("[INFO] Construyendo índice FAISS FlatL2...")
    dimension = embeddings.shape[1] # Debe ser 128
    
    # Crear el índice FAISS
    index = faiss.IndexFlatL2(dimension)
    
    # Agregar vectores de características al índice (FAISS requiere float32)
    index.add(embeddings.astype('float32'))
    print(f"[INFO] Indexados {index.ntotal} vectores en FAISS.")
    
    # Guardar el índice
    index_save_path = os.path.join(settings.BASE_DIR, embed_config['index_path'])
    faiss.write_index(index, index_save_path)
    print(f"[INFO] Índice FAISS guardado correctamente en: {index_save_path}")

if __name__ == "__main__":
    build_vector_index()
