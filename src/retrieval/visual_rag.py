import os
import json
import numpy as np
import tensorflow as tf
import faiss
from config import settings
from src.utils.config_utils import load_yaml_config

class VisualRAGSystem:
    def __init__(self, config_path="config/retrieval_config.yaml"):
        """
        Inicializa el sistema RAG Visual cargando el modelo CNN, el índice FAISS y las rutas de imágenes.
        """
        self.config = load_yaml_config(config_path)
        self.embed_config = self.config['embeddings']
        self.retrieval_config = self.config['retrieval']
        
        # 1. Cargar el modelo CNN completo
        model_path = os.path.join(settings.BASE_DIR, self.embed_config['model_path'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo CNN en: {model_path}")
        self.full_model = tf.keras.models.load_model(model_path)
        
        # Obtener las clases del dataset
        # Nota: Keras guarda los metadatos de las clases si entrenamos con image_dataset_from_directory,
        # pero es más seguro deducirlas del directorio de imágenes directamente.
        raw_data_dir = settings.RAW_DATA_DIR
        if os.path.exists(raw_data_dir):
            self.class_names = sorted([f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))])
        else:
            self.class_names = []
            
        # 2. Construir el extractor de embeddings latentes
        latent_layers = []
        for layer in self.full_model.layers:
            latent_layers.append(layer)
            if layer.name == "embedding_latent":
                break
        self.latent_model = tf.keras.Sequential(latent_layers)
        
        # 3. Cargar el índice FAISS
        index_path = os.path.join(settings.BASE_DIR, self.embed_config['index_path'])
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No se encontró el índice FAISS en: {index_path}")
        self.index = faiss.read_index(index_path)
        
        # 4. Cargar la lista de rutas de imágenes indexadas
        metadata_path = os.path.join(settings.BASE_DIR, self.embed_config['metadata_path'])
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No se encontró la metadata de imágenes en: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.image_paths = json.load(f)

    def preprocess_image(self, image_path_or_array):
        """
        Carga y preprocesa una imagen (ruta o array de pixeles) a forma (1, 224, 224, 3).
        """
        if isinstance(image_path_or_array, str):
            # Cargar desde ruta
            img = tf.io.read_file(image_path_or_array)
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            # Asumir que ya es un numpy array (p. ej. de Streamlit)
            img = tf.convert_to_tensor(image_path_or_array, dtype=tf.float32)
            if len(img.shape) == 2:
                # Escala de grises a RGB
                img = tf.image.grayscale_to_rgb(img)
            elif img.shape[2] == 4:
                # RGBA a RGB
                img = img[:, :, :3]
                
        img = tf.image.resize(img, (224, 224))
        img = tf.expand_dims(img, axis=0) # Añadir dimensión de lote (1, 224, 224, 3)
        return img

    def query(self, image_path_or_array, k=None):
        """
        Realiza una consulta RAG Visual completa:
        1. Clasifica la enfermedad foliar y calcula la confianza.
        2. Extrae el embedding latente.
        3. Busca los k vecinos más cercanos en el índice FAISS.
        Retorna la predicción, confianza y las rutas de las 5 imágenes más similares.
        """
        if k is None:
            k = self.retrieval_config.get('k', 5)
            
        # Preprocesar imagen
        preprocessed_img = self.preprocess_image(image_path_or_array)
        
        # 1. Obtener predicción de clasificación
        preds = self.full_model.predict(preprocessed_img, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        
        predicted_class = self.class_names[class_idx] if self.class_names else f"Clase {class_idx}"
        
        # 2. Generar embedding visual
        embedding = self.latent_model.predict(preprocessed_img, verbose=0)
        
        # 3. Buscar en el índice FAISS
        # FAISS requiere float32
        distances, indices = self.index.search(embedding.astype('float32'), k)
        
        # 4. Obtener las rutas de las imágenes vecinas
        similar_images = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.image_paths):
                # Obtener ruta absoluta de la imagen para visualización
                rel_path = self.image_paths[idx]
                abs_path = os.path.abspath(os.path.join(settings.BASE_DIR, rel_path))
                similar_images.append(abs_path)
                
        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "similar_images": similar_images,
            "distances": [float(d) for d in distances[0]]
        }
