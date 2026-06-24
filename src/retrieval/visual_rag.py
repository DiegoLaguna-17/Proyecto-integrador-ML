import os
import json
import numpy as np
import tensorflow as tf
import faiss
from config import settings
from src.utils.config_utils import load_yaml_config

class VisualRAGSystem:
    def __init__(self, config_path="config/retrieval_config.yaml"):
        self.config = load_yaml_config(config_path)
        self.embed_config = self.config['embeddings']
        self.retrieval_config = self.config['retrieval']
        
        model_path = os.path.join(settings.BASE_DIR, self.embed_config['model_path'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo CNN en: {model_path}")
        self.full_model = tf.keras.models.load_model(model_path)

        raw_data_dir = settings.RAW_DATA_DIR
        if os.path.exists(raw_data_dir):
            self.class_names = sorted([f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))])
        else:
            self.class_names = []
            
        latent_layers = []
        for layer in self.full_model.layers:
            latent_layers.append(layer)
            if layer.name == "embedding_latent":
                break
        self.latent_model = tf.keras.Sequential(latent_layers)
        
        index_path = os.path.join(settings.BASE_DIR, self.embed_config['index_path'])
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No se encontró el índice FAISS en: {index_path}")
        self.index = faiss.read_index(index_path)
        
        metadata_path = os.path.join(settings.BASE_DIR, self.embed_config['metadata_path'])
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No se encontró la metadata de imágenes en: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.image_paths = json.load(f)

    def preprocess_image(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            img = tf.io.read_file(image_path_or_array)
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            img = tf.convert_to_tensor(image_path_or_array, dtype=tf.float32)
            if len(img.shape) == 2:
                img = tf.image.grayscale_to_rgb(img)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
                
        img = tf.image.resize(img, (224, 224))
        img = tf.expand_dims(img, axis=0)
        return img

    def query(self, image_path_or_array, k=None):
        if k is None:
            k = self.retrieval_config.get('k', 5)
            
        preprocessed_img = self.preprocess_image(image_path_or_array)
        
        preds = self.full_model.predict(preprocessed_img, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        
        predicted_class = self.class_names[class_idx] if self.class_names else f"Clase {class_idx}"
        
        embedding = self.latent_model.predict(preprocessed_img, verbose=0)
        
        distances, indices = self.index.search(embedding.astype('float32'), k)
        
        similar_images = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.image_paths):
                rel_path = self.image_paths[idx]
                abs_path = os.path.abspath(os.path.join(settings.BASE_DIR, rel_path))
                similar_images.append(abs_path)
                
        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "similar_images": similar_images,
            "distances": [float(d) for d in distances[0]]
        }
