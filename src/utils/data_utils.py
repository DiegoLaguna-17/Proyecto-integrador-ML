import os
import zipfile
import hashlib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import settings

def extract_dataset(zip_path, extract_to):
    """
    Descomprime el archivo zip en el directorio especificado.
    """
    if os.path.exists(settings.RAW_DATA_DIR):
        print(f"[INFO] El dataset ya está extraído en: {settings.RAW_DATA_DIR}")
        return

    print(f"[INFO] Iniciando extracción de {zip_path} en {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extraer todo
        zip_ref.extractall(extract_to)
    
    print(f"[INFO] ¡Extracción completada con éxito!")

def get_dataset_version(zip_path):
    """
    Genera un hash MD5 a partir del archivo zip del dataset para usar como versión.
    """
    if not os.path.exists(zip_path):
        return "unknown_no_zip_file"
        
    print("[INFO] Calculando la versión del dataset (hash MD5)...")
    hash_md5 = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    version = hash_md5.hexdigest()
    print(f"[INFO] Versión del dataset identificada: {version}")
    return version

def load_ml_data(csv_path=settings.FEATURES_CSV_PATH, test_size=0.2, random_state=42):
    """
    Carga las características extraídas (histogramas HSV) y las divide en train/val.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de características en: {csv_path}. "
            "Por favor ejecuta primero la etapa prepare_data."
        )
    
    print(f"[INFO] Cargando datos estructurados de: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Separar características (X) y etiquetas (y)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # Dividir en train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_val, y_train, y_val

def get_cnn_datasets(data_dir=settings.RAW_DATA_DIR, img_size=(224, 224), batch_size=32, val_split=0.2, seed=42):
    """
    Carga el conjunto de datos de imágenes para la CNN y retorna datasets de TensorFlow.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"No existe el directorio de imágenes: {data_dir}. "
            "Por favor ejecuta primero la etapa prepare_data."
        )
        
    print(f"[INFO] Cargando subconjunto de entrenamiento desde: {data_dir}")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        shuffle=True,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    print(f"\n[INFO] Cargando subconjunto de validación desde: {data_dir}")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        shuffle=True,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_dataset.class_names
    
    # Optimizar el rendimiento de carga con prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.shuffle(100).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset, class_names
