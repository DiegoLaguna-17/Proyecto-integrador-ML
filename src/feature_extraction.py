import os
import cv2
import pandas as pd
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def extract_features(image_path):
    """Extrae el histograma de color de una imagen."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
 
    img = cv2.resize(img, settings.IMAGE_SIZE)
    

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    

    hist = cv2.calcHist([hsv_img], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist_flatten = cv2.normalize(hist, hist).flatten()
    
    return hist_flatten

def process_dataset():
    """Recorre las carpetas, extrae características y guarda en CSV."""
    features_list = []
    labels_list = []
    
    print("Iniciando extracción de características...")
    
    if not os.path.exists(settings.RAW_DATA_DIR):
        print(f"Error: No se encontró la ruta {settings.RAW_DATA_DIR}")
        print("Revisa que la carpeta de imágenes esté bien ubicada.")
        return

    # Recorrer las 38 carpetas
    for folder_name in os.listdir(settings.RAW_DATA_DIR):
        folder_path = os.path.join(settings.RAW_DATA_DIR, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        print(f"Procesando clase: {folder_name}")
        
        # Procesar cada imagen dentro de la carpeta
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            features = extract_features(image_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(folder_name)
                
    print("Guardando datos estructurados...")
    df = pd.DataFrame(features_list)
    df['label'] = labels_list
    df.to_csv(settings.FEATURES_CSV_PATH, index=False)
    print(f"¡Proceso completado! Archivo guardado en: {settings.FEATURES_CSV_PATH}")

if __name__ == "__main__":
    process_dataset()