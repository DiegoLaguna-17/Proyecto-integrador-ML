import os

# Definir la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas de datos
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Asegúrate de que la carpeta extraída dentro de /data se llame 'PlantVillage'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'Plant_leave_diseases_dataset_with_augmentation') 
FEATURES_CSV_PATH = os.path.join(DATA_DIR, 'features.csv')

# Parámetros de procesamiento
IMAGE_SIZE = (128, 128)