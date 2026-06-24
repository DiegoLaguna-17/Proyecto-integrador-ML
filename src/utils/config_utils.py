import yaml
import os

def load_yaml_config(file_path):
    """
    Carga un archivo de configuración en formato YAML.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
