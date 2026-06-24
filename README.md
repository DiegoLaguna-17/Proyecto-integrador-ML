# Clasificación de Fitopatologías - PlantVillage Dataset

Sistema completo de detección automatizada de enfermedades en plantas con **Machine Learning clásico**, **Deep Learning (CNN)**, **trazabilidad MLOps con MLflow** y **Recuperación Visual Aumentada (Visual RAG)**

---

## Descripción del Proyecto

Este repositorio contiene un pipeline de ciencia de datos que clasifica **61,486 imágenes de hojas de plantas** del dataset **PlantVillage** en **39 categorías** (sanas y enfermas), comparando modelos ML clásicos, una red neuronal convolucional CNN, y recuperación visual de casos similares mediante un sistema RAG visual.

### ¿Qué resuelve el sistema RAG?

Dada una foto de una hoja de planta, el sistema:

1. **Predice** la enfermedad (o salud) de la planta.
2. **Muestra la confianza** de la predicción.
3. **Recupera** las 5 imágenes más similares del dataset usando embeddings visuales + FAISS.

---

## Estructura del Repositorio

```
Proyecto-integrador-ML/
│
├── app/
│   └── streamlit_app.py          ← Interfaz web Streamlit (Visual RAG Dashboard)
│
├── config/
│   ├── settings.py               ← Rutas base del proyecto
│   ├── train_config.yaml         ← Parámetros de entrenamiento (RF, CNN, MLflow)
│   └── retrieval_config.yaml     ← Parámetros de índice FAISS y recuperación
│
├── data/
│   ├── raw/
│   │   └── Plant_leaf_diseases_dataset_with_augmentation.zip
│   ├── Plant_leave_diseases_dataset_with_augmentation/
│   │   ├── Apple___Apple_scab/   ← Carpetas de clases (39 en total)
│   │   └── ...
│   └── features.csv              ← Histogramas HSV extraídos (generado por prepare_data)
│
├── mlruns/                       ← Tracking local de MLflow (generado automáticamente)
│
├── models/
│   ├── baseline_rf.pkl           ← Modelo Random Forest entrenado (generado por train_ml)
│   ├── cnn_model.keras           ← Modelo CNN entrenado (generado por train_cnn)
│   ├── embeddings.npy            ← Embeddings del dataset (generado por generate_embeddings)
│   ├── faiss_index.bin           ← Índice vectorial FAISS (generado por build_index)
│   └── image_paths.json          ← Rutas de imágenes indexadas (generado por build_index)
│
├── notebooks/                    ← Notebooks originales de referencia
│   ├── AUTOML.ipynb
│   ├── CNN_platn_dicesase.ipynb
│   ├── Clustering_Visual.ipynb
│   ├── KNN_model.ipynb
│   ├── RF_model.ipynb
│   └── SVC_model.ipynb
│
├── reports/                      ← Reportes y métricas generadas (artefactos MLflow)
│
├── src/
│   ├── feature_extraction.py     ← Extractor de histogramas HSV con OpenCV
│   ├── mlops/
│   │   ├── __init__.py
│   │   └── mlflow_utils.py       ← Logging de métricas y artefactos a MLflow
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── build_index.py        ← Construcción del índice FAISS
│   │   └── visual_rag.py         ← Sistema de recuperación visual RAG
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_cnn.py          ← Entrenamiento de la CNN con TensorFlow/Keras
│   │   └── train_ml.py           ← Entrenamiento de los modelos de ML Clásicos
│   └── utils/
│       ├── __init__.py
│       ├── config_utils.py       ← Cargador de archivos YAML
│       └── data_utils.py         ← Utilidades de carga y preprocesamiento de datos
│
├── main.py                       ← Orquestador CLI del pipeline por etapas
├── README.md
└── REQUIREMENTS.txt
```

---

## Resultados del Proyecto

### Fase 1, 2 — Machine Learning Clásico

| Estrategia            | Algoritmo            | Accuracy (Val) | F1-Score |
| :-------------------- | :------------------- | :------------- | :------- |
| **Manual (Baseline)** | SVC (Pipeline PCA)   | **87.49%**     | 87.10%   |
| **AutoML Benchmark**  | Extra Trees (ET)     | **87.02%**     | 86.73%   |
| **Manual**            | KNN (RAPIDS GPU)     | 86.00%         | 86.00%   |
| **Manual**            | Random Forest        | 78.00%         | 77.00%   |
| **No Supervisado**    | K-Means (Clustering) | 80.00%         | 80.00%   |

### Fase 3 — Deep Learning CNN

| Modelo         | Accuracy Val (Época 12) | Val Loss |
| :------------- | :---------------------- | :------- |
| **CNN custom** | **93.48%**              | 0.2505   |

> **Conclusión:** La CNN supera al techo predictivo (~87.5%) de los modelos clásicos, validando la transición a Deep Learning. Su capa latente `embedding_latent` (dim=128) alimenta el sistema RAG Visual, sin embargo, el modelo CNN aún no se encuentra en su capacidad más óptima debido a limitaciones de hardware para entrenamiento.

---

## Stack Tecnológico

| Componente         | Tecnología                             |
| :----------------- | :------------------------------------- |
| ML Clásico         | scikit-learn (Random Forest, SVC, KNN) |
| Deep Learning      | TensorFlow / Keras                     |
| MLOps              | MLflow (tracking local)                |
| Vector DB          | FAISS                                  |
| Interfaz Web RAG   | Streamlit                              |
| Feature Extraction | OpenCV (Histogramas HSV)               |
| Config             | YAML (PyYAML)                          |
| Entorno            | Python 3.13 + venv                     |

---

## Requisitos e Instalación

### Prerrequisitos

- Python 3.10 o superior (probado en Python 3.13)
- ~2 GB de espacio en disco para el dataset + modelos

### 1. Clonar el repositorio

```bash
git clone https://github.com/DiegoLaguna-17/Proyecto-integrador-ML.git
cd Proyecto-integrador-ML
```

### 2. Crear y activar el entorno virtual

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Actualizar pip e instalar dependencias base

```bash
python -m pip install --upgrade pip setuptools wheel

pip install -r REQUIREMENTS.txt
```

O, en caso de que el archivo de requirements no funcione:

```bash
pip install tensorflow mlflow faiss-cpu streamlit pyyaml opencv-python scikit-learn pandas numpy matplotlib seaborn joblib
```

> **Nota:** `tensorflow` para Windows nativo sólo soporta CPU (≥ TF 2.11). Para GPU se requiere WSL2 o el plugin DirectML. La CNN puede entrenarse en CPU aunque a menor velocidad — ajustar el número de epochs en `config/train_config.yaml`.

### 4. Colocar el dataset

Asegurarse de que el archivo ZIP del dataset se encuentre en:

```
data/raw/Plant_leaf_diseases_dataset_with_augmentation.zip
```

---

## Fase 4 - MLflow (Seguimiento de Experimentos) + RAG

### Iniciar el servidor de MLflow UI

```bash
mlflow server --backend-store-uri sqlite:///mlflow_local.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Abrir el navegador en: **http://127.0.0.1:5000**

El experimento registrado se llama: `PlantVillage_Disease_Classification`

### ¿Qué registra MLflow?

Cada run de entrenamiento registra automáticamente:

- **Parámetros**: hiperparámetros, tamaño de imagen, épocas, learning rate, etc.
- **Métricas**: Accuracy, Precision, Recall, F1 (weighted y macro)
- **Artefactos**:
  - `classification_report_*.txt` — reporte por clase
  - `confusion_matrix_*.png` — mapa de calor de la matriz de confusión
  - `config_*.json` — snapshot de la configuración usada
  - `model/` — modelo serializado (sklearn pkl / Keras SavedModel)
- **Metadata**: versión del dataset (hash MD5), tipo de modelo, nombre del run

---

## Pasos de Reproducción (Pipeline por Etapas)

Todos los comandos se ejecutan desde la raíz del proyecto con el entorno virtual activado y el servidor mlflow corriendo.

### Etapa 1 — Preparar el dataset

Extrae el ZIP y genera el archivo `data/features.csv` con histogramas HSV.

```bash
python main.py --stage prepare_data
```

### Etapa 2 — Entrenar el modelo ML Baseline (Random Forest) y Modelos Clásicos

Entrena el Random Forest sobre los histogramas HSV, entrena los modelos clásicos y loguea el experimento a MLflow.

```bash
python main.py --stage train_ml
```

### Etapa 3 — Entrenar la CNN

Entrena la red convolucional desde cero. Usa `--quick` para una prueba rápida (1 época, pocos batches).

```bash
# Entrenamiento completo (recomendado, puede tomar horas en CPU)
python main.py --stage train_cnn

# Prueba rápida (verificación del pipeline, ~2 minutos en CPU)
python main.py --stage train_cnn --quick
```

### Etapa 4 — Evaluar modelos

Muestra métricas de validación de ambos modelos en consola.

```bash
python main.py --stage evaluate
```

### Etapa 5 — Generar embeddings visuales

Extrae los vectores latentes de la CNN para todas las imágenes del dataset.

```bash
# Extracción completa (~61,000 imágenes — puede tomar 30+ minutos en CPU)
python main.py --stage generate_embeddings

# Extracción rápida (200 imágenes para pruebas)
python main.py --stage generate_embeddings --quick
```

### Etapa 6 — Construir el índice FAISS

Indexa los embeddings en una base de datos vectorial para búsqueda por similitud.

```bash
python main.py --stage build_index
```

### Etapa 7 — Probar la recuperación visual

Ejecuta una búsqueda de prueba con una imagen del dataset para verificar el sistema RAG.

```bash
python main.py --stage retrieval_test
```

### Pipeline completo (todas las etapas en secuencia)

```bash
# Pipeline completo con CNN + embeddings rápidos (para verificación)
python main.py --stage full_pipeline --quick

# Pipeline completo en producción (tarda varias horas en CPU)
python main.py --stage full_pipeline
```

---

## Interfaz Streamlit — Visual RAG Dashboard

### Iniciar la aplicación

```bash
streamlit run app/streamlit_app.py
```

Abre el navegador en: **http://localhost:8501**

### Funcionalidades

- **Subida de imagen**: Arrastra o selecciona una foto de hoja de planta
- **Predicción CNN**: Muestra la clase de enfermedad detectada y la confianza
- **Recuperación Visual RAG**: Muestra las 5 imágenes más similares del dataset con su distancia L2

> **Prerequisito**: Deben haberse ejecutado las etapas `train_cnn`, `generate_embeddings` y `build_index` previamente.

---

## Arquitectura del Sistema Visual RAG

```
Imagen de consulta
        │
        ▼
┌──────────────────┐
│  Preprocessing   │  → Resize 224×224, Float32
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   CNN Forward    │  → Rescaling(1/255) → Conv2D(32) → Conv2D(64) → Conv2D(128)
│     Pass         │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────────────┐
│ Pred. │  │ embedding_latent │ → Vector 128D (GlobalAveragePooling2D)
│ Class │  └────────┬─────────┘
│ Conf. │           │
└───────┘           ▼
              ┌─────────────┐
              │ FAISS Index │ → IndexFlatL2 (distancia Euclidiana)
              └─────┬───────┘
                    │
                    ▼
         Top-5 Imágenes Similares
```

---

## Archivos Generados por el Pipeline

| Archivo                        | Generado en etapa         | Descripción                             |
| :----------------------------- | :------------------------ | :-------------------------------------- |
| `data/features.csv`            | `prepare_data`            | Histogramas HSV de todas las imágenes   |
| `models/baseline_rf.pkl`       | `train_ml`                | Modelo Random Forest serializado        |
| `models/clustering_kmeans.pkl` | `train_ml`                | Modelo Clustering Kmeans serializado    |
| `models/knn_model.pkl`         | `train_ml`                | Modelo KNN serializado                  |
| `models/svc_model.pkl`         | `train_ml`                | Modelo SVC serializado                  |
| `models/cnn_model.keras`       | `train_cnn`               | Modelo CNN Keras completo               |
| `models/embeddings.npy`        | `generate_embeddings`     | Matriz de embeddings latentes (N × 128) |
| `models/faiss_index.bin`       | `build_index`             | Índice vectorial FAISS                  |
| `models/image_paths.json`      | `generate_embeddings`     | Lista de rutas relativas de imágenes    |
| `reports/*.txt / *.png`        | `train_ml` / `train_cnn`  | Reportes y matrices de confusión        |
| `mlruns/`                      | Todas las etapas de train | Artefactos y metadata de MLflow         |

---

## Notas Importantes

### TensorFlow en Windows (CPU)

TensorFlow ≥ 2.11 no soporta GPU en Windows nativo. Los mensajes `oneDNN custom operations are on` son informativos y no afectan el funcionamiento. Para usar GPU en Windows instala WSL2 o el plugin `tensorflow-directml`.

### Tiempo de ejecución estimado (CPU)

| Etapa                  | Tiempo estimado (CPU) |
| :--------------------- | :-------------------- |
| `prepare_data`         | 10–30 min             |
| `train_ml`             | 5–15 min              |
| `train_cnn` (5 épocas) | 2–6 horas             |
| `generate_embeddings`  | 30–60 min             |
| `build_index`          | < 1 min               |

### Flag `--quick` para desarrollo

Usa `--quick` en `train_cnn` y `generate_embeddings` para verificar que el pipeline funciona end-to-end en minutos antes de ejecutar el entrenamiento completo.

### Reproducibilidad

Todos los modelos se entrenan con `random_state=42` y `seed=42`. Los runs de MLflow son nombrados descriptivamente (no aleatoriamente), garantizando trazabilidad completa de cada experimento.

### Dataset — Versión y Trazabilidad

El hash MD5 del archivo ZIP del dataset se calcula automáticamente al inicio de cada run de entrenamiento y se registra en MLflow como parámetro `dataset_version`, garantizando la trazabilidad de la versión de los datos.

---

## Notebooks de Referencia

Los siguientes notebooks son la fuente de verdad conceptual del proyecto. Los scripts de este repositorio reimplementan su lógica de forma reproducible y modular:

| Notebook                   | Contenido                                         |
| :------------------------- | :------------------------------------------------ |
| `CNN_platn_dicesase.ipynb` | Arquitectura CNN, entrenamiento, embeddings y RAG |
| `RF_model.ipynb`           | Random Forest con RAPIDS GPU + histogramas HSV    |
| `SVC_model.ipynb`          | Pipeline manual SVC + PCA                         |
| `KNN_model.ipynb`          | KNN con RAPIDS GPU                                |
| `AUTOML.ipynb`             | Benchmarking con PyCaret 3.4                      |
| `Clustering_Visual.ipynb`  | K-Means + PCA para análisis no supervisado        |

---

## Propuesta de Monitoreo y Reentrenamiento

Para garantizar la estabilidad, robustez y mejora continua del sistema en un entorno productivo, se define la siguiente estrategia:

### 1. Estrategia de Monitoreo

El sistema de monitoreo recopilará y analizará de forma continua los datos de inferencia (imágenes de consulta e inputs de usuario):

- **Monitoreo por Clase y Categoría**: Registro de la distribución de consultas recibidas en producción. Si la distribución de cultivos (ej. manzana, papa, tomate) varía drásticamente respecto al dataset de entrenamiento, se identificará un desvío de datos (_data drift_).
- **Monitoreo por Precisión (Métricas de Negocio)**: Implementación de un flujo de etiquetado manual aleatorio del 5% de las imágenes de consultas reales (_ground truth_ obtenido por expertos agrónomos). Esto permitirá recalcular semanalmente la precisión, recall y F1-score del modelo en producción. Se activarán alertas automatizadas si la precisión ponderada decae por debajo del **85%**.
- **Monitoreo de Calidad RAG**: Trazabilidad de la distancia euclidiana mínima ($L_2$) devuelta por FAISS en las consultas de recuperación visual. Una distancia promedio inusualmente alta indica que el usuario está subiendo imágenes de plantas u objetos fuera de distribución (clases no indexadas).

### 2. Estrategia de Actualización y Reentrenamiento

La actualización del modelo se realizará bajo dos modalidades:

- **Reentrenamiento Programado**: Actualizaciones semestrales del pipeline utilizando nuevas muestras recolectadas del campo y etiquetadas por expertos.
- **Reentrenamiento por Evento (Gatillos/Triggers)**:
  - Caída de la precisión ponderada por debajo del **85%** en las muestras monitoreadas.
  - Detección de _concept drift_ sostenido durante más de dos semanas.
- **Flujo de Despliegue de Actualizaciones**:
  1. Incorporar el nuevo lote de datos validados al dataset y recalcular el hash de versión.
  2. Ejecutar el orquestador en su flujo completo: `python main.py --stage full_pipeline`.
  3. Evaluar el desempeño en MLflow comparando el nuevo run contra el actual (_champion vs. challenger_).
  4. Si las métricas mejoran y no hay degradación en clases críticas, promover el nuevo modelo en MLflow y actualizar en caliente el contenedor Streamlit/servidor de inferencia.

---

## Propuesta de integración de AutoML en MLflow

> **Nota**: La integración de AutoML (PyCaret) en el pipeline MLflow no fue implementada en esta versión debido a la complejidad de su instalación y a los posibles conflictos de dependencias con TensorFlow en el mismo entorno virtual. A continuación se propone la estrategia técnica para realizarla.

### Estrategia recomendada

**Librería sugerida:** [PyCaret 3.x](https://pycaret.org/) — la misma utilizada en el notebook `AUTOML.ipynb`.

#### Pasos de integración

1. **Instalar PyCaret en un entorno separado (o con extras):**

   ```bash
   pip install pycaret[full]
   ```

   > Se recomienda un entorno virtual independiente para evitar conflictos con TensorFlow.

2. **Crear `src/training/train_automl.py`** con la siguiente lógica:

   ```python
   import mlflow
   from pycaret.classification import setup, compare_models, pull
   from src.mlops.mlflow_utils import setup_mlflow
   from src.utils.data_utils import load_ml_data

   def train_automl_pipeline(config_path="config/train_config.yaml"):
       from src.utils.config_utils import load_yaml_config
       config = load_yaml_config(config_path)
       setup_mlflow(config)

       X_train, X_val, y_train, y_val = load_ml_data()

       import pandas as pd
       train_df = pd.DataFrame(X_train)
       train_df['label'] = y_train

       with mlflow.start_run(run_name="automl_pycaret_v1"):
           # PyCaret setup con integración MLflow nativa
           clf = setup(data=train_df, target='label',
                       log_experiment=True,
                       experiment_name="PlantVillage_Disease_Classification",
                       session_id=42, verbose=False)
           best_model = compare_models(n_select=1)
           results = pull()
           mlflow.log_table(results.reset_index(), "automl_leaderboard.json")
   ```

3. **Añadir la etapa al orquestador `main.py`:**

   ```python
   elif stage == "train_automl":
       from src.training.train_automl import train_automl_pipeline
       train_automl_pipeline()
   ```

4. **Ejecutar:**
   ```bash
   python main.py --stage train_automl
   ```

#### Ventajas de esta integración

- PyCaret 3.x tiene **integración nativa con MLflow**: cada modelo evaluado se registra automáticamente como un run hijo.
- El `experiment_name` puede apuntar al mismo experimento `PlantVillage_Disease_Classification` para centralizar todos los runs.
- El mejor modelo devuelto por `compare_models()` puede serializarse con `joblib` y registrarse como artefacto adicional.

#### Consideraciones

- PyCaret requiere Python 3.8–3.11; verificar compatibilidad con la versión del entorno.
- La ejecución de AutoML sobre 61 000 imágenes (256 features HSV) puede tardar varias horas en CPU; se recomienda usar un subconjunto estratificado (~10 000 muestras) para el benchmark.
- El resultado del benchmark AutoML debe incluirse en la tabla comparativa de la **Fase 2** para cumplir con el requisito académico.
