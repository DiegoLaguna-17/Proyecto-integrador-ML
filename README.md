# Clasificación de Fitopatologías en PlantVillage - Fase 1 (Machine Learning Clásico)

Este repositorio contiene la Fase 1 del proyecto de detección automatizada de enfermedades en plantas utilizando el dataset **PlantVillage**. En esta etapa, el enfoque se centra en el uso de **Machine Learning Clásico** y descriptores de color para establecer un baseline predictivo sólido.

## 🚀 Descripción del Proyecto

El objetivo es clasificar hojas de plantas en 38 categorías fisiológicas (sanas y enfermas) basándose exclusivamente en su firma cromática. Se implementó un pipeline completo que abarca desde la extracción de características (*Feature Engineering*) hasta la optimización mediante **AutoML**.

### Estructura del Repositorio

* `src/`: Scripts de procesamiento.
    * `feature_extraction.py`: Script en Python encargado de transformar las imágenes en histogramas de color de 256 variables.
* `notebooks/`: Experimentos detallados y validación.
    * `Clustering_Visual.ipynb`: Análisis no supervisado con K-Means y proyección PCA.
    * `SVC_model.ipynb`: Implementación del baseline manual (Pipeline: Scaler + PCA + SVC).
    * `KNN_model.ipynb` & `RF_model.ipynb`: Modelos clásicos comparativos (incluye aceleración GPU con RAPIDS).
    * `AUTOML.ipynb`: Benchmarking exhaustivo utilizando la librería PyCaret 3.4.


## 🛠️ Pipeline Técnico

### 1. Extracción de Características
Se omitió el uso de píxeles en bruto para favorecer el uso de **Histogramas de Color**. Cada imagen se convirtió en un vector de 256 dimensiones, capturando la distribución de intensidades cromáticas pero eliminando la información espacial/topológica.

### 2. Preprocesamiento y Componente No Supervisado
* **Normalización**: Uso de `StandardScaler` para balancear las magnitudes de los histogramas.
* **Reducción de Dimensionalidad (PCA)**: Se integró PCA para retener el **95% de la varianza**, reduciendo el espacio de 256 a **138 componentes ortogonales**, eliminando la colinealidad.
* **Clustering (K-Means)**: Validación no supervisada para identificar agrupamientos naturales de enfermedades basados puramente en color.

### 3. Modelado Predictivo
Se evaluaron múltiples arquitecturas para determinar el techo de rendimiento del color:
* **SVC (Support Vector Classifier)**: Baseline manual con kernel RBF.
* **Extra Trees (AutoML Winner)**: Modelo de ensamble que demostró la mejor robustez frente al ruido del PCA.
* **KNN & Random Forest**: Modelos comparativos para análisis de varianza.

## 📊 Resultados Obtenidos FASE 2

| Estrategia | Algoritmo | Accuracy (Test) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Manual (Baseline)** | SVC (Pipeline PCA) | **87.49%** | 87.10% |
| **AutoML Benchmark** | **Extra Trees (ET)** | **87.02%** | 86.73% |
| **Manual** | KNN (RAPIDS GPU) | 86.00% | 86.00% |
| **Manual** | Random Forest | 78.00% | 77.00% |
| **No Supervisado** | K-Means (Clustering) | 80.00% | 80.00% |

> **Conclusión técnica:** Se identificó un "techo predictivo" del **87.5%**. El error residual (~12.5%) se atribuye a la ambigüedad cromática entre patologías necróticas (sarnas y tizones), lo que justifica la transición hacia Deep Learning (CNN) en la Fase 2.

## 🔧 Requisitos e Instalación

Para replicar este entorno, se recomienda el uso de Python 3.10+ y la instalación de las dependencias:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pycaret
```
NOTA:
El dataset no se encuentra en el repositorio debido a limites de tamaño de github, el archivo csv de features sobrepasa los 100MB
