import os
import json
import joblib
import mlflow
import numpy as np
from scipy.stats import mode
from config import settings
from src.utils.config_utils import load_yaml_config
from src.utils.data_utils import load_ml_data, get_dataset_version
from src.mlops.mlflow_utils import setup_mlflow, log_run_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _save_model(model, filename):
    """Persists a model to /models/<filename> and returns the path."""
    os.makedirs(os.path.join(settings.BASE_DIR, "models"), exist_ok=True)
    path = os.path.join(settings.BASE_DIR, "models", filename)
    joblib.dump(model, path)
    print(f"[INFO] Modelo guardado localmente en: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 1. Random Forest Baseline
# ──────────────────────────────────────────────────────────────────────────────

def _run_random_forest(X_train, X_val, y_train, y_val, class_names, dataset_version, config):
    """Trains and logs the Random Forest baseline."""
    rf_params = config['random_forest']
    run_name = config['mlflow'].get('ml_run_name', 'random_forest_v1')

    with mlflow.start_run(run_name=run_name):
        print(f"\n[INFO] === Random Forest === run: {run_name}")
        model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            random_state=rf_params['random_state'],
            n_jobs=rf_params['n_jobs']
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        log_run_metrics(
            y_true=y_val,
            y_pred=y_pred,
            class_names=class_names,
            dataset_version=dataset_version,
            parameters=rf_params,
            config_dict=config,
            stage_name="train_ml_rf",
            model=model,
            model_type="sklearn"
        )
        _save_model(model, "baseline_rf.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# 2. KNN
# ──────────────────────────────────────────────────────────────────────────────

def _run_knn(X_train, X_val, y_train, y_val, class_names, dataset_version, config):
    """Trains and logs the KNN classifier (sklearn CPU, n_neighbors=5)."""
    knn_params = config.get('knn', {'n_neighbors': 5})
    run_name = config['mlflow'].get('knn_run_name', 'knn_v1')

    with mlflow.start_run(run_name=run_name):
        print(f"\n[INFO] === KNN === run: {run_name}")
        model = KNeighborsClassifier(n_neighbors=knn_params.get('n_neighbors', 5))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        log_run_metrics(
            y_true=y_val,
            y_pred=y_pred,
            class_names=class_names,
            dataset_version=dataset_version,
            parameters=knn_params,
            config_dict=config,
            stage_name="train_ml_knn",
            model=model,
            model_type="sklearn"
        )
        _save_model(model, "knn_model.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# 3. SVC  (Pipeline: StandardScaler → PCA(0.95) → SVC)
# ──────────────────────────────────────────────────────────────────────────────

def _run_svc(X_train, X_val, y_train, y_val, class_names, dataset_version, config):
    """Trains and logs the SVC pipeline (faithful to the notebook)."""
    svc_params = config.get('svc', {'kernel': 'rbf', 'probability': True, 'random_state': 42})
    run_name = config['mlflow'].get('svc_run_name', 'svc_v1')

    with mlflow.start_run(run_name=run_name):
        print(f"\n[INFO] === SVC === run: {run_name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('svc', SVC(
                kernel=svc_params.get('kernel', 'rbf'),
                probability=svc_params.get('probability', True),
                random_state=svc_params.get('random_state', 42)
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        log_params = {
            "kernel": svc_params.get('kernel', 'rbf'),
            "probability": svc_params.get('probability', True),
            "random_state": svc_params.get('random_state', 42),
            "pca_n_components": 0.95,
            "scaler": "StandardScaler"
        }
        log_run_metrics(
            y_true=y_val,
            y_pred=y_pred,
            class_names=class_names,
            dataset_version=dataset_version,
            parameters=log_params,
            config_dict=config,
            stage_name="train_ml_svc",
            model=pipeline,
            model_type="sklearn"
        )
        _save_model(pipeline, "svc_model.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Clustering — KMeans (unsupervised)
# ──────────────────────────────────────────────────────────────────────────────

def _run_clustering(X_train, X_val, y_train, y_val, class_names, dataset_version, config):
    """
    Trains KMeans (n_clusters=10, as in the notebook) and logs to MLflow.
    Because clustering is unsupervised, we:
      - Report ARI and Silhouette on the validation split.
      - Build a cluster→class mapping (majority vote) to also report accuracy.
    """
    clustering_params = config.get('clustering', {'n_clusters': 10, 'random_state': 42})
    n_clusters = clustering_params.get('n_clusters', 10)
    random_state = clustering_params.get('random_state', 42)
    run_name = config['mlflow'].get('clustering_run_name', 'clustering_kmeans_v1')

    with mlflow.start_run(run_name=run_name):
        print(f"\n[INFO] === KMeans Clustering === run: {run_name}")

        # Encode string labels to integers for ARI/majority-vote
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_val_enc   = le.transform(y_val)

        # Combine train + val features for fitting (unsupervised — no leakage concern)
        X_all = np.vstack([X_train, X_val])
        y_all_enc = np.concatenate([y_train_enc, y_val_enc])

        # Fit KMeans (as in notebook)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(X_train)          # fit on train only
        train_labels = kmeans.labels_
        val_labels   = kmeans.predict(X_val)

        # ── Cluster → class mapping (majority vote on train set) ──────────────
        cluster_to_class = {}
        for c in range(n_clusters):
            mask = train_labels == c
            if mask.sum() == 0:
                cluster_to_class[c] = 0
                continue
            votes = y_train_enc[mask]
            m = mode(votes, keepdims=True)
            cluster_to_class[c] = int(m.mode[0])

        # Map val cluster labels → predicted class indices
        val_pred_enc = np.array([cluster_to_class.get(lbl, 0) for lbl in val_labels])
        val_pred_labels = le.inverse_transform(val_pred_enc)

        # ── Unsupervised metrics ───────────────────────────────────────────────
        ari = adjusted_rand_score(y_val_enc, val_labels)
        # Silhouette on a sample (can be slow on full data)
        sample_size = min(3000, len(X_val))
        idx = np.random.RandomState(42).choice(len(X_val), sample_size, replace=False)
        sil = silhouette_score(X_val[idx], val_labels[idx])
        mapped_acc = accuracy_score(y_val, val_pred_labels)

        print(f"[INFO] ARI={ari:.4f} | Silhouette={sil:.4f} | Mapped Accuracy={mapped_acc:.4f}")

        # ── Log to MLflow ──────────────────────────────────────────────────────
        log_params = {
            "n_clusters": n_clusters,
            "random_state": random_state,
            "algorithm": "KMeans",
            "note": "Unsupervised — accuracy via majority-vote cluster mapping"
        }
        mlflow.log_params(log_params)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("model_type", "sklearn")
        mlflow.log_param("stage", "train_ml_clustering")

        mlflow.log_metric("adjusted_rand_index", ari)
        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("mapped_accuracy", mapped_acc)

        # Save cluster-to-class mapping as JSON artifact
        os.makedirs("reports", exist_ok=True)
        mapping_path = os.path.join("reports", "cluster_class_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in cluster_to_class.items()}, f, indent=4)
        mlflow.log_artifact(mapping_path, artifact_path="evaluation")

        # Save config snapshot
        config_path = os.path.join("reports", "config_train_ml_clustering.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        mlflow.log_artifact(config_path, artifact_path="config")

        # Log the KMeans model itself
        mlflow.sklearn.log_model(kmeans, artifact_path="model")

        _save_model(kmeans, "clustering_kmeans.pkl")
        print(f"[MLflow] Métricas de clustering logueadas.")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def train_ml_baseline(config_path="config/train_config.yaml"):
    """
    Trains all ML models (Random Forest, KNN, SVC, KMeans Clustering)
    and logs each to a separate MLflow run inside PlantVillage_Disease_Classification.
    """
    config = load_yaml_config(config_path)

    # Configure MLflow
    setup_mlflow(config)

    # Dataset version
    zip_path = os.path.join(settings.DATA_DIR, "raw", "Plant_leaf_diseases_dataset_with_augmentation.zip")
    dataset_version = get_dataset_version(zip_path)

    # Load data
    try:
        X_train, X_val, y_train, y_val = load_ml_data()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"[INFO] Datos de ML cargados. Train: {X_train.shape}, Val: {X_val.shape}")

    # Sorted unique class names (string labels from CSV)
    class_names = sorted(list(np.unique(y_train)))

    # ── Run all models ────────────────────────────────────────────────────────
    _run_random_forest(X_train, X_val, y_train, y_val, class_names, dataset_version, config)
    _run_knn          (X_train, X_val, y_train, y_val, class_names, dataset_version, config)
    _run_svc          (X_train, X_val, y_train, y_val, class_names, dataset_version, config)
    _run_clustering   (X_train, X_val, y_train, y_val, class_names, dataset_version, config)

    print("\n[INFO] Todos los modelos ML han sido entrenados y registrados en MLflow.")


if __name__ == "__main__":
    train_ml_baseline()
