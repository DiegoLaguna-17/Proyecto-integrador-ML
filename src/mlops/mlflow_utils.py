import mlflow
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def setup_mlflow(config):
    db_path = os.path.abspath("mlflow_local.db")
    tracking_uri = f"sqlite:///{db_path}"
    
    os.makedirs("mlruns", exist_ok=True)
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config['mlflow']['experiment_name']
    mlflow.set_experiment(experiment_name)

def log_run_metrics(y_true, y_pred, class_names, dataset_version, parameters, config_dict, stage_name, model, model_type="sklearn"):
    accuracy = accuracy_score(y_true, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    mlflow.log_params(parameters)
    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("stage", stage_name)
    
    os.makedirs("reports", exist_ok=True)
    config_path = os.path.join("reports", f"config_{stage_name}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    mlflow.log_artifact(config_path, artifact_path="config")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision_w)
    mlflow.log_metric("recall_weighted", recall_w)
    mlflow.log_metric("f1_weighted", f1_w)
    mlflow.log_metric("precision_macro", precision_m)
    mlflow.log_metric("recall_macro", recall_m)
    mlflow.log_metric("f1_macro", f1_m)

    first_elem = y_true[0] if len(y_true) > 0 else 0
    if isinstance(first_elem, (int, np.integer)):
        labels = list(range(len(class_names)))
    else:
        labels = class_names

    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
    report_path = os.path.join("reports", f"classification_report_{stage_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path="evaluation")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.title(f'Matriz de Confusión - {stage_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join("reports", f"confusion_matrix_{stage_name}.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="evaluation")

    if model_type == "sklearn":
        mlflow.sklearn.log_model(model, artifact_path="model")
    elif model_type == "keras":
        mlflow.tensorflow.log_model(model, artifact_path="model")
        
    print(f"[MLflow] Métricas y artefactos logueados para el run: {mlflow.active_run().info.run_name}")
    print(f"Accuracy: {accuracy:.4f} | F1 weighted: {f1_w:.4f}")
