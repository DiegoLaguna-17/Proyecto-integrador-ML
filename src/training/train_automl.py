import mlflow
from pycaret.classification import ClassificationExperiment
from src.mlops.mlflow_utils import setup_mlflow
from src.utils.data_utils import load_ml_data

def train_automl_pipeline(config_path="config/train_config.yaml", quick=False):
    from src.utils.config_utils import load_yaml_config
    config = load_yaml_config(config_path)
    setup_mlflow(config)

    X_train, X_val, y_train, y_val = load_ml_data()

    import pandas as pd
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train

    if quick:
        print("[INFO] Quick mode activo para AutoML. Reduciendo subconjunto de datos...")
        train_df = train_df.sample(n=min(500, len(train_df)), random_state=42)

    with mlflow.start_run(run_name="automl_pycaret_v1"):
        exp = ClassificationExperiment(
            target='label',
            session_id=42, 
            verbose=False
        ).fit(train_df)
                  
        compare_result = exp.compare_models(n_select=1)
        best_model = compare_result.best
        results_df = compare_result.leaderboard
        
        mlflow.log_table(results_df.reset_index(), "automl_leaderboard.json")