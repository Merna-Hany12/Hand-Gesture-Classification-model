import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
MLRUNS_PATH = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLRUNS_PATH, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{MLRUNS_PATH}")

def setup_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def start_model_run(run_name, model, params, metrics, artifacts=None, X_train=None, y_train=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        input_example = X_train[:5] if X_train is not None else None
        mlflow.sklearn.log_model(sk_model=model, name="model", input_example=input_example)

        if artifacts:
            for artifact in artifacts:
                if os.path.exists(artifact):
                    folder = "plots" if artifact.endswith(".png") else "artifacts"
                    mlflow.log_artifact(artifact, artifact_path=folder)
        if X_train is not None and y_train is not None:
            dataset = mlflow.data.from_pandas(pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name="label")], axis=1), name="gesture_train_set")
            mlflow.log_input(dataset, context="training")

def register_best_model(experiment_name, metric_name="val_accuracy", registered_model_name="HandGestureClassifier2"):
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"]
    )

    if runs.empty:
        raise ValueError("No runs found in this experiment.")

    best_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=result.version,
            stage="Production"
        )
    except Exception as e:
        print("Warning: could not transition model stage due to serialization:", e)

    return best_run_id