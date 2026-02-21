import mlflow
import mlflow.sklearn
import os
import mlflow.data
import pandas as pd
import numpy as np
# --------- Set Tracking Directory Outside Notebooks ---------

# Project root (one level above src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MLRUNS_PATH = os.path.join(BASE_DIR, "mlruns")

# Create mlruns folder if it doesn't exist
os.makedirs(MLRUNS_PATH, exist_ok=True)

# IMPORTANT: Set tracking URI BEFORE using MLflow
mlflow.set_tracking_uri(f"file:///{MLRUNS_PATH}")


# --------- Experiment Setup ---------

def setup_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)


# --------- Start Model Run ---------

def start_model_run(run_name,model,params,metrics,artifacts=None,input_example=None,X_train=None,y_train=None,X_test=None,y_test=None):
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example
        )

        if artifacts:
            for artifact in artifacts:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact)
                else:
                    print(f"WARNING: Artifact not found: {artifact}")

        if X_train is not None and y_train is not None:
            y_train_series = pd.Series(y_train, name="label") if isinstance(y_train, np.ndarray) else y_train.rename("label")
            train_df = pd.concat([pd.DataFrame(X_train), y_train_series], axis=1)

            train_dataset = mlflow.data.from_pandas(
                train_df,
                source="hand_landmarks_data.csv",
                name="train_data"
            )
            mlflow.log_input(train_dataset, context="training")

def register_model(model_name):
    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
    )