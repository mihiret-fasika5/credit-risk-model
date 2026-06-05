import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Import your pipeline from Task 4
from data_processing import execute_full_processing_pipeline

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates comprehensive classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    }

def train_and_track_experiments(processed_data_path=None):
    # Set up MLflow Experiment
    mlflow.set_experiment("Credit_Risk_Proxy_Modeling")
    
    # 1. Load Data (Using Mock Data if no path provided for runtime execution)
    if processed_data_path and os.path.exists(processed_data_path):
        df = pd.read_csv(processed_data_path)
    else:
        print("No processed data file found. Generating inline dataset for training simulation...")
        # Simulating output from task-4 architecture
        np.random.seed(42)
        mock_rows = 200
        df = pd.DataFrame({
            'Amount': np.random.randn(mock_rows),
            'Value': np.random.randn(mock_rows),
            'TransactionHour': np.random.randint(0, 24, mock_rows),
            'ProductCategory_airtime': np.random.randint(0, 2, mock_rows),
            'ProductCategory_utility': np.random.randint(0, 2, mock_rows),
            'is_high_risk': np.random.choice([0, 1], size=mock_rows, p=[0.8, 0.2])
        })

    # Separate Features and engineered Target
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    best_overall_f1 = -1
    best_run_id = None
    best_model_name = "CreditRisk_Best_Model"

    # Define experiments to run
    models_to_run = {
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "grid": {
                "n_estimators": [50, 100],
                "max_depth": [5, 10]
            },
            "type": "sklearn"
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric="logloss"),
            "grid": {
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6]
            },
            "type": "xgboost"
        }
    }

    for model_name, config in models_to_run.items():
        with mlflow.start_run(run_name=model_name) as run:
            print(f"\n--- Tuning and Training: {model_name} ---")
            
            # Hyperparameter Tuning using Grid Search
            cv = GridSearchCV(
                estimator=config["model"],
                param_grid=config["grid"],
                scoring="f1",
                cv=3,
                n_jobs=-1
            )
            cv.fit(X_train, y_train)
            
            best_model = cv.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            
            # Performance Assessment
            metrics = evaluate_model(y_test, y_pred, y_prob)
            
            # Log Parameters & Metrics to MLflow
            mlflow.log_params(cv.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_param("algorithm", model_name)
            
            print(f"Best Params: {cv.best_params_}")
            print(f"Test F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Log Model Artifact based on Framework type
            if config["type"] == "sklearn":
                mlflow.sklearn.log_model(best_model, artifact_path="model")
            elif config["type"] == "xgboost":
                mlflow.xgboost.log_model(best_model, artifact_path="model")
            
            # Track the champion model across runs based on F1 Score
            if metrics["f1_score"] > best_overall_f1:
                best_overall_f1 = metrics["f1_score"]
                best_run_id = run.info.run_id

    # 3. Register the Champion Model in the MLflow Model Registry
    if best_run_id:
        print(f"\nRegistering champion model from run [{best_run_id}] with F1: {best_overall_f1:.4f}")
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, best_model_name)

if __name__ == "__main__":
    train_and_track_experiments()