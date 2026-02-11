# from src.data_transform import transform_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score , f1_score,recall_score,precision_score
from xgboost import XGBClassifier
import joblib
import os
from dotenv import load_dotenv
import mlflow
import mlflow.xgboost
import optuna
import time
import sys
from src.logging import logging
from src.exception import CustomException


class TrainModel:
    def __init__(self, transform):
        self.transform = transform  
    
    def train_model(self):
        logging.info("model_training started")
        try:
            load_dotenv()
            mlflow.set_experiment("Telco Churn - XGBoost")
            df = self.transform
            X = df.drop(columns=['Churn'])
            y = df['Churn']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            THRESHOLD = 0.25
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                    "random_state": 42,
                    "n_jobs": -1,
                    "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
                    "eval_metric": "logloss"
                }
                model = XGBClassifier(**params)
                model.fit(X_train,y_train)
                proba = model.predict_proba(X_test)[:, 1]
                y_pred = (proba >= THRESHOLD).astype(int)  # Keep your tuned threshold
                return recall_score(y_test, y_pred, pos_label=1) 


            study = optuna.create_study(direction="maximize")
            study.optimize(objective,n_trials=30)

            print("Best Params:", study.best_params)
            print("Best Recall:", study.best_value)
            
            # mlflow.set_tracking_uri("http://localhost:5000")
            
            with mlflow.start_run():
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

                # Best params from Optuna
                best_params = study.best_params
                best_params.update({
                    "random_state": 42,
                    "n_jobs": -1,
                    "scale_pos_weight": scale_pos_weight,
                    "eval_metric": "logloss"
                })
                mlflow.log_params(best_params)
                mlflow.set_tag("model_type", "xgboost")
                mlflow.set_tag("threshold", THRESHOLD)

                # Training timer

                
                start_train = time.time()
                xgb = XGBClassifier(**best_params)
                xgb.fit(X_train, y_train)
                train_time = time.time() - start_train
                mlflow.log_metric("train_time", train_time)

                start_pred = time.time()
                proba = xgb.predict_proba(X_test)[:, 1]
                y_pred = (proba >= THRESHOLD).astype(int)
                pred_time = time.time() - start_pred
                mlflow.log_metric("pred_time", pred_time)
            # Metrics
                precision = precision_score(y_test, y_pred, pos_label=1)
                recall = recall_score(y_test, y_pred, pos_label=1)
                f1 = f1_score(y_test, y_pred, pos_label=1)
                auc = roc_auc_score(y_test, proba)

                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc_auc", auc)

                # Save model
                from mlflow.models import infer_signature
                signature = infer_signature(X_test, xgb.predict(X_test))
                
                mlflow.xgboost.log_model(
                    xgb, 
                    artifact_path="model",
                    signature=signature,
                    input_example=X_test.iloc[:5] # Good for documentation
                )

                print(classification_report(y_test, y_pred, digits=3))
            logging.info("model_training + mlflow complete")    
        except Exception as e:
            logging.info("error occured in model_train")
            raise CustomException(e,sys)
            