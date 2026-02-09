# from src.data_transform import transform_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
mlflow.sklearn.autolog()
def train_model(transform):
    mlflow.set_experiment("Churn Prediction Experiment")
    df = transform
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    cat_features = [
        "gender", "Partner", "Dependents", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "Contract", "PaperlessBilling", "PaymentMethod", "PhoneService"
    ]
    num_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, max_depth=12),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, max_depth=12),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_estimators=300)
    }
    with mlflow.start_run(run_name="Model_Comparison_Session"):
        results = {}
        for name, clf in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                pipe = ImbPipeline([
                    ("prep", preprocessor),
                    ("smote", SMOTE(random_state=42)),
                    ("model", clf)
                ])
                mlflow.log_param("model_type", name)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_prob = pipe.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_prob)

                metrics = {
                    "Accuracy": np.mean(y_pred == y_test),
                    "ROC-AUC": roc,
                    "Precision": classification_report(y_test, y_pred, output_dict=True)["1"]["precision"],
                    "Recall": classification_report(y_test, y_pred, output_dict=True)["1"]["recall"],
                    "F1": classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"],
                }
                mlflow.log_metrics(metrics)
                
                results[name] = metrics
                
        results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
        print("\n=== Model Comparison ===")
        print(results_df)

        best_model_name = results_df.index[0]
        print(f"\nBest model selected: {best_model_name}")
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_roc_auc", results_df.loc[best_model_name, "ROC-AUC"])
        mlflow.sklearn.autolog()
        with mlflow.start_run(run_name="Final_Best_Model_Export", nested=True):
            final_pipe = ImbPipeline([
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", models[best_model_name])
            ])
            final_pipe.fit(X_train, y_train)

            # 7. Log the model with a "Signature" (Input/Output schema)
            # This is crucial for FastAPI/Deployment later
            signature = infer_signature(X_test, final_pipe.predict(X_test))
            
            mlflow.sklearn.log_model(
                sk_model=final_pipe,
                artifact_path="model",
                signature=signature,
                registered_model_name="Churn_Production_Model" # Registers it immediately!
            )

            # Optional: Save locally as you did before
            joblib.dump(final_pipe, "outputs/churn_prediction_model.pkl")
            joblib.dump(X.columns.tolist(), "outputs/feature_order.pkl")
            
            print(f"Workflow complete. Best Model: {best_model_name} is now in MLflow Registry.")