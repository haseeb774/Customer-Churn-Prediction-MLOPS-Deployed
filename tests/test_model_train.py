import os
import numpy as np
import pandas as pd
import joblib
from src.model_train import TrainModel


class DummyClassifier:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        self._n_features = X.shape[1]
        return self

    def predict_proba(self, X):
        probs = np.vstack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5]).T
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DummyStudy:
    def __init__(self):
        self.best_params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
        }
        self.best_value = 1.0

    def optimize(self, objective, n_trials):
        # no-op (skip heavy optimization)
        return


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_train_model_creates_model_file(monkeypatch):
    # small synthetic dataset
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "feat1": rng.randn(100),
        "feat2": rng.randn(100),
        "Churn": rng.choice([0, 1], size=100),
    })

    # monkeypatch heavy/IO dependencies to keep test fast and deterministic
    monkeypatch.setattr("src.model_train.optuna.create_study", lambda direction: DummyStudy())
    monkeypatch.setattr("src.model_train.XGBClassifier", DummyClassifier)

    # mlflow no-ops
    monkeypatch.setattr("src.model_train.mlflow.set_experiment", lambda *a, **k: None)
    monkeypatch.setattr("src.model_train.mlflow.start_run", lambda *a, **k: DummyContext(), raising=False)
    monkeypatch.setattr("src.model_train.mlflow.log_params", lambda *a, **k: None)
    monkeypatch.setattr("src.model_train.mlflow.set_tag", lambda *a, **k: None)
    monkeypatch.setattr("src.model_train.mlflow.log_metric", lambda *a, **k: None)
    monkeypatch.setattr("src.model_train.mlflow.xgboost.log_model", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("src.model_train.mlflow.models.infer_signature", lambda X, y: None, raising=False)

    # run training (should be fast because we've stubbed heavy parts)
    tm = TrainModel(transform=df)
    tm.train_model()

    assert os.path.exists("outputs/model.pkl")

    # validate that the saved file is loadable
    m = joblib.load("outputs/model.pkl")
    assert hasattr(m, "predict_proba") or hasattr(m, "predict")

    # cleanup
    try:
        os.remove("outputs/model.pkl")
    except Exception:
        pass
