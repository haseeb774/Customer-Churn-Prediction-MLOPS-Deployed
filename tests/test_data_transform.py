import pandas as pd
from src.data_transform import Datatransform


def _make_input_df():
    return pd.DataFrame({
        "customerID": ["id1", "id2"],
        "gender": ["Female", "Male"],
        "TotalCharges": ["100.0", "200.5"],
        "Churn": ["Yes", "No"],
        "CLV_proxy": [1, 2],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No phone service", "Yes"],
        "InternetService": ["DSL", "No"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "OnlineBackup": ["Yes", "No internet service"],
        "OnlineSecurity": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "StreamingTV": ["No internet service", "No"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
    })


def test_transform_data_mappings_and_onehot():
    df = _make_input_df()
    dt = Datatransform(df.copy())
    out = dt.transform_data()

    # gender mapping
    assert out["gender"].tolist() == [0, 1]

    # TotalCharges numeric
    assert str(out["TotalCharges"].dtype).startswith("float")

    # Churn mapping
    assert set(out["Churn"].unique()) <= {0, 1}

    # customerID and CLV_proxy should be removed
    assert "customerID" not in out.columns
    assert "CLV_proxy" not in out.columns

    # payment method should be one-hot encoded
    assert any(col.startswith("PaymentMethod_") for col in out.columns)

    # services -> 0/1
    services = ["OnlineBackup", "OnlineSecurity", "StreamingMovies", "StreamingTV", "DeviceProtection", "TechSupport"]
    for col in services:
        assert set(out[col].unique()) <= {0, 1}
