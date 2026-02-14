import os
import pandas as pd
import sys
from src.exception import CustomException
from src.data_ingest import DataIngest


def test_data_import_writes_and_returns_dataframe(tmp_path):
    # prepare a small CSV
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    di = DataIngest(str(csv_path))
    result = di.data_import()

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df)

    # ensure side-effect file was created
    assert os.path.exists("data/raw/churn.csv")

    # cleanup
    try:
        os.remove("data/raw/churn.csv")
    except Exception as e:
        raise CustomException(e,sys)