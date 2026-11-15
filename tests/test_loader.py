from io import BytesIO
import pandas as pd
import pytest
from ml_workbench.data.loader import load_dataframe, validate_dataframe
from ml_workbench.exceptions import DatasetError

def test_csv_loader():
    data = b"x,y\n1,2\n3,4\n"
    df = load_dataframe(data, "sample.csv")
    assert df.shape == (2, 2)

def test_xlsx_loader():
    buffer = BytesIO()
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_excel(buffer, index=False, engine="openpyxl")
    df = load_dataframe(buffer.getvalue(), "sample.xlsx")
    assert df.shape == (2, 2)

def test_loader_rejects_unsupported_extension():
    with pytest.raises(DatasetError):
        load_dataframe(b"x", "sample.ipynb")

def test_validation_rejects_duplicate_after_trim():
    with pytest.raises(DatasetError):
        validate_dataframe(pd.DataFrame([[1, 2]], columns=["x", " x "]))
