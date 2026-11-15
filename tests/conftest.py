import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

@pytest.fixture
def classification_df():
    X, y = make_classification(n_samples=180, n_features=6, n_informative=4,
                               n_redundant=1, random_state=7)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["segment"] = np.where(np.arange(len(df)) % 3 == 0, "A", "B")
    df["target"] = y
    df.loc[df.index[:8], "x0"] = np.nan
    return df

@pytest.fixture
def regression_df():
    X, y = make_regression(n_samples=180, n_features=6, noise=10, random_state=7)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df
