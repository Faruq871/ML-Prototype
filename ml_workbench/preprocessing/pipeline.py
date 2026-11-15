from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, KBinsDiscretizer
from ml_workbench.exceptions import TrainingError

NON_NEGATIVE_ALGORITHMS = {28}
BERNOULLI_ALGORITHMS = {29}


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable columns and normalize pandas extension dtypes."""
    X = X.copy()
    X = X.dropna(axis=1, how="all")
    # Treat infinities as missing values before imputation.
    numeric_columns = X.select_dtypes(include=["number"]).columns
    if len(numeric_columns):
        X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
    # Normalize mixed categorical values so OneHotEncoder never sees mixed Python types.
    categorical_columns = [c for c in X.columns if c not in numeric_columns]
    for col in categorical_columns:
        X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    if X.shape[1] == 0:
        raise TrainingError("Tidak ada feature yang bisa digunakan setelah kolom kosong dihapus.")
    return X


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # compatibility for older scikit-learn releases
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, algorithm_id: int) -> ColumnTransformer:
    X = prepare_features(X)
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []

    if numeric:
        if algorithm_id in NON_NEGATIVE_ALGORITHMS:
            num_steps = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", MinMaxScaler(clip=True)),
            ]
        elif algorithm_id in BERNOULLI_ALGORITHMS:
            num_steps = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("discretizer", KBinsDiscretizer(n_bins=2, encode="onehot-dense", strategy="quantile")),
            ]
        else:
            num_steps = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        transformers.append(("numeric", Pipeline(num_steps), numeric))

    if categorical:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", _one_hot_encoder()),
        ]
        transformers.append(("categorical", Pipeline(cat_steps), categorical))

    if not transformers:
        raise TrainingError("Tidak ditemukan kolom numerik atau kategorikal yang dapat diproses.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
