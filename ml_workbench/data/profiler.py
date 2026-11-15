from __future__ import annotations
import pandas as pd
from ml_workbench.schemas import DatasetProfile


def profile_dataframe(data: pd.DataFrame, target: str | None = None) -> DatasetProfile:
    df = data
    numeric = df.select_dtypes(include=["number"]).columns.astype(str).tolist()
    categorical = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.astype(str).tolist()
    datetimes = df.select_dtypes(include=["datetime", "datetimetz"]).columns.astype(str).tolist()
    missing_cells = int(df.isna().sum().sum())
    total_cells = max(int(df.shape[0] * df.shape[1]), 1)
    high_cardinality = [str(c) for c in df.columns if df[c].nunique(dropna=True) > min(100, max(30, int(len(df) * 0.5)))]
    constant = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    class_count = None
    minority_ratio = None
    if target is not None and target in df.columns:
        counts = df[target].dropna().value_counts()
        class_count = int(counts.size)
        if counts.sum() > 0:
            minority_ratio = float(counts.min() / counts.sum())
    return DatasetProfile(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        numeric_columns=numeric,
        categorical_columns=categorical,
        datetime_columns=datetimes,
        missing_cells=missing_cells,
        missing_ratio=float(missing_cells / total_cells),
        duplicate_rows=int(df.duplicated().sum()),
        memory_mb=float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        high_cardinality_columns=high_cardinality,
        constant_columns=constant,
        class_count=class_count,
        minority_class_ratio=minority_ratio,
    )


def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "column": data.columns.astype(str),
        "missing_count": data.isna().sum().values,
        "missing_percent": (data.isna().mean().values * 100).round(2),
        "dtype": data.dtypes.astype(str).values,
        "unique_values": data.nunique(dropna=True).values,
    })
    return summary.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)
