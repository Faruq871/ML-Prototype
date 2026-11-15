from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ml_workbench.algorithms.catalog import get_algorithm
from ml_workbench.algorithms.factory import build_estimator
from ml_workbench.preprocessing.pipeline import build_preprocessor, prepare_features
from ml_workbench.evaluation.metrics import evaluate_supervised, evaluate_unsupervised
from ml_workbench.exceptions import TrainingError
from ml_workbench.schemas import TrainingResult

SUPERVISED_TASKS = {"classification", "regression"}
UNSUPERVISED_TASKS = {"clustering", "dimensionality_reduction", "anomaly_detection"}


def _clean_target_data(data: pd.DataFrame, target: str, drop_columns: list[str] | None):
    if target not in data.columns:
        raise TrainingError(f"Target '{target}' tidak ditemukan.")
    working = data.loc[data[target].notna()].copy()
    y = working[target]
    excluded = set(drop_columns or []) | {target}
    X = working.drop(columns=[c for c in excluded if c in working.columns])
    X = prepare_features(X)
    if len(X) < 5:
        raise TrainingError("Minimal 5 baris dengan target yang tidak kosong diperlukan untuk training.")
    return X, y


def _split_supervised(X, y, task, test_size, random_state):
    stratify = None
    if task == "classification":
        counts = y.value_counts(dropna=False)
        if len(counts) < 2:
            raise TrainingError("Classification membutuhkan minimal dua kelas pada target.")
        if counts.min() < 2:
            raise TrainingError("Setiap kelas memerlukan minimal dua observasi agar split dan evaluasi tidak menyesatkan.")
        n_test = int(np.ceil(len(y) * test_size))
        n_train = len(y) - n_test
        if n_test < len(counts) or n_train < len(counts):
            raise TrainingError("Jumlah baris terlalu sedikit untuk stratified train/test split. Tambahkan data atau ubah test size.")
        stratify = y
    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise TrainingError("Target regression harus berupa numerik.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def train_model(data: pd.DataFrame, algorithm_id: int, *, target: str | None = None,
                test_size: float = 0.2, random_state: int = 42, n_clusters: int = 3, n_components: int = 2,
                drop_columns: list[str] | None = None, model_params: dict | None = None) -> TrainingResult:
    if not 0.1 <= float(test_size) <= 0.5:
        raise TrainingError("test_size harus berada antara 0.10 dan 0.50.")
    info = get_algorithm(algorithm_id)
    task = info.task
    if task in SUPERVISED_TASKS:
        if not target:
            raise TrainingError("Target variable wajib dipilih untuk classification/regression.")
        X, y = _clean_target_data(data, target, drop_columns)
        X_train, X_test, y_train, y_test = _split_supervised(X, y, task, test_size, random_state)
        counts = y_train.value_counts() if task == "classification" else None
        stacking_cv = min(5, int(counts.min())) if counts is not None else 5
        if algorithm_id == 26 and stacking_cv < 2:
            raise TrainingError("Stacking membutuhkan sedikitnya dua observasi per kelas dalam training fold.")
        preprocessor = build_preprocessor(X_train, algorithm_id)
        estimator = build_estimator(algorithm_id, n_train=len(X_train), n_features=X_train.shape[1],
                                    random_state=random_state, stacking_cv=max(2, stacking_cv))
        if model_params:
            estimator.set_params(**model_params)
        pipeline = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
        except Exception as exc:
            raise TrainingError(f"Training {info.name} gagal: {exc}") from exc
        metrics = evaluate_supervised(task, pipeline, X_test, y_test, predictions)
        output = pd.DataFrame({"row_index": X_test.index.astype(str), "actual": y_test.astype(str).values,
                               "predicted": pd.Series(predictions).astype(str).values})
        return TrainingResult(task, algorithm_id, info.name, pipeline, preprocessor, metrics, predictions,
                              output, {"train_rows": len(X_train), "test_rows": len(X_test),
                                       "target": target, "feature_columns": list(X.columns),
                                       "random_state": random_state, "test_size": test_size})

    if task not in UNSUPERVISED_TASKS:
        raise TrainingError(f"Task {task} belum didukung oleh training engine.")
    excluded = set(drop_columns or [])
    if target and target in data.columns:
        excluded.add(target)
    X = prepare_features(data.drop(columns=[c for c in excluded if c in data.columns]))
    if len(X) < 3:
        raise TrainingError("Unsupervised learning membutuhkan minimal tiga observasi.")
    # fit preprocessing on all rows: no labels or test set are used in this unsupervised workflow.
    preprocessor = build_preprocessor(X, algorithm_id)
    try:
        transformed = preprocessor.fit_transform(X)
        estimator = build_estimator(algorithm_id, n_train=len(X), n_features=transformed.shape[1],
                                    n_clusters=n_clusters, n_components=n_components, random_state=random_state)
        if model_params:
            estimator.set_params(**model_params)
        if task == "dimensionality_reduction":
            reduced = estimator.fit_transform(transformed)
            output = pd.DataFrame(reduced, index=X.index,
                                  columns=[f"component_{i+1}" for i in range(reduced.shape[1])])
            predictions = None
            metrics = evaluate_unsupervised(task, estimator, transformed)
        else:
            if task == "clustering" and n_clusters >= len(X) and algorithm_id != 34:
                raise TrainingError("Jumlah cluster harus lebih kecil daripada jumlah observasi.")
            labels = estimator.fit_predict(transformed)
            output = pd.DataFrame({"row_index": X.index.astype(str), "cluster_or_label": labels})
            predictions = labels
            metrics = evaluate_unsupervised(task, estimator, transformed, labels)
    except TrainingError:
        raise
    except Exception as exc:
        raise TrainingError(f"Eksekusi {info.name} gagal: {exc}") from exc
    bundle = {"preprocessor": preprocessor, "estimator": estimator, "task": task,
              "algorithm_id": algorithm_id, "feature_columns": list(X.columns)}
    return TrainingResult(task, algorithm_id, info.name, bundle, preprocessor, metrics, predictions,
                          output, {"rows": len(X), "feature_columns": list(X.columns), "n_clusters": n_clusters, "n_components": n_components})
