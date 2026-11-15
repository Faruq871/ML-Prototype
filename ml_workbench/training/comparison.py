from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from ml_workbench.algorithms.catalog import get_algorithm
from ml_workbench.algorithms.factory import build_estimator
from ml_workbench.preprocessing.pipeline import build_preprocessor, prepare_features
from ml_workbench.exceptions import TrainingError


def compare_models(data: pd.DataFrame, algorithm_ids: list[int], *, target: str,
                   test_size: float = 0.2, cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Compare supervised candidates using CV on the training partition only."""
    if not algorithm_ids:
        raise TrainingError("Pilih minimal satu algoritma untuk dibandingkan.")
    infos = [get_algorithm(aid) for aid in algorithm_ids]
    tasks = {i.task for i in infos}
    if len(tasks) != 1 or not tasks.issubset({"classification", "regression"}):
        raise TrainingError("Model comparison hanya menerima algoritma classification atau regression dalam satu task.")
    task = next(iter(tasks))
    if target not in data.columns:
        raise TrainingError("Target variable tidak ditemukan.")
    valid = data[target].notna()
    X = prepare_features(data.loc[valid].drop(columns=[target]))
    y = data.loc[valid, target]
    if len(X) < 8:
        raise TrainingError("Minimal 8 baris diperlukan untuk perbandingan cross-validation.")
    if task == "classification":
        counts = y.value_counts()
        if len(counts) < 2 or counts.min() < 2:
            raise TrainingError("Setiap kelas harus memiliki minimal dua observasi untuk CV.")
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        n_splits = min(int(cv), int(y_train.value_counts().min()))
        if n_splits < 2:
            raise TrainingError("Tidak cukup observasi per kelas untuk cross-validation.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {"accuracy": "accuracy", "precision_macro": "precision_macro",
                   "recall_macro": "recall_macro", "f1_macro": "f1_macro"}
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise TrainingError("Target regression harus numerik.")
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)
        n_splits = min(int(cv), len(X_train))
        if n_splits < 2:
            raise TrainingError("Tidak cukup observasi untuk cross-validation.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}

    rows = []
    for aid in algorithm_ids:
        info = get_algorithm(aid)
        try:
            min_inner = int(y_train.value_counts().min()) if task == "classification" else 5
            stacking_cv = max(2, min(5, min_inner - 1)) if aid == 26 else 5
            preprocessor = build_preprocessor(X_train, aid)
            estimator = build_estimator(aid, n_train=len(X_train), n_features=X_train.shape[1],
                                        random_state=random_state, stacking_cv=stacking_cv)
            pipeline = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
            scores = cross_validate(pipeline, X_train, y_train, cv=splitter, scoring=scoring,
                                    return_train_score=False, error_score="raise", n_jobs=1)
            row = {"algorithm_id": aid, "algorithm": info.name, "cv_folds": n_splits}
            for metric in scoring:
                values = scores[f"test_{metric}"]
                if metric in {"mae", "rmse"}:
                    values = -values
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
            rows.append(row)
        except Exception as exc:
            rows.append({"algorithm_id": aid, "algorithm": info.name, "cv_folds": n_splits, "error": str(exc)})
    results = pd.DataFrame(rows)
    sort_column = "f1_macro_mean" if task == "classification" else "rmse_mean"
    if sort_column in results.columns:
        results = results.sort_values(sort_column, ascending=(task == "regression"), na_position="last")
    return results.reset_index(drop=True)
