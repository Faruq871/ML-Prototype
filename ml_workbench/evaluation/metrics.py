from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score)


def evaluate_supervised(task: str, estimator, X_test: pd.DataFrame, y_test: pd.Series,
                        predictions: np.ndarray) -> dict:
    if task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        }
        if hasattr(estimator, "predict_proba"):
            try:
                probabilities = estimator.predict_proba(X_test)
                classes = getattr(estimator, "classes_", None)
                if probabilities.shape[1] == 2 and classes is not None:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
                elif probabilities.shape[1] > 2:
                    metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro", labels=classes))
            except (ValueError, TypeError):
                pass
        return metrics
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan"),
    }


def evaluate_unsupervised(task: str, estimator, transformed: np.ndarray, labels=None) -> dict:
    metrics = {}
    if task == "clustering" and labels is not None:
        labels = np.asarray(labels)
        noise_mask = labels != -1
        valid_labels = labels[noise_mask]
        valid_X = transformed[noise_mask]
        unique = np.unique(valid_labels)
        metrics["cluster_count"] = int(len(unique))
        metrics["noise_count"] = int(np.sum(~noise_mask))
        if 1 < len(unique) < len(valid_labels):
            try:
                sample_size = min(2000, len(valid_labels))
                metrics["silhouette_score"] = float(silhouette_score(valid_X, valid_labels, sample_size=sample_size, random_state=42))
            except ValueError:
                metrics["silhouette_score"] = None
        else:
            metrics["silhouette_score"] = None
    elif task == "anomaly_detection" and labels is not None:
        labels = np.asarray(labels)
        metrics["normal_count"] = int(np.sum(labels == 1))
        metrics["anomaly_count"] = int(np.sum(labels == -1))
    elif task == "dimensionality_reduction":
        if hasattr(estimator, "explained_variance_ratio_"):
            ratios = estimator.explained_variance_ratio_
            metrics["explained_variance_ratio"] = [float(x) for x in ratios]
            metrics["explained_variance_total"] = float(np.sum(ratios))
    return metrics
