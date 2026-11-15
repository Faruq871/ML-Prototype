from __future__ import annotations
import pandas as pd
from ml_workbench.algorithms.catalog import get_algorithms, get_algorithm
from ml_workbench.data.profiler import profile_dataframe
from ml_workbench.exceptions import RecommendationError
from ml_workbench.schemas import Recommendation

BASELINES = {
    "classification": [5, 15, 19, 11, 27, 17],
    "regression": [1, 2, 16, 20, 18, 12],
    "clustering": [32, 33, 34, 35, 37],
    "dimensionality_reduction": [38, 39],
    "anomaly_detection": [40],
}

class RecommendationEngine:
    """Explainable, rule-based candidate recommender for tabular datasets."""

    def recommend(self, data: pd.DataFrame, task: str, target: str | None = None,
                  limit: int = 6) -> list[Recommendation]:
        if task not in BASELINES:
            raise RecommendationError(f"Task tidak didukung: {task}")
        if task in {"classification", "regression"} and (target is None or target not in data.columns):
            raise RecommendationError("Pilih target variable yang valid untuk supervised learning.")

        profile = profile_dataframe(data, target if task == "classification" else None)
        candidates = BASELINES[task]
        scored = []
        for aid in candidates:
            info = get_algorithm(aid)
            score = 60.0
            rationale = ["Termasuk baseline yang umum dipakai untuk task ini."]
            caveats = []
            if task in {"classification", "regression"}:
                if profile.rows < 500:
                    if aid in {5, 1, 2, 13, 14, 27}:
                        score += 10
                        rationale.append("Ukuran dataset relatif kecil; model baseline yang lebih sederhana layak dicoba.")
                elif profile.rows >= 10000:
                    if aid in {15, 16, 17, 18, 19, 20, 21, 22}:
                        score += 10
                        rationale.append("Jumlah observasi relatif besar; ensemble pohon layak dievaluasi.")
                if len(profile.categorical_columns) > len(profile.numeric_columns):
                    if aid in {15, 16, 17, 18, 19, 20}:
                        score += 5
                        rationale.append("Dataset memiliki banyak fitur kategorikal; pipeline memakai one-hot encoding.")
                if len(profile.numeric_columns) > 0 and len(profile.numeric_columns) >= 30:
                    if aid in {2, 3, 4, 5, 11, 12}:
                        score += 5
                        rationale.append("Ada cukup banyak fitur numerik; regularisasi atau margin-based model dapat dibandingkan.")
                if task == "classification":
                    counts = data[target].dropna().value_counts()
                    if len(counts) > 1:
                        minority = counts.min() / counts.sum()
                        if minority < 0.15:
                            if aid in {5, 15, 17, 19}:
                                score += 4
                                rationale.append("Target terlihat imbalanced; evaluasi macro-F1, recall, dan confusion matrix.")
                            caveats.append("Belum ada penanganan class imbalance otomatis; pertimbangkan class_weight atau resampling setelah baseline.")
                if profile.missing_ratio > 0.05:
                    rationale.append("Ada missing values; preprocessing akan melakukan imputasi di dalam pipeline.")
                if profile.high_cardinality_columns:
                    caveats.append("Fitur berkardinalitas tinggi dapat menghasilkan one-hot matrix yang besar.")
            elif task == "clustering":
                if aid == 33 and profile.rows >= 10000:
                    score += 15
                    rationale.append("Mini-Batch K-Means lebih hemat untuk dataset besar.")
                if aid == 34:
                    rationale.append("DBSCAN dapat menandai noise, tetapi sensitif terhadap parameter eps dan skala fitur.")
                    caveats.append("DBSCAN dapat menghasilkan satu cluster atau seluruh titik sebagai noise.")
                if aid in {32, 33, 35, 37}:
                    rationale.append("Model ini memerlukan jumlah cluster yang ditentukan pengguna.")
            elif task == "dimensionality_reduction":
                rationale.append("Gunakan explained variance dan/atau performa downstream untuk menilai representasi hasil.")
                if aid == 39:
                    caveats.append("Truncated SVD tidak melakukan centering; hasilnya tidak identik dengan PCA.")
            elif task == "anomaly_detection":
                rationale.append("Isolation Forest tidak membutuhkan label anomaly saat training.")
                caveats.append("Tanpa label validasi, hasil deteksi perlu ditinjau berdasarkan konteks domain.")
            scored.append(Recommendation(aid, info.name, min(score, 99.0), rationale, caveats))
        scored.sort(key=lambda item: (-item.score, item.algorithm_id))
        return scored[:max(1, limit)]
