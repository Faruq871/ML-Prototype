from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class AlgorithmInfo:
    id: int
    name: str
    task: str
    family: str
    description: str
    scaled: bool = False

# Tabular-compatible subset of the source guidebook's 40 scikit-learn entries.
_ITEMS = [
    (1, "Linear Regression", "regression", "Linear models", "Linear baseline for continuous targets.", True),
    (2, "Ridge Regression", "regression", "Linear models", "L2-regularized linear regression.", True),
    (3, "Lasso Regression", "regression", "Linear models", "L1-regularized regression with feature selection behavior.", True),
    (4, "Elastic Net", "regression", "Linear models", "Combines L1 and L2 regularization.", True),
    (5, "Logistic Regression", "classification", "Linear models", "Interpretable linear classification baseline.", True),
    (6, "SGD Classifier", "classification", "Linear models", "Scalable linear classifier trained with stochastic gradient descent.", True),
    (7, "Perceptron", "classification", "Linear models", "Simple linear classifier trained with the perceptron update rule.", True),
    (8, "Passive Aggressive Classifier", "classification", "Linear models", "Online-style linear classifier.", True),
    (9, "K-Nearest Neighbors Classifier", "classification", "Nearest neighbors", "Distance-based classifier; scaling is important.", True),
    (10, "K-Nearest Neighbors Regressor", "regression", "Nearest neighbors", "Distance-based regression; scaling is important.", True),
    (11, "Support Vector Classifier (SVC)", "classification", "Support vector machines", "Margin-based classifier with a nonlinear RBF default.", True),
    (12, "Support Vector Regression (SVR)", "regression", "Support vector machines", "Margin-based regression with an RBF default.", True),
    (13, "Decision Tree Classifier", "classification", "Trees and ensembles", "Nonlinear tree classifier with limited preprocessing assumptions.", False),
    (14, "Decision Tree Regressor", "regression", "Trees and ensembles", "Nonlinear tree regressor.", False),
    (15, "Random Forest Classifier", "classification", "Trees and ensembles", "Bagged decision trees for robust tabular classification.", False),
    (16, "Random Forest Regressor", "regression", "Trees and ensembles", "Bagged decision trees for tabular regression.", False),
    (17, "Extra Trees Classifier", "classification", "Trees and ensembles", "Highly randomized tree ensemble.", False),
    (18, "Extra Trees Regressor", "regression", "Trees and ensembles", "Highly randomized tree ensemble for regression.", False),
    (19, "Gradient Boosting Classifier", "classification", "Trees and ensembles", "Sequential boosting of shallow decision trees.", False),
    (20, "Gradient Boosting Regressor", "regression", "Trees and ensembles", "Boosted tree regression.", False),
    (21, "Histogram Gradient Boosting Classifier", "classification", "Trees and ensembles", "Histogram-based gradient boosting for larger tabular datasets.", False),
    (22, "Histogram Gradient Boosting Regressor", "regression", "Trees and ensembles", "Histogram-based gradient boosting for regression.", False),
    (23, "AdaBoost Classifier", "classification", "Trees and ensembles", "Adaptive boosting classifier.", False),
    (24, "Bagging Classifier", "classification", "Trees and ensembles", "Bootstrap aggregation of decision trees.", False),
    (25, "Voting Classifier", "classification", "Ensembles", "Combines distinct base classifiers with hard voting.", False),
    (26, "Stacking Classifier", "classification", "Ensembles", "Trains a meta-classifier on cross-validated base predictions.", False),
    (27, "Gaussian Naive Bayes", "classification", "Probabilistic models", "Probabilistic classifier using Gaussian likelihoods.", False),
    (28, "Multinomial Naive Bayes", "classification", "Probabilistic models", "Naive Bayes for non-negative count-like or weighted features.", False),
    (29, "Bernoulli Naive Bayes", "classification", "Probabilistic models", "Naive Bayes for binary-valued features.", False),
    (30, "Linear Discriminant Analysis", "classification", "Discriminant analysis", "Linear class boundaries under shared covariance assumptions.", True),
    (31, "Quadratic Discriminant Analysis", "classification", "Discriminant analysis", "Quadratic class boundaries with class-specific covariance.", False),
    (32, "K-Means", "clustering", "Clustering", "Partitions observations into a specified number of clusters.", True),
    (33, "Mini-Batch K-Means", "clustering", "Clustering", "K-Means variant using mini-batches for faster fitting.", True),
    (34, "DBSCAN", "clustering", "Clustering", "Density-based clustering that can identify noise points.", True),
    (35, "Agglomerative Clustering", "clustering", "Clustering", "Hierarchical bottom-up clustering.", True),
    (36, "Spectral Clustering", "clustering", "Clustering", "Graph-based clustering for non-convex structures.", True),
    (37, "Gaussian Mixture Model", "clustering", "Clustering", "Probabilistic mixture model for soft cluster assignments.", True),
    (38, "Principal Component Analysis (PCA)", "dimensionality_reduction", "Dimensionality reduction", "Linear projection onto directions of high variance.", True),
    (39, "Truncated SVD", "dimensionality_reduction", "Dimensionality reduction", "Low-rank decomposition without centering.", True),
    (40, "Isolation Forest", "anomaly_detection", "Anomaly detection", "Tree-based detector that isolates unusual observations.", False),
]
ALGORITHM_CATALOG = {item[0]: AlgorithmInfo(*item) for item in _ITEMS}


def get_algorithms(task: str | None = None) -> list[AlgorithmInfo]:
    items = list(ALGORITHM_CATALOG.values())
    return [item for item in items if task is None or item.task == task]


def get_algorithm(algorithm_id: int) -> AlgorithmInfo:
    try:
        return ALGORITHM_CATALOG[int(algorithm_id)]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Algorithm ID tidak dikenal: {algorithm_id}") from exc
