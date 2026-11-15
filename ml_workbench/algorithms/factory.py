from __future__ import annotations
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, AdaBoostClassifier, BaggingClassifier, VotingClassifier,
    StackingClassifier, IsolationForest,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD
from ml_workbench.algorithms.catalog import get_algorithm


def build_estimator(algorithm_id: int, *, n_train: int, n_features: int, n_clusters: int = 3, n_components: int = 2,
                    random_state: int = 42, stacking_cv: int = 5):
    """Create a fresh scikit-learn estimator for an algorithm catalog ID."""
    get_algorithm(algorithm_id)  # validate ID before constructing anything
    n_train = max(int(n_train), 1)
    k = min(5, n_train)

    # Factories are lazy: constructing one algorithm does not instantiate the other 39.
    factories = {
        1: lambda: LinearRegression(),
        2: lambda: Ridge(alpha=1.0),
        3: lambda: Lasso(alpha=0.1, max_iter=10000),
        4: lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        5: lambda: LogisticRegression(max_iter=2000),
        6: lambda: SGDClassifier(max_iter=2000, tol=1e-3, random_state=random_state),
        7: lambda: Perceptron(max_iter=2000, random_state=random_state),
        8: lambda: PassiveAggressiveClassifier(max_iter=2000, random_state=random_state),
        9: lambda: KNeighborsClassifier(n_neighbors=k),
        10: lambda: KNeighborsRegressor(n_neighbors=k),
        11: lambda: SVC(probability=True, random_state=random_state),
        12: lambda: SVR(),
        13: lambda: DecisionTreeClassifier(random_state=random_state),
        14: lambda: DecisionTreeRegressor(random_state=random_state),
        15: lambda: RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        16: lambda: RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        17: lambda: ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        18: lambda: ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        19: lambda: GradientBoostingClassifier(random_state=random_state),
        20: lambda: GradientBoostingRegressor(random_state=random_state),
        21: lambda: HistGradientBoostingClassifier(random_state=random_state),
        22: lambda: HistGradientBoostingRegressor(random_state=random_state),
        23: lambda: AdaBoostClassifier(random_state=random_state),
        24: lambda: BaggingClassifier(estimator=DecisionTreeClassifier(random_state=random_state),
                                     n_estimators=100, random_state=random_state, n_jobs=-1),
        25: lambda: VotingClassifier(estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
            ("svc", SVC(probability=True, random_state=random_state)),
        ], voting="hard"),
        26: lambda: StackingClassifier(estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
            ("svc", SVC(probability=True, random_state=random_state)),
        ], final_estimator=LogisticRegression(max_iter=2000), cv=max(2, int(stacking_cv))),
        27: lambda: GaussianNB(),
        28: lambda: MultinomialNB(alpha=1.0),
        29: lambda: BernoulliNB(alpha=1.0),
        30: lambda: LinearDiscriminantAnalysis(),
        31: lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
        32: lambda: KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state),
        33: lambda: MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=random_state, batch_size=256),
        34: lambda: DBSCAN(eps=0.5, min_samples=5),
        35: lambda: AgglomerativeClustering(n_clusters=n_clusters),
        36: lambda: SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
                                      n_neighbors=min(10, max(n_train - 1, 1)), random_state=random_state),
        37: lambda: GaussianMixture(n_components=n_clusters, random_state=random_state),
        38: lambda: PCA(n_components=max(1, min(int(n_components), n_train, n_features))),
        39: lambda: TruncatedSVD(n_components=max(1, min(int(n_components), n_features, n_train)), random_state=random_state),
        40: lambda: IsolationForest(contamination="auto", random_state=random_state, n_jobs=-1),
    }
    return factories[algorithm_id]()
