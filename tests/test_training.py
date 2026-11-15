from ml_workbench.training.trainer import train_model

def test_random_forest_classifier_runs(classification_df):
    result = train_model(classification_df, 15, target="target", random_state=7)
    assert "accuracy" in result.metrics
    assert len(result.output_frame) > 0

def test_ridge_regressor_runs(regression_df):
    result = train_model(regression_df, 2, target="target", random_state=7)
    assert "rmse" in result.metrics
    assert result.metrics["rmse"] >= 0

def test_kmeans_runs(classification_df):
    result = train_model(classification_df.drop(columns=["target"]), 32, n_clusters=3, random_state=7)
    assert "cluster_count" in result.metrics


def test_model_comparison_runs(classification_df):
    from ml_workbench.training.comparison import compare_models
    table = compare_models(classification_df, [5, 15], target="target", cv=3, random_state=7)
    assert len(table) == 2
    assert "f1_macro_mean" in table.columns
