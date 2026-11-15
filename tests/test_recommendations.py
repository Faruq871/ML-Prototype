from ml_workbench.recommendations.engine import RecommendationEngine

def test_classification_recommendations(classification_df):
    result = RecommendationEngine().recommend(classification_df, "classification", target="target")
    assert result
    assert result[0].algorithm_id in {5, 15, 19, 11, 27, 17}
