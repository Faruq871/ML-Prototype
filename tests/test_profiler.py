from ml_workbench.data.profiler import profile_dataframe

def test_profile_detects_missing_and_duplicates(classification_df):
    profile = profile_dataframe(classification_df, target="target")
    assert profile.rows == len(classification_df)
    assert profile.missing_cells > 0
    assert profile.class_count == 2
