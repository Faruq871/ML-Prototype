from ml_workbench.algorithms.catalog import ALGORITHM_CATALOG, get_algorithms

def test_catalog_contains_40_algorithms():
    assert len(ALGORITHM_CATALOG) == 40
    assert len(get_algorithms("classification")) > 0
    assert len(get_algorithms("regression")) > 0
