import numpy as np
from parapix.permtest import permutation_pvalue

def test_perm_deterministic():
    x = np.arange(100, dtype=float)
    assert permutation_pvalue(x, 1) == permutation_pvalue(x, 1)

def test_perm_negative_near_one():
    x = -np.ones(100)
    assert permutation_pvalue(x, 1) > 0.9

def test_perm_strong_edge_small():
    x = np.ones(100) * 5
    assert permutation_pvalue(x, 1) < 0.01
