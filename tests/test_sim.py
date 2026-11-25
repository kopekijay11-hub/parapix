import numpy as np
from parallax.sim import iid_noise, injected_edge, ar1

def test_repro_iid():
    a = iid_noise(100, 42, 1.0)
    b = iid_noise(100, 42, 1.0)
    assert np.array_equal(a, b)

def test_repro_edge():
    a = injected_edge(100, 42, 1.0, 0.5)
    b = injected_edge(100, 42, 1.0, 0.5)
    assert np.array_equal(a, b)

def test_repro_ar1():
    a = ar1(100, 42, 1.0, 0.3)
    b = ar1(100, 42, 1.0, 0.3)
    assert np.array_equal(a, b)

def test_different_seeds():
    a = iid_noise(100, 1, 1.0)
    b = iid_noise(100, 2, 1.0)
    assert not np.array_equal(a, b)

def test_ar1_autocorr():
    x = ar1(200000, 7, 1.0, 0.3)
    ac = np.corrcoef(x[1:], x[:-1])[0, 1]
    assert abs(ac - 0.3) < 0.02
