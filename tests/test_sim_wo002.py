import numpy as np
from parapix.sim import student_t_noise, vol_cluster

def test_student_t_repro():
    a = student_t_noise(100, 42, 1.0, 5)
    b = student_t_noise(100, 42, 1.0, 5)
    assert np.array_equal(a, b)

def test_vol_cluster_repro():
    a = vol_cluster(100, 42, 1.0)
    b = vol_cluster(100, 42, 1.0)
    assert np.array_equal(a, b)

def test_vol_cluster_ac():
    x = vol_cluster(20000, 7, 1.0)
    r_ac = np.corrcoef(x[1:], x[:-1])[0, 1]
    s_ac = np.corrcoef(x[1:]**2, x[:-1]**2)[0, 1]
    assert s_ac > r_ac
