import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from .metric import JensenShannon, Triangular, PNorm, Euclid


def test_prob():
    p0 = np.array([0, 1, 0])
    p1 = np.array([0, 0.1, 0.9])
    list_of_ps = np.vstack((p0, p1))
    assert list_of_ps.shape == (2, 3)

    metric = JensenShannon()
    metric(p0, p1)
    metric(list_of_ps, p1)

    garbage_a = p0 + 1
    garbage_b = p1 - 10
    garbage_c = list_of_ps.copy()
    garbage_c[0, 1] = 2
    garbage_d = list_of_ps.copy()
    garbage_d[0, 1] = -0.01

    for g in [garbage_a, garbage_b, garbage_c, garbage_d]:
        with pytest.raises(ValueError):
            metric(g, g)


def test_shapes():
    p0 = np.array([0.01, 0.98, 0.01])
    p1 = np.array([0.01, 0.1, 0.89])
    list_of_ps = np.vstack((p0, p1))

    for m in [Triangular(), JensenShannon()]:
        assert np.shape(m(p0, p1)) == tuple()
        assert m(list_of_ps, list_of_ps).shape == (2,)


def test_speedups():
    rng = np.random.default_rng(0xFEED)
    a = rng.random([10, 3]) * 20 - 10
    b = rng.random([10, 3]) * 20 - 10

    target = PNorm(2)
    actual = Euclid()

    assert_almost_equal(actual(a, b), target(a, b))
    assert_almost_equal(actual(a[0], b), target(a[0], b))
    assert_almost_equal(actual(a[0], b[0]), target(a[0], b[0]))
