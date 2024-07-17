"""Numerical functions for evaluating tetrahedron bounds for 4-embeddable metric spaces.
Duplicates code from the juypter notebook of the same name."""

import numpy as np
import numpy.testing as npt
import warnings
from scipy import spatial
from sympy import lambdify, sqrt, Piecewise, Symbol, true, Rational

from joblib import Memory

mem = Memory("/tmp/joblib")


o0 = Symbol("o0")
o1 = Symbol("o1")
p = Symbol("p")
q0 = Symbol("q0")
q1 = Symbol("q1")

# Code generated in the tetrahedron.ipynb using mathematica and sympy.printing.print_python
lower_bound_symbolic = sqrt(
    (
        sqrt(-(o0 - o1 - p) * (o0 - o1 + p) * (o0 + o1 - p)) * sqrt(o0 + o1 + p)
        - sqrt(-(p - q0 - q1) * (p - q0 + q1) * (p + q0 - q1)) * sqrt(p + q0 + q1)
    )
    ** 2
    + (o0**2 - o1**2 - q0**2 + q1**2) ** 2
) / (2 * p)
upper_bound_symbolic = (
    o0**2 * p**2
    + p**2 * q0**2
    + sqrt((-o0 + o1 + p) * (o0 - o1 + p) * (o0 + o1 - p))
    * sqrt((-p + q0 + q1) * (p - q0 + q1) * (p + q0 - q1))
    * sqrt(o0 + o1 + p)
    * sqrt(p + q0 + q1)
    / 2
    - (o0**2 - o1**2 + p**2) * (p**2 + q0**2 - q1**2) / 2
) ** Rational(1, 4) / sqrt(p)

_lower_bound = lambdify([p, q0, q1, o0, o1], lower_bound_symbolic, "numpy")
_upper_bound = lambdify([p, q0, q1, o0, o1], upper_bound_symbolic, "numpy")


def lower_bound(
    pivot_pivot_dist,
    p0_query_dist,
    p1_query_dist,
    p0_object_dist,
    p1_object_dist,
):
    return _lower_bound(
        pivot_pivot_dist, p0_query_dist, p1_query_dist, p0_object_dist, p1_object_dist
    )


def upper_bound(
    pivot_pivot_dist,
    p0_query_dist,
    p1_query_dist,
    p0_object_dist,
    p1_object_dist,
):
    return _upper_bound(
        pivot_pivot_dist, p0_query_dist, p1_query_dist, p0_object_dist, p1_object_dist
    )


sympy_height_over_base = sqrt(
    -(-o0 - o1 + p) * (-o0 + o1 + p) * (o0 - o1 + p) * (o0 + o1 + p)
) / (2 * p)
sympy_width_relative_to_p0 = (o0**2 - o1**2 + p**2) / (2 * p)

__N_height_over_base = lambdify([o0, o1, p], sympy_height_over_base, "numpy", cse=True)
__width_relative_to_p0 = lambdify(
    [o0, o1, p], sympy_width_relative_to_p0, "numpy", cse=True
)


def height_over_base(o0_, o1_, p_):
    if p_.size == 1:
        p_ = np.full_like(o0_, p_)
    return __N_height_over_base(o0_, o1_, p_)


def width_relative_to_p0(o0_, o1_, p_):
    if p_.size == 1:
        p_ = np.full_like(o0_, p_)
    return __width_relative_to_p0(o0_, o1_, p_)


def project_to_2d_euclidean(points, p0, p1, dist_func):
    assert not np.allclose(p0, p1), "You need to choose two distinct pivots!"
    numeric_p = dist_func(p0, p1)
    numeric_o0 = dist_func(points, p0)
    numeric_o1 = dist_func(points, p1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # if the pivots are part of `points`, we get a warning form labdify
        y = height_over_base(numeric_o0, numeric_o1, numeric_p)
    # if the pivots are part of `points`, their y is calculated as 'nan'
    y[(numeric_o0 == 0) | (numeric_o1 == 0)] = 0

    x = width_relative_to_p0(numeric_o0, numeric_o1, numeric_p)

    rv = np.column_stack((x, y))
    return rv


def euclidean2d_lower_bound_dist(x, y):
    return spatial.minkowski_distance(x, y)


def euclidean2d_upper_bound_dist(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), "float64")

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    return np.sum((y[0] - x[0]) ** 2 + (y[1] + x[1]) ** 2, axis=-1)


def upper_bound_dist_matrix(x, y):
    ubs = np.asarray([euclidean2d_upper_bound_dist(a, b) for a in x for b in y])
    return ubs.reshape(len(x), len(y))
