"""Numerical functions for evaluating tetrahedron bounds for 4-embeddable metric spaces.
Duplicates code from the juypter notebook of the same name."""

import numpy as np
import warnings
from sympy import *


o0 = Symbol('o0')
o1 = Symbol('o1')
p = Symbol('p')
q0 = Symbol('q0')
q1 = Symbol('q1')
# Code generated in the tetrahedron.ipynb using sympy.printing.print_python
lower_bound_symbolic = sqrt((-2 * sqrt(
    (-o0 / 2 + o1 / 2 + p / 2) * (o0 / 2 - o1 / 2 + p / 2) * (o0 / 2 + o1 / 2 - p / 2)) * sqrt(
    o0 / 2 + o1 / 2 + p / 2) / p + 2 * sqrt(
    (-p / 2 + q0 / 2 + q1 / 2) * (p / 2 - q0 / 2 + q1 / 2) * (p / 2 + q0 / 2 - q1 / 2)) * sqrt(
    p / 2 + q0 / 2 + q1 / 2) / p) ** 2 + (-Piecewise((-sqrt(
    o0 ** 2 - 4 * (-o0 / 2 + o1 / 2 + p / 2) * (o0 / 2 - o1 / 2 + p / 2) * (o0 / 2 + o1 / 2 - p / 2) * (
            o0 / 2 + o1 / 2 + p / 2) / p ** 2), o1 ** 2 > o0 ** 2 + p ** 2), (sqrt(
    o0 ** 2 - 4 * (-o0 / 2 + o1 / 2 + p / 2) * (o0 / 2 - o1 / 2 + p / 2) * (o0 / 2 + o1 / 2 - p / 2) * (
            o0 / 2 + o1 / 2 + p / 2) / p ** 2), true)) + Piecewise((-sqrt(
    q0 ** 2 - 4 * (-p / 2 + q0 / 2 + q1 / 2) * (p / 2 - q0 / 2 + q1 / 2) * (p / 2 + q0 / 2 - q1 / 2) * (
            p / 2 + q0 / 2 + q1 / 2) / p ** 2), q1 ** 2 > p ** 2 + q0 ** 2), (sqrt(
    q0 ** 2 - 4 * (-p / 2 + q0 / 2 + q1 / 2) * (p / 2 - q0 / 2 + q1 / 2) * (p / 2 + q0 / 2 - q1 / 2) * (
            p / 2 + q0 / 2 + q1 / 2) / p ** 2), true))) ** 2)
upper_bound_symbolic = sqrt(
    (2 * sqrt((-o0 / 2 + o1 / 2 + p / 2) * (o0 / 2 - o1 / 2 + p / 2) * (o0 / 2 + o1 / 2 - p / 2)) * sqrt(
        o0 / 2 + o1 / 2 + p / 2) / p + 2 * sqrt(
        (-p / 2 + q0 / 2 + q1 / 2) * (p / 2 - q0 / 2 + q1 / 2) * (p / 2 + q0 / 2 - q1 / 2)) * sqrt(
        p / 2 + q0 / 2 + q1 / 2) / p) ** 2 + (-Piecewise((-sqrt(
        o0 ** 2 - 4 * (-o0 / 2 + o1 / 2 + p / 2) * (o0 / 2 - o1 / 2 + p / 2) * (o0 / 2 + o1 / 2 - p / 2) * (
                o0 / 2 + o1 / 2 + p / 2) / p ** 2), o1 ** 2 > o0 ** 2 + p ** 2), (sqrt(
        o0 ** 2 - 4 * (-o0 / 2 + o1 / 2 + p / 2) * (o0 / 2 - o1 / 2 + p / 2) * (o0 / 2 + o1 / 2 - p / 2) * (
                o0 / 2 + o1 / 2 + p / 2) / p ** 2), true)) + Piecewise((-sqrt(
        q0 ** 2 - 4 * (-p / 2 + q0 / 2 + q1 / 2) * (p / 2 - q0 / 2 + q1 / 2) * (p / 2 + q0 / 2 - q1 / 2) * (
                p / 2 + q0 / 2 + q1 / 2) / p ** 2), q1 ** 2 > p ** 2 + q0 ** 2), (sqrt(
        q0 ** 2 - 4 * (-p / 2 + q0 / 2 + q1 / 2) * (p / 2 - q0 / 2 + q1 / 2) * (p / 2 + q0 / 2 - q1 / 2) * (
                p / 2 + q0 / 2 + q1 / 2) / p ** 2), true))) ** 2)

_lower_bound = lambdify([p, q0, q1, o0, o1], lower_bound_symbolic, "numpy")
_upper_bound = lambdify([p, q0, q1, o0, o1], upper_bound_symbolic, "numpy")


def lower_bound(
        pivot_pivot_dist,
        p0_query_dist,
        p1_query_dist,
        p0_object_dist,
        p1_object_dist, ):
    return _lower_bound(pivot_pivot_dist, p0_query_dist, p1_query_dist, p0_object_dist, p1_object_dist)

def upper_bound(
        pivot_pivot_dist,
        p0_query_dist,
        p1_query_dist,
        p0_object_dist,
        p1_object_dist, ):
    return _upper_bound(pivot_pivot_dist, p0_query_dist, p1_query_dist, p0_object_dist, p1_object_dist)


height_over_base = sqrt(-(-o0 - o1 + p)*(-o0 + o1 + p)*(o0 - o1 + p))*sqrt(o0 + o1 + p)/(2*p)
width_relative_to_p0 = Piecewise((-sqrt(o0**2 + (-o0 - o1 + p)*(-o0 + o1 + p)*(o0 - o1 + p)*(o0 + o1 + p)/(4*p**2)), o1**2 > o0**2 + p**2), (sqrt(o0**2 + (-o0 - o1 + p)*(-o0 + o1 + p)*(o0 - o1 + p)*(o0 + o1 + p)/(4*p**2)), true))

def project_to_2d_euclidean(points, p0, p1, dist_func):
    numeric_p = dist_func(p0,p1)
    numeric_o0 = dist_func(points, p0)
    numeric_o1 = dist_func(points, p1)
    
    h = height_over_base.subs({p:numeric_p})
    h = lambdify([o0,o1], h, "numpy")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # if the pivots are part of `points`, we get a warning form labdify
        y = h(numeric_o0, numeric_o1)

    # if the pivots are part of `points`, their y is calculated as 'nan'
    y[(numeric_o0 == 0) | (numeric_o1 == 0)] = 0

    w = width_relative_to_p0.subs({p:numeric_p})
    w = lambdify([o0,o1], w, "numpy")
    x = w(numeric_o0, numeric_o1)

    rv = np.empty([len(points), 2])
    rv[:,0] = x
    rv[:,1] = y
    return rv
    
    
    