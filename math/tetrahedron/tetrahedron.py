"""Numerical functions for evaluating tetrahedron bounds for 4-embeddable metric spaces.
Duplicates code from the juypter notebook of the same name."""

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

_lower_bound = lambdify([p, q_0, q_1, o_0, o_1], lower_bound_symbolic, "numpy")
_upper_bound = lambdify([p, q_0, q_1, o_0, o_1], upper_bound_symbolic, "numpy")


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
