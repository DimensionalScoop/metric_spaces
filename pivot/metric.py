"""Calculate the distances between two points or a list of points"""
import numpy as np


def _preproc_points(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    # do we have two points, or lists of points?
    dim = max(len(a.shape), len(b.shape))
    assert 1 <= dim <= 2
    return a,b,dim
    

class Metric:
    def __init__(self, name):
        self.name = name

    def _calc_distance(self, a:np.ndarray, b:np.ndarray, is_list:bool) -> np.array:
        pass

    def __call__(self, a, b):
        a, b, dim = _preproc_points(a, b)
        return self._calc_distance(a, b, dim == 2)


class Euclid(Metric):
    def __init__(self):
        super().__init__(r"\|\cdot\|_2")

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        d_squared = np.sum((a-b)**2, axis=axis)
        return np.sqrt(d_squared)

class Chebyshev(Metric):
    def __init__(self):
        super().__init__(r"\|\cdot\|_\infty")

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        return np.max(np.abs(a-b), axis=axis)

class Manhattan(Metric):
    def __init__(self):
        super().__init__(r"\|\cdot\|_1")

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        return np.sum(np.abs(a-b), axis=axis)

class Minkowski(Metric):
    def __init__(self, p):
        self.p = p
        super().__init__(r"\|\cdot\|_"+p)

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        sum_ = np.sum((a-b)**self.p, axis=axis)
        return sum_**(1/self.p)
