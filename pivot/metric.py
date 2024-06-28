"""Calculate the distances between two points or a list of points"""
import numpy as np

# TODO: Replace minkwosky distances with scipy.spacial.minkowski_distance

def _preproc_points(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    # do we have two points, or lists of points?
    dim = max(len(a.shape), len(b.shape))
    assert 1 <= dim <= 2
    return a, b, dim


class Metric:
    def __init__(self, name):
        self.name = name

    def _calc_distance(self, a: np.ndarray, b: np.ndarray, is_list: bool) -> np.array:
        pass

    def __call__(self, a, b):
        a, b, dim = _preproc_points(a, b)
        return self._calc_distance(a, b, dim == 2)


class ProbMetric(Metric):
    """A metric that compares two probability distributions
    or two lists of prob dist pairwise."""

    def _check_prob(self, p):
        if np.any(p < 0):
            raise ValueError("expected probability distributions, got negative values.")
        if not np.allclose(1, np.sum(p, axis=-1)):
            raise ValueError("expected probability distributions, but distribution does not sum to 1")

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.array:
        self._check_prob(a)
        self._check_prob(b)
        return super().__call__(a, b)


# TODO: refactor Euclid, Cheb, Manha to use Minkowski
class Euclid(Metric):
    def __init__(self):
        super().__init__(r"\|\cdot\|_2")

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        d_squared = np.sum((a - b) ** 2, axis=axis)
        return np.sqrt(d_squared)


class Chebyshev(Metric):
    def __init__(self):
        super().__init__(r"\|\cdot\|_\infty")

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        return np.max(np.abs(a - b), axis=axis)


class Manhattan(Metric):
    def __init__(self):
        super().__init__(r"\|\cdot\|_1")

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        return np.sum(np.abs(a - b), axis=axis)


class Minkowski(Metric):
    def __init__(self, p):
        self.p = p
        super().__init__(r"\|\cdot\|_" + p)

    def _calc_distance(self, a, b, is_list):
        axis = 1 if is_list else 0
        sum_ = np.sum((a - b) ** self.p, axis=axis)
        return sum_ ** (1 / self.p)


class JensenShannon(ProbMetric):
    """The metric-ized version of the divergence.
    Implemented from Connor's 2016 Hilbert Exclusion paper."""

    def __init__(self):
        super().__init__(r"Jenson-Shannon-Distance")

    def _divergence(self, v, w):
        """v and w: probability distributions or lists of probability distributions."""

        def H(x):
            return -x * np.log2(x)

        mixture = H(v) + H(w) - H(v + w)
        return 1 - 0.5 * mixture.sum(axis=-1)

    def _calc_distance(self, a: np.ndarray, b: np.ndarray, is_list: bool) -> np.array:
        return np.sqrt(self._divergence(a, b))


class Triangular(ProbMetric):
    """Topsøe's triangular discrimination,
    implemented from Connor's 2016 Hilbert Exclusion paper."""

    def __init__(self):
        super().__init__(r"Triangular Discrimination")

    def _calc_distance(self, a: np.ndarray, b: np.ndarray, is_list: bool) -> np.array:
        k = (a - b)**2 / (a+b)
        return np.sqrt(k.sum(axis=-1))
