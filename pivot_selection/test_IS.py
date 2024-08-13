import pytest
import numpy as np

from ..generate import point_generator
from . import heuristics_fast as fast


@pytest.fixture
def rng():
    return np.random.default_rng(0xFEED)


@pytest.fixture
def points(rng):
    GENERATOR = "clusters, overlapping"
    N_SAMPLES = 512
    DIM = 5

    generators = point_generator.get_generator_dict(N_SAMPLES)

    gen_func = generators[GENERATOR]
    points = gen_func(dim=DIM, rng=rng)
    return points


def test_IS_candidates(rng, points):
    target_candidates = int(np.sqrt(len(points)))
    piv_candidates, lhs, rhs = fast._select_IS_candidates(points, np.sqrt, rng)

    match piv_candidates.shape:
        case n, dims:
            assert n == target_candidates
            assert dims == points.shape[1]
        case _:
            raise Exception()

    assert len(lhs) == len(rhs)
    assert len(lhs) == target_candidates

    match lhs.shape:
        case n, dims:
            assert n == target_candidates
            assert dims == points.shape[1]
        case _:
            raise Exception()
