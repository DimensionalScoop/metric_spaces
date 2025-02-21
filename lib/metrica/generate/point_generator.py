import numpy as np
from sklearn.datasets import make_blobs

DEFAULT_SAMPLE_SIZE = 120


def skew(points, rng):
    """stretch each axis by a random factor"""
    scale = 1 + 10 * rng.random(size=points.shape[-1])
    return points * scale[np.newaxis, :]


def generate_univar_points(rng, n_samples=DEFAULT_SAMPLE_SIZE, dim=7, homogenious=True):
    points = rng.random(size=[n_samples, dim]) * 37 - 37 / 2
    if not homogenious:
        points = skew(points, rng)
    return points


def generate_univar_grid_points(rng, n_samples=DEFAULT_SAMPLE_SIZE, dim=7):
    points = rng.integers(low=37 // 2, high=37 * 2, size=[n_samples, dim])
    return points


def generate_gaussian_points(
    rng, n_samples=DEFAULT_SAMPLE_SIZE, dim=7, homogenious=False
):
    points = rng.standard_normal([n_samples, dim])
    if not homogenious:
        points = skew(points, rng)
    return points


def generate_multi_cluster(rng, n_samples=DEFAULT_SAMPLE_SIZE, dim=7):
    std = 3  # good overlap, but still differentiable
    points, _ = make_blobs(
        n_samples=n_samples,
        n_features=dim,
        centers=4,
        cluster_std=std,
        random_state=rng.integers(np.iinfo("int32").max),
    )
    return points


def generate_sparse_cluster(rng, n_samples=DEFAULT_SAMPLE_SIZE, dim=7):
    std = 1  # almost no overlap
    points, _ = make_blobs(
        n_samples=n_samples,
        n_features=dim,
        centers=4,
        cluster_std=std,
        random_state=rng.integers(np.iinfo("int32").max),
    )
    return points


def get_generator_dict(n_samples=DEFAULT_SAMPLE_SIZE):
    return {
        "univariate, idd": lambda **kwargs: generate_univar_points(
            **kwargs, homogenious=False, n_samples=n_samples
        ),
        "univariate, stretched": lambda **kwargs: generate_univar_points(
            **kwargs, homogenious=False, n_samples=n_samples
        ),
        "gaussian, circular": lambda **kwargs: generate_gaussian_points(
            **kwargs, homogenious=True, n_samples=n_samples
        ),
        "gaussian, eliptic": lambda **kwargs: generate_gaussian_points(
            **kwargs, homogenious=False, n_samples=n_samples
        ),
        "clusters, overlapping": lambda **kwargs: generate_multi_cluster(
            **kwargs, n_samples=n_samples
        ),
        "clusters, sparse": lambda **kwargs: generate_sparse_cluster(
            **kwargs, n_samples=n_samples
        ),
    }
