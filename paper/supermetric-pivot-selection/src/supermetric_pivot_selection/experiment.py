from uuid import uuid4
from joblib.parallel import itertools
import numpy as np
import pandas as pd
import polars as pl
import duckdb
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm
from datetime import datetime, timedelta
from pprint import pprint
import platform
import time
import psutil
import json
import itertools

from meters import pivot_selection
from meters.generate import point_generator
from meters.metric.metric import Euclid
from meters.tetrahedron import proj_quality, tetrahedron

ALL_ALGORITHMS = pivot_selection.get_selection_algos()
RAISE_EXCEPTIONS = True


def run(seed: int, algorithm: str, dataset_type: str, dim: int, config: dict) -> dict:
    """Conducts a single experiment with the given parameters"""

    generate_points = point_generator.GENERATORS[dataset_type]
    if config["metric"] == ("Euclidean", 2):
        metric = Euclid(2)
    else:
        NotImplementedError()
    select_pivots = ALL_ALGORITHMS[algorithm]
    rng = np.random.default_rng(seed)

    start_time = datetime.now()
    points = generate_points(rng=rng, dim=dim, n_samples=config["n_samples"])
    r = proj_quality.get_average_k_nn_dist(points, metric, k=10)
    try:
        p0, p1 = select_pivots(points, rng=rng)
        points_p = tetrahedron.project_to_2d_euclidean(points, p0, p1, metric)
        candidate_set_size = proj_quality.candidate_set_size(points_p, r, metric)
        useful_partition_size = proj_quality.hilbert_quality(points_p, r)
    except Exception as e:
        error = _generate_error_message(
            e, seed, algorithm, dataset_type, dim, config, start_time
        )
        if RAISE_EXCEPTIONS:
            raise
        return error

    rv = _create_return_value(config, algorithm, seed, dataset_type, dim, start_time)

    rv["candidate_set_size"] = candidate_set_size
    rv["userful_partition_size"] = useful_partition_size
    rv["notes"] = "{}"

    return rv


def _create_return_value(config, algorithm, seed, dataset_type, dim, start_time):
    rv = dict(
        experiment_id=config["experiment_id"],
        algorithm=algorithm,
        seed=seed,
    )
    if ", " in dataset_type:
        rv["dataset_supertype"], rv["dataset_subtype"] = dataset_type.split(", ")
    else:
        rv["dataset_supertype"] = dataset_type
        rv["dataset_subtype"] = ""
    rv["dim"] = dim
    rv["runtime_sec"] = (datetime.now() - start_time).total_seconds()
    return rv


def _generate_error_message(
    exception,
    seed: int,
    algorithm: str,
    dataset_type: str,
    dim: int,
    config: dict,
    start_time,
):
    rv = _create_return_value(config, algorithm, seed, dataset_type, dim, start_time)
    rv["candidate_set_size"] = -1
    rv["useful_partition_size"] = -1
    rv["notes"] = json.dumps(
        dict(error_type=type(exception).__name__, message=str(exception))
    )

    return rv
