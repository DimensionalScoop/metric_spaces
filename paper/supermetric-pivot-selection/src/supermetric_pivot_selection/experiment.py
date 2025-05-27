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
import logging

from meters import pivot_selection
from meters.generate import point_generator
from meters.metric.metric import Euclid
from meters.tetrahedron import proj_quality, tetrahedron

ALL_ALGORITHMS = pivot_selection.get_selection_algos()
RAISE_EXCEPTIONS = False
logger = logging.getLogger(__name__)


SCHEMA = pl.Schema(
    [
        ("experiment_id", pl.Object),
        ("algorithm", pl.String),
        ("seed", pl.Int64),
        ("dataset_supertype", pl.String),
        ("dataset_subtype", pl.String),
        ("dim", pl.Int64),
        ("runtime_sec", pl.Float64),
        ("candidate_set_size", pl.Float64),
        ("userful_partition_size", pl.Float64),
        ("notes", pl.String),
    ]
)


def run(
    seed: int, algorithm: str, dataset_type: str, dim: int, config: dict
) -> pl.DataFrame:
    """Conducts a single experiment with the given parameters"""

    # record input parameters
    start_time = datetime.now()
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

    # run experiment
    try:
        run_result = _run(seed, algorithm, dataset_type, dim, config)
        rv.update(run_result)
    except Exception as e:
        rv["candidate_set_size"] = -1
        rv["useful_partition_size"] = -1
        rv["notes"] = json.dumps(dict(error_type=type(e).__name__, message=str(e)))

        logger.exception(
            "function arguments",
            dict(seed=seed, algorithm=algorithm, dataset_type=dataset_type, dim=dim),
        )
        logger.info("CONFIG for above exception", config)
        if RAISE_EXCEPTIONS:
            raise

    # record additional metadata
    rv["runtime_sec"] = (datetime.now() - start_time).total_seconds()

    df = pl.DataFrame(rv, schema=SCHEMA, strict=False)
    # polars df isn't serializable yet
    return df.to_pandas()


def _run(seed: int, algorithm: str, dataset_type: str, dim: int, config: dict) -> dict:
    """Actually runs the experiment."""

    generate_points = point_generator.GENERATORS[dataset_type]
    if config["metric"] == ("Euclidean", 2):
        metric = Euclid(2)
    else:
        NotImplementedError()
    select_pivots = ALL_ALGORITHMS[algorithm]
    rng = np.random.default_rng(seed)

    points = generate_points(rng=rng, dim=dim, n_samples=config["n_samples"])
    r = proj_quality.get_average_k_nn_dist(points, metric, k=10)
    p0, p1 = select_pivots(points, rng=rng)
    points_p = tetrahedron.project_to_2d_euclidean(points, p0, p1, metric)
    candidate_set_size = proj_quality.candidate_set_size(points_p, r, metric)
    useful_partition_size = proj_quality.hilbert_quality(points_p, r)

    rv = dict()
    rv["candidate_set_size"] = candidate_set_size
    rv["userful_partition_size"] = useful_partition_size
    rv["notes"] = "{}"

    return rv
