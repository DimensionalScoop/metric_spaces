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


def run(seed: int, algorithm: str, dataset_type: str, dim: int) -> pl.DataFrame:
    """Conducts a single experiment with the given parameters"""
    rng = np.random.default_rng(seed)
    time.sleep(rng.random() * 0.1)
    return pl.DataFrame({"a": [1, 2]})
