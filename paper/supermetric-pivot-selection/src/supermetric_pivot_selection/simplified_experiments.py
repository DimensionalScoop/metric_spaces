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


metric = Euclid(2)

CONFIG = dict(
    metric=("Euclidean", 2),
    n_runs=8,
    n_samples=512,
    dims=list(range(2, 18)),
    n_cpus=-1,
    seed=0xDEAD,
)

CONFIG["machine_os"] = platform.system() + " " + platform.release()
CONFIG["machine_mem_GB"] = int(psutil.virtual_memory().total / 1e9)
CONFIG["machine_cores"] = psutil.cpu_count()

PATH = f"results/experiment_{datetime.now().isoformat()}.duck"
CONFIG["file"] = PATH
db = duckdb.connect(PATH)

generators = point_generator.get_generator_dict(CONFIG["n_samples"])
piv_selectors = pivot_selection.get_selection_algos(True)

CONFIG["datasets"] = list(generators.keys())
CONFIG["algorithms"] = list(piv_selectors.keys())

db.sql("""
    CREATE SEQUENCE id_sequence START 1;
    CREATE TABLE IF NOT EXISTS experiments (
        id UUID PRIMARY KEY DEFAULT uuid(),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config json
        )
""")
db.execute("INSERT INTO experiments (config) VALUES (?)", [json.dumps(CONFIG)])

print("starting experiments with this config:")
pprint(CONFIG)


def run_single_experiment(seed, algorithm, dataset_type, dim) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    time.sleep(rng.random() * 0.1)
    return pl.DataFrame({"a": [1, 2]})


# plan all experiments
def create_jobs():
    seed = itertools.count(CONFIG["seed"])
    for algorithm in CONFIG["algorithms"]:
        for dataset_type in CONFIG["datasets"]:
            for dim in CONFIG["dims"]:
                params = (next(seed), algorithm, dataset_type, dim)
                yield delayed(run_single_experiment)(*params)


jobs = list(create_jobs())

# actually run them
with parallel_config(backend="loky", inner_max_num_threads=2):
    runner = Parallel(n_jobs=-1, backend="loky", return_as="generator_unordered")
    results = runner(jobs)
    batch = []
    timer = datetime.now()

    def _save_batch():
        global timer, batch, db
        batch_df = pl.concat(batch)
        try:
            db.execute("INSERT INTO results SELECT * FROM batch_df")
        except duckdb.CatalogException:
            db.execute("CREATE TABLE results AS SELECT * FROM batch_df")
        batch = []
        timer = datetime.now()
        print("batch saved")

    for df in tqdm(results, total=len(jobs)):
        batch.append(df)
        if timer - datetime.now() > timedelta(seconds=5):
            _save_batch()
    _save_batch()
