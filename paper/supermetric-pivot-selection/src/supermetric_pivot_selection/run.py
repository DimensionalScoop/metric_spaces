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
from frozendict import frozendict
import logging

import experiment
from meters import pivot_selection
from meters.generate.point_generator import GENERATORS as POINT_GENERATORS
from meters.metric.metric import Euclid
from meters.tetrahedron import proj_quality, tetrahedron


PATH = f"results/experiment_{datetime.now().isoformat()}"
DB_FILE = PATH + ".duck"
LOG_FILE = PATH + ".log"

logging.basicConfig(
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


EXPERIMENT_ID = uuid4()

CONFIG = dict(
    metric=("Euclidean", 2),
    n_runs=20,
    n_samples=512,
    dims=list(range(2, 18)),
    n_cpus=-1,
    seed=hash(datetime.now()),
)

CONFIG["machine_os"] = platform.system() + " " + platform.release()
CONFIG["machine_mem_GB"] = int(psutil.virtual_memory().total / 1e9)
CONFIG["machine_cores"] = psutil.cpu_count()

CONFIG["file"] = PATH
db = duckdb.connect(PATH)

generators = POINT_GENERATORS
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
db.execute(
    "INSERT INTO experiments (id, config) VALUES (?, ?)",
    [EXPERIMENT_ID, json.dumps(CONFIG)],
)
# only add it now, so the id isn't stored twice in the DB
CONFIG["experiment_id"] = EXPERIMENT_ID

print("starting experiments with this config:")
pprint(CONFIG)
logging.info(CONFIG)


# plan all experiments
def create_jobs():
    seed = itertools.count(CONFIG["seed"])
    config = frozendict(CONFIG)
    for _ in range(CONFIG["n_runs"]):
        for algorithm in CONFIG["algorithms"]:
            for dataset_type in CONFIG["datasets"]:
                for dim in CONFIG["dims"]:
                    params = (next(seed), algorithm, dataset_type, dim, config)
                    yield delayed(experiment.run)(*params)


jobs = list(create_jobs())

# actually run them
with parallel_config(backend="loky", inner_max_num_threads=2):
    runner = Parallel(n_jobs=-1, return_as="generator_unordered", verbose=3)
    results = runner(jobs)
    batch = []
    timer = datetime.now()

    def _save_batch():
        print("saving batch...", end="")
        global timer, batch, db
        batch_df = pl.concat(batch).to_pandas()
        try:
            db.execute("INSERT INTO results SELECT * FROM batch_df")
        except duckdb.CatalogException:
            db.execute("CREATE TABLE results AS SELECT * FROM batch_df")
        db.execute("CHECKPOINT")  # flush changes to disk
        batch = []
        timer = datetime.now()
        print(" done!")

    for df in tqdm(results, total=len(jobs)):
        batch.append(pl.DataFrame(df))
        if datetime.now() - timer > timedelta(seconds=5):
            _save_batch()
    _save_batch()
