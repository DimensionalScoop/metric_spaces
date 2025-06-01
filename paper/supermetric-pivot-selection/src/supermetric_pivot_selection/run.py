from uuid import uuid4
import pandas as pd
import duckdb
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm
from datetime import datetime, timedelta
from pprint import pprint
import platform
import psutil
import json
import itertools
from frozendict import frozendict
import logging

import experiment
from meters import pivot_selection
from meters.generate.point_generator import GENERATORS as POINT_GENERATORS


def run(**config_overwrite):
    if config_overwrite is None:
        config_overwrite = dict()

    PATH = "results/"

    EXPERIMENT_ID = uuid4()

    CONFIG = dict(
        name=f"e_{datetime.now().isoformat().replace(':', '.')}",
        metric=("Euclidean", 2),
        n_runs=30,
        n_samples=512,
        n_queries=128,
        # n_samples=128,
        # n_queries=20,
        dims=list(range(2, 18)),
        n_cpus=-1,
        seed=abs(hash(datetime.now())),
        verbose=False,
    )

    CONFIG["machine_os"] = platform.system() + " " + platform.release()
    CONFIG["machine_mem_GB"] = int(psutil.virtual_memory().total / 1e9)
    CONFIG["machine_cores"] = psutil.cpu_count()

    CONFIG["files"] = PATH + CONFIG["name"]
    CONFIG["dbfile"] = PATH + CONFIG["name"] + ".duck"
    CONFIG["logfile"] = PATH + CONFIG["name"] + ".log"

    generators = POINT_GENERATORS
    piv_selectors = pivot_selection.get_selection_algos(True)
    CONFIG["datasets"] = list(generators.keys())
    CONFIG["algorithms"] = list(piv_selectors.keys())
    CONFIG.update(config_overwrite)

    logging.basicConfig(
        handlers=[logging.FileHandler(CONFIG["logfile"]), logging.StreamHandler()],
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    db = duckdb.connect(CONFIG["dbfile"])
    try:
        db.sql("""
            CREATE SEQUENCE id_sequence START 1;
            CREATE TABLE IF NOT EXISTS experiments (
                id UUID PRIMARY KEY DEFAULT uuid(),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config json
                )
        """)
    except duckdb.CatalogException:
        logger.info("using existing db scheme")
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
        try:
            seed = iter(CONFIG["seed"])
        except TypeError:
            seed = itertools.count(CONFIG["seed"])
        config = frozendict(CONFIG)

        try:
            for _ in range(CONFIG["n_runs"]):
                for dataset_type in CONFIG["datasets"]:
                    for dim in CONFIG["dims"]:
                        this_seed = next(seed)  # use the same seed for all algorithms
                        for algorithm in CONFIG["algorithms"]:
                            params = (this_seed, algorithm, dataset_type, dim, config)
                            yield delayed(experiment.run)(*params)
        except StopIteration:
            pass  # seed list ran out; that's okay!

    jobs = list(create_jobs())

    # actually run them
    with parallel_config(backend="loky", inner_max_num_threads=2):
        runner = Parallel(n_jobs=CONFIG["n_cpus"], return_as="generator_unordered")
        results = runner(jobs)
        batch = []
        timer = datetime.now()

        def _save_batch(timer, batch):
            logging.info("saving batch...")
            if len(batch) == 0:
                logging.info("nothing to save.")
            try:
                # weird things happen trying to put a polars df
                # into the duckdb database, so stick with pandas for now
                batch_df = pd.concat(batch)  # noqa: F841
                try:
                    db.execute("INSERT INTO results SELECT * FROM batch_df")
                except duckdb.CatalogException:
                    db.execute("CREATE TABLE results AS SELECT * FROM batch_df")
            except:
                logging.exception("while saving batch to db")

            db.execute("CHECKPOINT")  # flush changes to disk
            logging.info("done saving batch!")
            batch = []
            timer = datetime.now()
            return timer, batch

        try:
            for df in tqdm(results, total=len(jobs)):
                batch.append(df)
                if datetime.now() - timer > timedelta(seconds=5):
                    timer, batch = _save_batch(timer, batch)
        except KeyboardInterrupt:
            logging.warning("user keyboard interrupt")
        finally:
            _save_batch(timer, batch)


if __name__ == "__main__":
    run()
