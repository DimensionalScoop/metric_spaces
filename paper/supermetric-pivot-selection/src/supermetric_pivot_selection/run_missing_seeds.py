from run import run
from datetime import datetime
import polars as pl
import duckdb

from missing_seeds import missing_seeds
from meters import pivot_selection
from meters.generate.point_generator import GENERATORS as POINT_GENERATORS

conn = duckdb.connect("results/final-m.duck", read_only=True)
raw = (
    conn.sql("""
    SELECT
      distinct seed, dim, dataset_supertype || ', ' || dataset_subtype as dataset_type
    FROM
      results
""")
    .pl()
    .to_pandas()
)

datasets = list(POINT_GENERATORS.keys())
algs = [
    "fair_max_dist",
    "remote_points_approx",
    "non_central_points_approx",
]  # set(pivot_selection.get_selection_algos(True).keys()) - set(["optimal"])

job_creator = []
for _, row in raw.iterrows():
    for algo in algs:
        job_creator.append((row.dataset_type, row.dim, row.seed, algo))

run(
    job_creator=job_creator,
    name=f"final_missing_seeds_{datetime.now().isoformat().replace(':', '.')}",
    metric=("Euclidean", 2),
    n_samples=512,
    n_queries=128,
    dims=list(range(2, 18)),
    n_cpus=-1,
    n_runs=99999,
    seed=missing_seeds,
    verbose=False,
    dbfile="results/missing_seeds.duck",
    algorithms=list(algs),
)
