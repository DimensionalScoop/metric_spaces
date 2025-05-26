import numpy as np
import pandas as pd
import polars as pl
import duckdb
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
import platform
import psutil

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

print("starting experiments with this config:")
pprint(CONFIG)


# def run_task(run_id, dim):
#     r = compare_projections(
#         generators,
#         piv_selectors,
#         [dim],
#         seed=100 * run_id + dim,
#         errors="skip",
#         verbose=False,
#     )
#     r["run"] = run_id
#     return r


# for run_id in range(0, 2000, N_RUNS):
#     print(f"============ run {run_id} to {run_id + N_RUNS} ==============")

#     jobs = []
#     for subrun in range(N_RUNS):
#         for dim in DIMS:
#             this_run_id = SEED_OFFSET + run_id + subrun
#             jobs.append(delayed(run_task)(this_run_id, dim))

#     results = pd.concat(Parallel(n_jobs=N_CPUS, verbose=1)(jobs))

#     notes = "-optimal_skipped" if SKIP_OPTIMAL_SELECTORS else ""
#     results.to_csv(
#         f"{PATH}fast-only/results_{run_id}-to-{run_id + N_RUNS}_{min(DIMS)}-to-{max(DIMS)}-dims_{N_SAMPLES}{notes}.csv"
#     )


# def compare_projections(
#     point_gen: dict,
#     pivot_selector: dict,
#     dims: list,
#     seed=0,
#     errors="raise",
#     verbose=False,
# ):
#     rv = []
#     rng = np.random.default_rng(seed)
#     point_gen_items = point_gen.items()

#     if verbose:
#         if verbose != "mp" or seed % 10 == 0:
#             point_gen_items = tqdm(
#                 point_gen_items,
#                 desc=f"Processing run {run_id} with dimensions {dims}",
#                 leave=False,
#             )

#     for dim in dims:
#         for gen_name, gen_func in point_gen_items:
#             points = gen_func(dim=dim, rng=rng)
#             r = proj_quality.get_average_k_nn_dist(points, metric, k=10)
#             for algo_name, select_pivots in pivot_selector.items():

#                 def doit():
#                     p0, p1 = select_pivots(points, rng=rng)
#                     points_p = tetrahedron.project_to_2d_euclidean(
#                         points, p0, p1, metric
#                     )
#                     rv.append(
#                         dict(
#                             dim=dim,
#                             dataset=gen_name,
#                             algorithm=algo_name,
#                             mean_candidate_set_size=proj_quality.candidate_set_size(
#                                 points_p, r, metric
#                             ),
#                             hilbert_quality=proj_quality.hilbert_quality(points_p, r),
#                             note="",
#                             seed=seed,
#                         )
#                     )

#                 if errors == "skip":
#                     try:
#                         doit()
#                     except:
#                         print(f"Skipped error at {dim} {gen_name} {algo_name}.")
#                         rv.append(
#                             dict(
#                                 dim=dim,
#                                 dataset=gen_name,
#                                 algorithm=algo_name,
#                                 mean_candidate_set_size=-1,
#                                 hilbert_quality=-1,
#                                 note="failed",
#                                 seed=seed,
#                             )
#                         )
#                 elif errors == "raise":
#                     doit()
#                 else:
#                     raise NotImplementedError()
#     rv = pd.DataFrame(rv)
#     return rv
