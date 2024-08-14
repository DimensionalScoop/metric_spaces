import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm
import time

from tetrahedron import tetrahedron, proj_quality
from metric.metric import Euclid

import pivot_selection
from generate import point_generator


def compare_projections(
    point_gen: dict,
    pivot_selector: dict,
    dims: list,
    seed=0,
    errors="raise",
    verbose=False,
):
    rv = []
    rng = np.random.default_rng(seed)
    point_gen_items = point_gen.items()

    if verbose:
        if verbose != "mp" or seed % 10 == 0:
            point_gen_items = tqdm(
                point_gen_items,
                desc=f"Processing run {run_id} with dimensions {dims}",
                leave=False,
            )

    for dim in dims:
        for gen_name, gen_func in point_gen_items:
            points = gen_func(dim=dim, rng=rng)
            r = proj_quality.get_average_k_nn_dist(points, metric, k=10)
            for algo_name, select_pivots in pivot_selector.items():

                def doit():
                    p0, p1 = select_pivots(points, rng=rng)
                    points_p = tetrahedron.project_to_2d_euclidean(
                        points, p0, p1, metric
                    )
                    rv.append(
                        dict(
                            dim=dim,
                            dataset=gen_name,
                            algorithm=algo_name,
                            mean_candidate_set_size=proj_quality.candidate_set_size(
                                points_p, r, metric
                            ),
                            hilbert_quality=proj_quality.hilbert_quality(points_p, r),
                            note="",
                            seed=seed,
                        )
                    )

                if errors == "skip":
                    try:
                        doit()
                    except:
                        rv.append(
                            dict(
                                dim=dim,
                                dataset=gen_name,
                                algorithm=algo_name,
                                mean_candidate_set_size=-1,
                                hilbert_quality=-1,
                                note="failed",
                                seed=seed,
                            )
                        )
                elif errors == "raise":
                    doit()
                else:
                    raise NotImplementedError()
    rv = pd.DataFrame(rv)
    return rv


# wait to manually change nice levels before starting subprocesses
# time.sleep(10)

metric = Euclid(2)
N_RUNS = 8
N_SAMPLES = 512
DIMS = range(2, 18)
N_CPUS = 20  # 64
SEED_OFFSET = 3710_000
# if you calculated the optimal stuff beforehand, set this to true for massive speedups
SKIP_OPTIMAL_SELECTORS = True

generators = point_generator.get_generator_dict(N_SAMPLES)
piv_selectors = pivot_selection.get_selection_algos(True)

if SKIP_OPTIMAL_SELECTORS:
    del piv_selectors["hilbert_optimal"]
    del piv_selectors["ccs_optimal"]
    del piv_selectors["opt_triangle_IS"]


def run_task(run_id, dim):
    r = compare_projections(
        generators,
        piv_selectors,
        [dim],
        seed=100 * run_id + dim,
        errors="raise",
        verbose=False,
    )
    r["run"] = run_id
    return r


for run_id in range(0, 100_000, N_RUNS):
    print(f"============ run {run_id} to {run_id+N_RUNS} ==============")

    jobs = []
    for subrun in range(N_RUNS):
        for dim in DIMS:
            this_run_id = SEED_OFFSET + run_id + subrun
            jobs.append(delayed(run_task)(this_run_id, dim))

    results = pd.concat(Parallel(n_jobs=N_CPUS, verbose=11)(jobs))

    notes = "-optimal_skipped" if SKIP_OPTIMAL_SELECTORS else ""
    results.to_csv(
        f"results/fast-only/results_{run_id}-to-{run_id+N_RUNS}_{min(DIMS)}-to-{max(DIMS)}-dims_{N_SAMPLES}{notes}.csv"
    )
