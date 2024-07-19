import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

import sys

sys.path.append("../../")

import tetrahedron
import proj_quality
from metric.metric import Euclid

import pivot_selection
import point_generator


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
                            )
                        )
                elif errors == "raise":
                    doit()
                else:
                    raise NotImplementedError()
    rv = pd.DataFrame(rv)
    return rv


metric = Euclid(2)
N_RUNS = range(1)
N_SAMPLES = 120
DIMS = range(2, 17)
N_CPUS = 10

generators = point_generator.get_generator_dict(N_SAMPLES)
piv_selectors = pivot_selection.get_selection_algos(True)


def run(run_id, dim):
    r = compare_projections(
        generators,
        piv_selectors,
        [dim],
        seed=100 * run_id + dim,
        errors="raise",
        verbose=True,
    )
    r["run"] = run_id
    return r


jobs = []
for run_id in N_RUNS:
    for dim in DIMS:
        jobs.append(delayed(run)(run_id, dim))

results = pd.concat(Parallel(n_jobs=N_CPUS, verbose=11)(jobs))
results.to_csv("./results.csv")
