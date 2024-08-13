# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt

import sys

sys.path.append("../..")

from tetrahedron import tetrahedron, proj_quality
from metric.metric import Euclid

import pivot_selection
from generate import point_generator

# %%
metric = Euclid(2)
N_SAMPLES = 512
DIM = 2
SEED = 0xFEED

GENERATOR = "gaussian, eliptic"

generators = point_generator.get_generator_dict(N_SAMPLES)
piv_selectors = pivot_selection.get_selection_algos(True)

rng = np.random.default_rng(SEED)
gen_func = generators[GENERATOR]
points = gen_func(dim=DIM, rng=rng)
r = proj_quality.get_average_k_nn_dist(points, metric, k=10)

# %%
rv = []

for algo_name, select_pivots in tqdm(piv_selectors.items()):
    try:
        p0, p1 = select_pivots(points, rng=rng)

        points_p = tetrahedron.project_to_2d_euclidean(points, p0, p1, metric)
        rv.append(
            dict(
                algorithm=algo_name,
                mean_candidate_set_size=proj_quality.candidate_set_size(
                    points_p, r, metric
                ),
                hilbert_quality=proj_quality.hilbert_quality(points_p, r),
                pivots = np.vstack((p0,p1))
            )
        )

    except ValueError:
        pass



# %%
df = pd.DataFrame(rv).sort_values("hilbert_quality").set_index("algorithm")
display(df)

# %%
plt.scatter(*points.T, marker="+")
for algo_name, piv in df.pivots.items():
    plt.plot(*piv.T, label=algo_name)
plt.legend()

# %%
