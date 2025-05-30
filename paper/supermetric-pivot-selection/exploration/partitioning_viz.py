# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
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

import matplotlib.pyplot as plt
import seaborn as sns

ALL_ALGORITHMS = pivot_selection.get_selection_algos()

# %%
seed = 2290903097003539631 + 33
algorithm = "maximize_dist"
dataset_type = "clusters, sparse"
dim = 2

# %%
algorithm = "IS_pto_1.5"
seed = 352537156968997297
dataset_type = "univariate, idd"
dim = 2

# %%
algorithm = "IS_pto_1.5_greedy"
seed = 352537156968997569
dataset_type = "clusters, sparse"
dim = 2

# %%
"│ IS_pto_1.5_greedy │  │           │ eliptic         │     3 │"

# %%
algorithm = "IS_pto_1.5_greedy"
seed = 352537156968997538
dataset_type = "gaussian, eliptic"
dim = 3

# %%

config = dict(n_samples=512)


generate_points = point_generator.GENERATORS[dataset_type]
metric = Euclid(2)
select_pivots = ALL_ALGORITHMS[algorithm]
rng = np.random.default_rng(seed)

points = generate_points(rng=rng, dim=dim, n_samples=config["n_samples"])
# example queries
queries = generate_points(rng=rng, dim=dim, n_samples=config["n_samples"])
# TODO: use queries instead of points
r = proj_quality.get_average_k_nn_dist(points, metric, k=10)

p0, p1 = select_pivots(points, rng=rng)
proj_conf = dict(p0=p0, p1=p1, dist_func=metric)
points_p = tetrahedron.project_to_2d_euclidean(points, **proj_conf)
queries_p = tetrahedron.project_to_2d_euclidean(queries, **proj_conf)
rv = dict()

rv["candidate_set_size"] = proj_quality.candidate_set_size(
    points_p, queries_p, r, metric
)

part = proj_quality.HilbertPartitioner(points_p, False)
rv["useful_partition_size"] = part.hyperplane_quality(points_p, r)
rv["single_partition_query_share"] = part.is_query_in_one_partition(queries_p, r)

dummy_part = proj_quality.HilbertPartitioner(points_p, dummy_transform=True)
rv["dummy_useful_partition_size"] = dummy_part.hyperplane_quality(points_p, r)
rv["dummy_single_partition_query_share"] = dummy_part.is_query_in_one_partition(
    queries_p, r
)


# %%
settings = dict(legend=None, hue=points[:, 0], s=3)

plt.subplot(2, 2, 1)
plt.title("original space")
sns.scatterplot(x=points[:, 0], y=points[:, 1], **settings)
pivots = np.array([p0, p1])
plt.scatter(pivots[:, 0], pivots[:, 1])

plt.subplot(2, 2, 2)
plt.title("pivot space")
sns.scatterplot(x=points_p[:, 0], y=points_p[:, 1], **settings)

plt.subplot(2, 2, 3)
plt.title("dummy PCA")
sns.swarmplot(x=dummy_part.pca.transform(points_p).flatten(), **settings)
plt.vlines(dummy_part.hyperplane, -0.5, 0.5)

plt.subplot(2, 2, 4)
plt.title("PCA")
sns.swarmplot(x=part.pca.transform(points_p).flatten(), **settings)
plt.vlines(part.hyperplane, -0.5, 0.5)

plt.tight_layout()

# %%
part.pca.transform(points_p).flatten().var()

# %%
dummy_part.pca.transform(points_p).flatten().var()

# %%
