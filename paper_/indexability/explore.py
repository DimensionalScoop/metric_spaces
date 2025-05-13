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
from tqdm.auto import tqdm

from metric.metric import Euclid


# %%
def get_datasets():
    """yield datasets"""
    pass


# %%
metrics = [Euclid()]


# %%
def process_single(dataset, metric):
    rv = dict()
    rv["dataset"] = dataset.name
    rv["metric"] = metric.name

    rv["idim_space"] = calc_intrinsic_dim(dataset.data, metric)

    dataset_tet = thetrahedral_project(dataset.data, metric)
    rv["idim_four_point"] = calc_intrinsic_dim(dataset_tet, Euclid())

    N_PIVOTS = int(np.sqrt(len(dataset.data)))
    new_metric = triangle_lb
    projections = project_to_1d_pivot(dataset.data, metric, N_PIVOTS)
    rv["idim_triangle"] = calc_intrinsic_dim_from_lb(projections, new_metric)

    return rv


# %%
results = pd.DataFrame(
    process_single(ds, m) for ds in tqdm(get_datasets()) for m in metrics
)
results
