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
from numba import njit
from tqdm import tqdm
import time
from itertools import permutations, combinations

import sys
sys.path.append("../..")

from tetrahedron import tetrahedron, proj_quality
from metric.metric import Euclid

import pivot_selection
from generate import point_generator

import matplotlib.pyplot as plt
import seaborn as sns

# %%
dims = 10
n_points = 1000
n_pivs = 20
budget = 2*n_points
metric = pivot_selection.lb_summation.METRIC

rng = np.random.default_rng(0xABACADABA)

data = point_generator.generate_univar_points(rng, n_points, dim=dims)
tri_pivots = pivot_selection.lb_summation.IS(data, n_pivs, budget, rng, "tri")
pto_pivots = pivot_selection.lb_summation.IS(data, n_pivs, budget, rng, "pto")

# %%
triu = np.triu_indices(n_points,1)
actual_dists = metric.distance_matrix(data,data)[triu]


# %%
def get_best_tri_lbs(pivots, data):
    def tri_lb_all_pairs_dist(pivots, data):
        for p in pivots:
            d_c = metric(p, data)
            lbs_tri = np.abs(d_c.reshape(-1,1) - d_c.reshape(1,-1))
            yield lbs_tri[triu]
    
    all_lbs = np.stack(list(tri_lb_all_pairs_dist(pivots, data)))
    best_lbs = np.amax(all_lbs, axis=0)
    return best_lbs


best_tri_tri_lbs = get_best_tri_lbs(tri_pivots, data)
best_pto_tri_lbs = get_best_tri_lbs(pto_pivots, data)


# %%
@njit
def pto_all_pairs(d1, d2, piv_piv_dist):
    n = len(d1)
    dist_matrix = np.zeros((n,n))
    
    for q in range(n):
        for x in range(n):
            dist_matrix[q,x] = d1[q]*d2[x] - d1[x]*d2[q]
    
    dist_matrix = 1/piv_piv_dist * np.abs(dist_matrix)
    return dist_matrix
            

def get_best_pto_lbs(pivots, data):
    def lb_all_pairs_dist(pivots, data):
        for p1,p2 in combinations(pivots,2):
            d1 = metric(p1, data)
            d2 = metric(p2, data)
            quotient = metric(p1, p2) 
            lbs = pto_all_pairs(d1, d2, quotient)
            yield lbs[triu]
    
    all_lbs = np.stack(list(lb_all_pairs_dist(pivots, data)))
    best_lbs = np.amax(all_lbs, axis=0)
    return best_lbs


best_tri_pto_lbs = get_best_pto_lbs(tri_pivots, data)
best_pto_pto_lbs = get_best_pto_lbs(pto_pivots, data)

# %%
#distances_from_pivot
# cached distances

_, bins, _ = plt.hist(actual_dists, bins=100, label="exact distances")
plt.hist(best_tri_tri_lbs, histtype="step", bins=bins, label="tri IS, tri access");
plt.hist(best_pto_tri_lbs, histtype="step", bins=bins, label="pto IS, tri access");
plt.hist(best_tri_pto_lbs, histtype="step", bins=bins, label="tri IS, pto access");
plt.hist(best_pto_pto_lbs, histtype="step", bins=bins, label="pto IS, pto access");

plt.legend()

# %%
