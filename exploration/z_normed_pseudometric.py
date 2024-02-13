# %%
"""Does the triangle inequality hold for the z-normed Euclidean distance
function? Yes, and the Ptolemaic ineq.!

This notebook shows that both hold for random points.
Additionally, distance histograms show how different lower bounds perform in
different dimensions.
"""
# %%
# %load_ext autoreload
# %autoreload 2
# %%
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict 
from itertools import combinations
from dataclasses import dataclass
import numpy.testing as npt
from numba import jit
import numba

import sys

sys.path.append("/home/elayn/epix/metric_spaces/")

from numerical.metric_test import MetricTest, NoVarianceError

# %%


# %%
def corr(x, y, ddof=1):
    var_x, cov, cov, var_y = np.cov(x, y, ddof=ddof).flatten()
    if var_x == 0 or var_y == 0:
        raise NoVarianceError("variable has no variance!")

    cov = cov / np.sqrt(var_x * var_y)
    return np.clip(cov, -1, 1)  # pearsonr(x,y).statistic


def cos_theta(x, y):
    x_n, y_n = np.linalg.norm(x), np.linalg.norm(y)
    return np.sum(x * y, axis=0) / x_n / y_n


def d(x, y):
    return np.sqrt(2 * (1 - corr(x, y)))


# %%
DIMS = 10
SAMPLES = 10000

dists = defaultdict(list)

for _ in tqdm(range(SAMPLES)):
    vecs = np.random.rand(2, DIMS) * 37 - 18.5

    try:
        dists["raw"].append(d(*vecs))
        dists["z"].append(d(*zscore(vecs, axis=1)))

        vecs_no_mean = vecs - vecs.sum(axis=0)
        dists["no_mean"].append(d(*vecs_no_mean))
    except NoVarianceError:
        pass

for k, v in dists.items():
    plt.hist(v, density=True, histtype="step", label=k, bins=100)
plt.legend()
plt.xlabel(f"$d_z$ distance between uniformly random {DIMS}-d points")
plt.ylabel("frequency")

# %%
differences = np.asarray(dists["raw"]) - np.asarray(dists["z"])
npt.assert_almost_equal(differences, 0)

differences = np.asarray(dists["raw"]) - np.asarray(dists["no_mean"])
npt.assert_almost_equal(differences, 0)

# %%
DIMS = 15


def generate_discrete_vecs(*number, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(-20, 20, size=[*number, DIMS])


def generate_float_vecs(*number, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.random(size=[*number, DIMS]) * 37 - 18.5


test = MetricTest(d, generate_float_vecs)
res = test.run_test(3000)
res


# %%
@dataclass
class MultivariateTimeSeries:
    discrete = True

    # number of sampels in a series
    WINDOW_LEN: int
    # number of time series
    N_SERIES: int

    def generate(self, *number, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        shape = (*number, self.N_SERIES, self.WINDOW_LEN)
        if self.discrete:
            return rng.integers(-20, 21, size=shape)
        else:
            return rng.random(size=shape) * 40 - 20

    def __call__(self, *number, rng=None):
        return self.generate(*number, rng=rng)


mts_generator = MultivariateTimeSeries(5, 4)
mts_generator.discrete = True

# %% 
corr_signature = numba.float64(numba.float64[:], numba.float64[:])
dist_signature = numba.float64(numba.float64[:], numba.float64[:])
multivar_dist_signature = numba.float64(numba.float64[:,:], numba.float64[:,:])

@jit(corr_signature, nopython=True)
def corr(x, y):
    ddof=1
    var_x, cov, cov, var_y = np.cov(x, y, ddof=ddof).flatten()
    if var_x == 0 or var_y == 0:
        raise NoVarianceError("variable has no variance!")

    cov = cov / np.sqrt(var_x * var_y)
    # numba doesn't like scalar use of np.clip
    if cov < -1:
        cov = -1
    elif cov > 1:
        cov = 1
    return cov # pearsonr(x,y).statistic


@jit(dist_signature, nopython=True)
def d(x, y):
    return np.sqrt(2 * (1 - corr(x, y)))

@jit(multivar_dist_signature, nopython=True)
def mvts_distance(x, y):
    n_series, _ = x.shape
    dist = [d(x[s], y[s]) ** 2 for s in range(n_series)]
    dist = np.asarray(dist)
    return np.sqrt(np.sum(dist))

# %%
def mvts_distance_chebichev(x, y):
    n_series, _ = x.shape
    dist = [d(x[s], y[s]) for s in range(n_series)]
    return np.max(dist)


def mvts_distance_taxi(x, y):
    n_series, _ = x.shape
    dist = [d(x[s], y[s]) for s in range(n_series)]
    return np.sum(dist)

# %% 

vecs = [(*mts_generator(4)) for _ in tqdm(range(10000))]
plt.hist(samples_dists, bins=100)

# %%
print("---Discrete Test---")
mts_generator = MultivariateTimeSeries(5, 4)
mts_generator.discrete = True

test = MetricTest(mvts_distance, mts_generator, atol=1e-8)
samples_per_min = 760_000 * 2 // 3
long_run = samples_per_min * 0.01
n = samples_per_min * 15
res = test.run_test(int(n))

def print_results(res):
    print("summary:")
    print(np.unique([msg for msg, _ in res], return_counts=True))
    print("---all events:")
    [print(m) for m in res]
    print("---")
print_results(res)

# %%
print("---Continous Test---")
mts_generator = MultivariateTimeSeries(5, 4)
mts_generator.discrete = False

test = MetricTest(mvts_distance, mts_generator, atol=1e-8)
res = test.run_test(int(n))

print_results(res)

# %%
# === Histogram comparisons

LOWER_BOUND_NAME = ("exact", "triangle", "two triangles", "Ptolemy")
sig = numba.float64[:](numba.float64[:,:], numba.float64[:,:], numba.float64[:,:], numba.float64[:,:])
@jit(sig,nopython=True)
def lower_bounds(p1, p2, q, o):
    d = mvts_distance

    exact = d(o,q)
    p1_lb = np.abs(d(p1, o) - d(p1, q))
    p2_lb = np.abs(d(p2, o) - d(p2, q))
    single_triangle = p1_lb
    two_triangle = max(p1_lb, p2_lb)

    numerator = d(p1, o) * d(p2, q) - d(p1, q) * d(p2, o)
    quotient = d(p1, p2)
    ptolemy = np.abs(numerator) / quotient

    return np.array((exact, single_triangle, two_triangle, ptolemy))


@jit(nopython=True)
def get_all_lower_bounds(sample_list):
    # numba doesn't like to take this more pythonic :c
    lbs = np.empty((len(sample_list), 4), dtype=float)
    for i, (p1,p2,q,o) in enumerate(sample_list):
        lbs[i,:] = lower_bounds(p1,p2,q,o)
    lbs = lbs.T
    return lbs

# %%

def _get_all_lower_bounds_for_all_pivot_choices(sample_list):
    QUERY_IDX = 0
    OBJ_IDX = 1

    n_samples, vecs_per_sample, *_ = sample_list.shape
    n_pivots = vecs_per_sample - 2
    assert n_pivots >= 2

    pivot_index = range(2, vecs_per_sample)
    pivot_combinations = list(combinations(pivot_index, 2))

    lbs = dict()
    for p1_idx, p2_idx in tqdm(pivot_combinations):
        points = (p1_idx, p2_idx, QUERY_IDX, OBJ_IDX)
        two_pivot_sample_list = sample_list[:,points]
        name = (p1_idx, p2_idx)
        lbs[name] = get_all_lower_bounds(two_pivot_sample_list)
    return lbs

def get_all_best_lower_bounds_multi_pivot(sample_list):
    lb_all_choices = _get_all_lower_bounds_for_all_pivot_choices(sample_list)
    lb_all_choices = np.asarray(list(lb_all_choices.values()))
    # expected shape:
    pivot_choice, lower_bound_type, sample = lb_all_choices.shape
    # choose the pivot combination that has the best lb available
    return lb_all_choices.max(axis=0)
    

mts_generator = MultivariateTimeSeries(5, 4)
mts_generator.discrete = False 
n_pivots = 10
samples_vecs = mts_generator(10000, 2 + n_pivots) 
print(samples_vecs.shape)

lbs = get_all_best_lower_bounds_multi_pivot(samples_vecs)
lbs = dict(zip(LOWER_BOUND_NAME, lbs))
# %% 

def plot_hist(lbs, mts_generator, n_pivots):
    shape = f"$ {mts_generator.N_SERIES} \\times {mts_generator.WINDOW_LEN} $"
    plt.xlabel(f"$d_m$ distance between two random {shape} time series\nusing {n_pivots} pivots") 
    plt.ylabel("density")

    exact_dists = lbs["exact"]
    hist_conf = dict(
            bins=np.linspace(0,max(lbs["exact"]), 50),
            density=True
            )
    plt.hist(exact_dists, label="exact distances", **hist_conf)

    for name, lb_samples in lbs.items():
        if name == "exact":
            continue
        plt.hist(lb_samples, histtype="step", label=name, **hist_conf)

    plt.legend()
    plt.grid()
    plt.tight_layout()

plot_hist(lbs, mts_generator, n_pivots)
plt.savefig("/fig/small_hist.png", dpi=300)
plt.show()

# %%

samples = int(1e3)
n_pivots = 15*4
mts_generator = MultivariateTimeSeries(15, 4)
mts_generator.discrete = False 

samples_vecs = mts_generator(samples, 2 + n_pivots) 

lbs = get_all_best_lower_bounds_multi_pivot(samples_vecs)
lbs = dict(zip(LOWER_BOUND_NAME, lbs))
lbs.pop("two triangles")

# %%
plot_hist(lbs, mts_generator, n_pivots)
plt.savefig("/fig/large_hist.png", dpi=300)
plt.show()



# %%
SAMPLES = 50000
mts_generator = MultivariateTimeSeries(4, 2)
mts_generator.discrete = False 

samples_vecs = mts_generator(SAMPLES,4) 

lbs = get_all_lower_bounds(samples_vecs, mvts_distance)

# %%
plot_hist(lbs, mts_generator)
plt.savefig("/fig/small_hist.svg")
plt.show()

# %%
lbs.keys()
# %%
plot_hist(lbs, mts_generator)
plt.savefig("/fig/small_hist.png" )
plt.show()


# %%
SAMPLES = 1000
NUM_PIVS = 20
mts_generator = MultivariateTimeSeries(5, 4)
mts_generator.discrete = False 

samples_vecs = mts_generator(SAMPLES,2+NUM_PIVS) 

lbs = get_all_lower_bounds_multi_pivot(samples_vecs, mvts_distance)

plot_hist(lbs, mts_generator)
plt.savefig("/fig/large_hist.svg")
plt.show()


# %% 
print("overlaps")
for k, v in lb.items():
    plt.hist(v, density=True, histtype="step", label="lb_" + k, bins=50)
    print(k, histogram_overlap(v, exact_dists))

everything = np.asarray([v for _, v in lb.items()])
best = everything.max(axis=0)
plt.hist(best, density=True, histtype="step", label="best_lb", bins=50)
print("max_lb", histogram_overlap(best, exact_dists))
