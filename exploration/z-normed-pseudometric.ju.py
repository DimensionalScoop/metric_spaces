# %%
"""Does the triangle inequality hold for the z-normed Euclidean distance
function? Yes, and the Ptolemaic ineq.!

This notebook shows that both hold for random points.
Additionally, distance histograms show how different lower bounds perform in
different dimensions.
"""
# %%
%load_ext autoreload
%autoreload 2
# %%
import numpy as np
from scipy.stats import pearsonr, zscore
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
import numpy.testing as npt

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
plt.show()

# %%
differences = np.asarray(dists["raw"]) - np.asarray(dists["z"])
npt.assert_almost_equal(differences, 0)

differences = np.asarray(dists["raw"]) - np.asarray(dists["no_mean"])
npt.assert_almost_equal(differences, 0)

# %%
DIMS = 15

def generate_discrete_vecs(*number):
    return np.random.randint(-20, 20, size=[*number, DIMS])

def generate_float_vecs(*number):
    return np.random.rand(*number, DIMS) * 37 - 18.5

test = MetricTest(d, generate_float_vecs)
res = test.run_test(3000)
res

# %%
@dataclass
class MultivariateTimeSeries:
    discrete = True

    # number of sampels in a series
    WINDOW_LEN:int
    # number of time series
    N_SERIES:int

    def generate(self, *number):
        shape = (*number, self.N_SERIES, self.WINDOW_LEN)
        if self.discrete:
            return np.random.randint(-20, 20, size=shape)
        else:
            return np.random.rand(*shape) * 37 - 18.5

    def __call__(self, *number):
        return self.generate(*number)

mts_generator = MultivariateTimeSeries(5, 3)
mts_generator.discrete = False

def mvts_distance(x,y):
    n_series, _ = x.shape
    dist = [d(x[s], y[s])**2 for s in range(n_series)]
    return np.sqrt(np.sum(dist))

def mvts_distance_chebichev(x,y):
    n_series, _ = x.shape
    dist = [d(x[s], y[s]) for s in range(n_series)]
    return np.max(dist)

def mvts_distance_taxi(x,y):
    n_series, _ = x.shape
    dist = [d(x[s], y[s]) for s in range(n_series)]
    return np.sum(dist)

samples_dists = [mvts_distance_taxi(*mts_generator(2)) for _ in tqdm(range(10000))]
plt.hist(samples_dists, bins=100)
plt.show()

# %%
test = MetricTest(mvts_distance_chebichev, mts_generator)
res = test.run_test(300000)
res

# %%



def histogram_overlap(data_a, data_b, bins=50):
    both = np.hstack((data_a, data_b))
    bins = np.histogram_bin_edges(both, bins=bins)
    a, _ = np.histogram(data_a, bins)
    b, _ = np.histogram(data_b, bins)
    overlap_absolute = np.vstack((a, b)).min(axis=0).sum()
    bin_width = bins[1] - bins[0]
    return overlap_absolute / len(data_a)


# %%

# here be dragons

lb = defaultdict(list)
for _ in tqdm(range(SAMPLES)):
    vecs = np.random.rand(4, DIMS) * 37 - 18.5
    x, y, z, _ = vecs
    lb["triangle"].append(np.abs(d(x, y) - d(z, y)))

    p1, p2, q, o = vecs
    numerator = d(p1, o) * d(p2, q) - d(p1, q) * d(p2, o)
    quotient = d(p1, p2)
    lb["ptolemy"].append(np.abs(numerator) / quotient)

    p1, p2, q, o = vecs
    p1_lb = np.abs(d(p1, o) - d(p1, q))
    p2_lb = np.abs(d(p2, o) - d(p2, q))
    lb["two_triangle"].append(max(p1_lb, p2_lb))

# %%
exact_dists = dists["raw"]
plt.hist(exact_dists, density=True, label="exact_distances", bins=50)

print("overlaps")
for k, v in lb.items():
    plt.hist(v, density=True, histtype="step", label="lb_" + k, bins=50)
    print(k, histogram_overlap(v, exact_dists))

# best of all
everything = np.asarray([v for _, v in lb.items()])
best = everything.max(axis=0)
plt.hist(best, density=True, histtype="step", label="best_lb", bins=50)
print("max_lb", histogram_overlap(best, exact_dists))

plt.legend()
plt.xlabel(f"$d_z$ distance between uniformly random {DIMS}-d points")
plt.ylabel("frequency")
plt.show()

# %%
# using four pivots:
# we can do that because points are uniformly random
pto_4_piv = np.array(lb["ptolemy"]).reshape(4, -1)
pto_4_piv = pto_4_piv.max(axis=0)
plt.hist(best, density=True, histtype="step", label="best_lb", bins=50)
plt.hist(pto_4_piv, density=True, histtype="step", label="pto_4_piv", bins=50)

# %%
dists = []
for _ in tqdm(range(100000)):
    vecs = np.random.randint(-20, 20, size=[2, 5])
    # vecs = zscore(vecs,axis=1)
    dists.append(d(*vecs))

plt.hist(dists, bins=100)

# %%
dists = []
# %%
for _ in tqdm(range(100000)):
    vecs = np.random.randint(-20, 20, size=[4, 5])
    # vecs = vecs - vecs.mean(axis=0)
    x, y, z, k = vecs
    if triangle_violation(x, y, z) > 1e-3:
        print(x, y, z)
    if pto_violation(x, y, z, k) > 1e-3:
        print("pto violation", x, y, z)


# %%
def cos_dist_triangle_inequality(x, y, z):
    def d_t(a, b):
        return np.sqrt(2 - 2 * cos_theta(a, b))

    lhs = d_t(x, y)
    rhs = d_t(x, z) + d_t(z, y)
    # lhs <= rhs
    violation = lhs - rhs
    return violation


side_diffs = []
for _ in tqdm(range(100000)):
    vecs = np.random.randint(-20, 20, size=[4, 3])
    vecs = vecs / np.linalg.norm(vecs, axis=0)  # - vecs.mean(axis=0)
    x, y, z, k = vecs
    diff = cos_dist_triangle_inequality(x, y, z)
    side_diffs.append(diff)
    if diff > 1e-3:
        print(diff, x, y, z)
        break


plt.hist(side_diffs)


# %%
def d(corr, m=10):
    return np.sqrt(2 * m * (1 - corr))


def is_triangle_inequal(x_var, y_var, z_var, xy_cov, xz_cov, yz_cov):
    xy_corr = xy_cov / np.sqrt(x_var * y_var)
    xz_corr = xz_cov / np.sqrt(x_var * z_var)
    yz_corr = yz_cov / np.sqrt(z_var * y_var)
    return d(xy_cov) <= d(xz_cov) + d(yz_cov)


# %%
for _ in tqdm(range(10000)):
    vars = np.random.rand(3) * 100
    covars = (np.random.rand(3) - 0.5) * 100
    if not is_triangle_inequal(*vars, *covars):
        print("violation")
        break

x, y = np.random.randint(-20, 20, size=[2, 5])
x, y = x - np.mean(x), y - np.mean(y)

corr(x, y), cos_theta(x, y)

# %%
np.linalg.norm(x)

# %%
while True:
    x, y = np.random.randint(-20, 20, size=[2, 4])
    if np.abs(corr(x, y)) < 1e-5:
        break

# %%
x, y, corr(x, y)

# %%
zscore(x), zscore(y)

# %%
d(x, y)

# %%
