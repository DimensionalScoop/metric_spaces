import numpy as np
from scipy import spatial
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import jit
from scipy import optimize
from scipy import stats

from measure import Measure


def dist_func(x, y):
    return spatial.minkowski_distance(x, y, 2)


rng = np.random.default_rng()


def generate_points(dim, n_samples):
    # return rng.integers(-37, 38, size=[n_samples, dim])
    # return rng.random(size=[n_samples, dim]) * (37 * 2) - 37
    return rng.normal(3, 10, size=[n_samples, dim])


def choose_pair(points):
    return rng.choice(points, size=2, replace=False)


def estimate_dist(points, sampels):
    """Uses `samples` distance evaluation to find the distance distribution's mean and standard deviation
    Also returns the most-separated pair
    """
    pairs = [choose_pair(points) for _ in range(sampels)]
    dists = [dist_func(a, b) for a, b in pairs]
    return np.mean(dists), np.std(dists), pairs[np.argmax(dists)]


def predict_farthes_pair(points, dist_func, discard_threshold):
    cycle_count = 0
    best_pairs = []
    best_dists = []

    points = list(points)
    while len(points) > 1:
        center = points.pop()
        cycle_count += len(points)
        ds = dist_func(center, points)
        best_point_idx = np.argmax(ds)
        best_pairs.append((center, points[best_point_idx]))
        best_dists.append(np.max(ds))

        points = [p for p, d in zip(points, ds) if d > discard_threshold]

    best_idx = np.argmax(best_dists)
    return best_pairs[best_idx], cycle_count


def run_experiment(dims, n_points, k_sigma_threshold):
    db = generate_points(dims, n_points)
    mu, sigma, _ = estimate_dist(db, n_points // 50)

    threshold = mu - sigma * k_sigma_threshold
    with Measure():
        pred, cycles = predict_farthes_pair(db, dist_func, threshold)

    all_dists = spatial.distance_matrix(db, db, 2)
    # best_p1, best_p2 = np.unravel_index(np.argmax(all_dists), all_dists.shape)
    # target = (db[best_p1], db[best_p2])

    # quality = dist_func(*pred) / dist_func(*target)
    quality = stats.percentileofscore(all_dists.flatten(), dist_func(*pred))
    return cycles, quality


def run_random_experiment(dims, n_points, k_sigma_threshold):
    del k_sigma_threshold
    db = generate_points(dims, n_points)

    cycles = int(n_points**1.4)
    a = rng.choice(db, size=cycles, replace=True)
    b = rng.choice(db, size=cycles, replace=True)
    ds = dist_func(a, b)
    best_idx = np.argmax(ds)
    pred = (a[best_idx], b[best_idx])

    all_dists = spatial.distance_matrix(db, db, 2)
    quality = stats.percentileofscore(all_dists.flatten(), dist_func(*pred))
    return cycles, quality


# distance array gets to big for 1e5 entries
for func in [run_experiment, run_random_experiment]:
    NS = np.logspace(2, 3.3, num=10)
    qualities = []
    cycles = []
    for n in tqdm(NS):
        n = int(n)
        results = [func(20, n, 1) for _ in range(30)]
        cycles.append(np.mean([r[0] for r in results]))
        qualities.append(np.mean([r[1] for r in results]))

    def polyn(x, expon, offset):
        return x**expon + offset

    params, _ = optimize.curve_fit(polyn, NS, cycles, p0=(2, 0))

    plt.subplot(2, 1, 1)
    plt.ylabel("cycles")
    plt.plot(
        NS, cycles, label=f"{func}" + "$\\mathcal{O}(n^{" + f"{params[0]:.1f}" + "})$"
    )
    plt.plot(NS, NS**2, label="baseline $\\mathcal{O}(n^2)$")
    plt.plot(NS, NS, label="target $\\mathcal{O}(n)$")
    plt.xlabel("database size")
    plt.legend()
    plt.loglog()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(NS, np.asarray(qualities), label=f"{func}")
    plt.ylabel("% of possible point pairs with lower dist")
    plt.xlabel("database size")
    plt.xscale("log")
    plt.grid()
    plt.legend()


plt.show()
