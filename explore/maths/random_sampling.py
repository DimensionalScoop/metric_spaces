import numpy as np
from scipy import spatial
from tqdm import tqdm
from scipy import stats

import matplotlib.pyplot as plt

rng = np.random.default_rng(0xFEED)

SAMPELS = 1000
linsize = 100000
n = 10000  # int(linsize ** (3 / 4))
best = []
best_approx = []

lin = np.linspace(0, 1, linsize)
best = []
for s in tqdm(range(SAMPELS)):
    a = rng.choice(lin, size=n, replace=False)
    best.append(np.max(a))

plt.hist(best, bins=100, density=True)

params = stats.weibull_max.fit(best)
# this seems to work for n << linsize
manual_params = (1, 1, 1 / n)

x = np.linspace(min(best), 0.99999, 150)
plt.plot(x, stats.weibull_max(*manual_params).pdf(x))
plt.title(params[-1] * n)
plt.show()


#
# for s in tqdm(range(SAMPELS)):
#     database = rng.uniform(-1, 1, size=[100, 10])
#     dist_matrix = spatial.distance_matrix(database, database)
#     all_indices = np.tril_indices(len(database), -1)
#     dists = dist_matrix[all_indices]
#
#     best.append(np.amax(dist_matrix))
#
#     candidates = rng.choice(all_indices, int(len(database) ** (3 / 4)), replace=False)
#     best_approx.append(np.amax(dist_matrix[candidates]))
