import numpy as np
from tqdm import tqdm
from scipy import spatial
from joblib import Memory

mem = Memory()

rand = np.random.default_rng()

dims = 5
n = 1_0000
db = rand.uniform(low=-1, high=1, size=[n, dims])


d = spatial.minkowski_distance
get_dist_matrix = mem.cache()(spatial.distance_matrix)


def find_lonely_points(db):
    dist_matrix = get_dist_matrix(db, db)
    # ignore distance of self
    np.fill_diagonal(dist_matrix, np.inf)
    # all points that are nearest neighbors
    nearest_neighbors = np.argmin(dist_matrix, axis=1)
    lonely = set(range(len(dist_matrix))) - set(nearest_neighbors)
    lonely = list(lonely)

    return lonely, np.min(dist_matrix, axis=1)[lonely]


idx_lonely, dists = find_lonely_points(db)
top_lonely_idx = np.argsort(dists)[::-1]

print("lonely", len(idx_lonely)/len(db), "of all points")
print("best lonely dists", dists[top_lonely_idx[10]])


dist_matrix = get_dist_matrix(db, db)
np.fill_diagonal(dist_matrix, np.inf)
average_10nn_dist = np.mean(np.sort(dist_matrix, axis=1)[:,9])
print("average 10nn dist", average_10nn_dist)

queries = 1000
q = rand.uniform(low=-1, high=1, size=[queries, dims])
most_lonely = db[top_lonely_idx[0]]
second_lonely = db[top_lonely_idx[1]]
pivot = rand.permutation(db)[:queries]
is_on_pivot_side = np.abs(d(q,most_lonely) - d(q,pivot)) > 2*average_10nn_dist
print("successful lonely exclusions",np.sum(is_on_pivot_side)/queries)

q = rand.uniform(low=-1, high=1, size=[queries, dims])
most_lonely = db[top_lonely_idx[0]]
pivot_1 = rand.permutation(db)[:queries]
pivot_2 = rand.permutation(db)[:queries]
is_on_pivot_side = np.abs(d(q,pivot_1) - d(q,pivot_2)) > 2*average_10nn_dist
print("successful random exclusions",np.sum(is_on_pivot_side)/queries)

