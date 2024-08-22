import numpy as np
import pandas as pd
from joblib import delayed, Parallel, Memory
from tqdm import tqdm
import time
from itertools import permutations

from tetrahedron import tetrahedron, proj_quality
from metric.metric import Euclid

import pivot_selection
from generate import point_generator

mem = Memory("/tmp/")

metric = Euclid(2)
N_RUNS = 1000
N_SAMPLES = 1024
SEED = 0xFEED
DIMS = range(3, 11)
PAIRS_PER_SAMPLE = 2000

rng = np.random.default_rng(SEED)
generators = point_generator.get_generator_dict(N_SAMPLES)


def Pto(p0, p1, x, y):
    p = metric(p0, p1)
    x0 = metric(p0, x)
    x1 = metric(p1, x)
    y0 = metric(p0, y)
    y1 = metric(p1, y)
    return 1 / p * np.abs(x0 * y1 - x1 * y0)


def pivot_quality(points, p0, p1, lb_func, rng):
    lhs = rng.choice(points, size=PAIRS_PER_SAMPLE)
    rhs = rng.choice(points, size=PAIRS_PER_SAMPLE)
    lb = lb_func(p0, p1, lhs, rhs)
    target = metric(lhs, rhs)
    return lb.sum() / target.sum()


@mem.cache
def run(N_RUNS, N_SAMPLES, SEED, DIMS, PAIRS_PER_SAMPLE):
    return pd.DataFrame(_run(N_RUNS, N_SAMPLES, SEED, DIMS, PAIRS_PER_SAMPLE))


def _run(N_RUNS, N_SAMPLES, SEED, DIMS, PAIRS_PER_SAMPLE):
    for run in tqdm(range(N_RUNS)):
        for name, gen in generators.items():
            for dim in DIMS:
                points = gen(rng=rng, dim=dim)
                pivs = points[:3]
                points = points[3:]
                for p0, p1 in permutations(range(len(pivs)), 2):
                    yield dict(
                        run=run,
                        name=name,
                        dim=dim,
                        p0=p0,
                        p1=p1,
                        quality=pivot_quality(points, pivs[p0], pivs[p1], Pto, rng),
                    )


df = run(N_RUNS, N_SAMPLES, SEED, DIMS, PAIRS_PER_SAMPLE)
print(df)


def one_much_better(group):
    s = sorted(group.quality)
    lower = 0.5 * (s[0] + s[1])
    upper = s[2]
    return (upper - lower) / lower


def average(group):
    s = sorted(group.quality)
    lower = 0.5 * (s[0] + s[1])
    upper = s[2]
    return lower, upper


r = df.groupby(["run", "name", "dim"]).apply(one_much_better).rename("top")

import seaborn as sns
import matplotlib.pyplot as plt

c = df.query("p0==0")

print(c.query("p1==1").quality.corr(c.query("p1==2").quality))

raise NotImplementedError()

# dim does not influence this
g = sns.FacetGrid(r.reset_index(), col="name", col_wrap=4)
g.map(
    sns.kdeplot,
    "top",
)
plt.show()


def better(group):
    def qual(p0, p1):
        return group.query("p0 == @p0 and p1 == @p1").quality

    return qual(0, 1) + qual(1, 2) + qual(0, 2)
