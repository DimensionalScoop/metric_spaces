from run import run
from datetime import datetime

from missing_seeds import missing_seeds
from meters import pivot_selection

algs = set(pivot_selection.get_selection_algos(True).keys()) - set(["optimal"])

run(
    name=f"e_missing_seeds_{datetime.now().isoformat().replace(':', '.')}",
    metric=("Euclidean", 2),
    n_samples=512,
    n_queries=128,
    dims=list(range(2, 18)),
    n_cpus=-1,
    n_runs=99999,
    seed=missing_seeds,
    verbose=False,
    dbfile="results/missing_seeds.duck",
    algorithms=list(algs),
)
