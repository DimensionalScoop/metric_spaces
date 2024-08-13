from . import optimal
from . import heuristics_fast as fast
from . import heuristics_complete as heuristics

__all__ = ["get_selection_algos", "optimal", "heuristics_fast", "heuristics_complete"]


def get_selection_algos(only_useful=False) -> dict[str, callable]:
    algs = dict(
        random=heuristics.random_pivots,
        maximize_dist=heuristics.max_dist_points,
        minimize_dist=heuristics.min_dist_points,
        central_points=heuristics.two_most_central,
        non_central_points=heuristics.two_least_central,
        non_central_points_approx=fast.two_least_central_heuristically,
        remoteness=heuristics.two_remote_points,
        central_and_distant=heuristics.central_and_distant,
        different_cluster_centers=heuristics.different_cluster_centers,
        triangle_IS=fast.triangular_incremental_selection,
        Ptolemy_IS=fast.ptolemys_incremental_selection,
        hilbert_optimal=optimal.hilbert_optimal_pivots,
        ccs_optimal=optimal.ccs_optimal_pivot,
    )
    if only_useful:
        del algs["minimize_dist"]
        del algs["central_points"]
    return algs
