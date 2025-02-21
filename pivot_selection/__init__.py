from . import optimal
from . import heuristics_fast as fast
from . import heuristics_complete as heuristics
from . import lb_summation as lbsum

__all__ = ["get_selection_algos", "optimal", "heuristics_fast", "heuristics_complete"]


def get_selection_algos(only_useful=False) -> dict[str, callable]:
    algs = dict(
        random=heuristics.random_pivots,
        maximize_dist=heuristics.max_dist_points,
        gnat_dist=fast.max_dist_GNAT,
        minimize_dist=heuristics.min_dist_points,
        central_points=heuristics.two_most_central,
        non_central_points=heuristics.two_least_central,
        non_central_points_approx=fast.two_least_central_heuristically,
        remoteness=heuristics.two_remote_points,
        central_and_distant=heuristics.central_and_distant,
        different_cluster_centers=heuristics.different_cluster_centers,
    )
    __add_lbsums(algs)
    algs.update(
        dict(
            hilbert_optimal=optimal.hilbert_optimal_pivots,
            ccs_optimal=optimal.ccs_optimal_pivot,
        )
    )
    if only_useful:
        del algs["minimize_dist"]
        del algs["central_points"]
    return algs


def __add_lbsums(algs):
    params = [
        dict(
            name="IS_tri_1.5",
            lb_type="tri",
            fixed_first_pivot=False,
            piv_exp=3 / 4,
            pair_exp=3 / 4,
        ),
        dict(
            name="IS_pto_1.5",
            lb_type="pto",
            fixed_first_pivot=False,
            piv_exp=3 / 4,
            pair_exp=3 / 4,
        ),
        dict(
            name="IS_tri_1.5_greedy",
            lb_type="tri",
            fixed_first_pivot=True,
            piv_exp=3 / 4,
            pair_exp=3 / 4,
        ),
        dict(
            name="IS_pto_1.5_greedy",
            lb_type="tri",
            fixed_first_pivot=True,
            piv_exp=3 / 4,
            pair_exp=3 / 4,
        ),
    ]

    def gen_f(params):
        p = params

        def f(ps, rng):
            n = len(ps)
            return lbsum.IS(
                ps=ps,
                rng=rng,
                n_pivs=int(n ** p["piv_exp"]),
                n_pairs=int(n ** p["pair_exp"]),
                lb_type=p["lb_type"],
                fixed_first_pivot=p["fixed_first_pivot"],
            )

        return f

    for p in params:
        algs[p["name"]] = gen_f(p)
    return algs
