from metrica.metric import Euclid

METRIC = Euclid(2)


def choose_reasonably_remote_partner(ps, p0):
    """Return the last a partner in the list for `p0` that is farther away than the average distance."""
    dists = METRIC(p0, ps)
    cutoff = dists.mean()
    p1 = ps[dists > cutoff][-1]
    return p1
