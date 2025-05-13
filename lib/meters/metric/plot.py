import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.spatial import ConvexHull

from matplotlib.collections import LineCollection
from dataclasses import dataclass
from contextlib import contextmanager

from pivot.transform import PivotSpace


def plot_grid(xx, yy, ax=None, xcolor=None, ycolor=None, **kwargs):
    """plot a grid from a meshgrid input"""
    # from https://stackoverflow.com/questions/47295473/how-to-plot-using-matplotlib-python-colahs-deformed-grid
    ax = ax or plt.gca()
    segs1 = np.stack((xx, yy), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color=xcolor, **kwargs))
    ax.add_collection(LineCollection(segs2, color=ycolor, **kwargs))
    ax.autoscale()


def plot_hull(points, ax=None, color="gray", **kwargs):
    hull = ConvexHull(points)
    hull_idx = np.hstack((hull.vertices, hull.vertices[0]))  # to close the polygon
    hull_points = points[hull_idx, :]

    line_segments = hull_points.reshape(1, -1, 2)

    ax = ax or plt.gca()
    ax.add_collection(
        LineCollection(line_segments, colors=color, facecolors=color, **kwargs)
    )
    ax.autoscale()


@dataclass
class GridPlot:
    x: np.ndarray
    y: np.ndarray

    resolution: int = 10
    cmap = mpl.colormaps["plasma"]


def _rescale_one_axis(old_lims, points_to_include):
    lb, ub = min(points_to_include), max(points_to_include)
    new_lims = min((old_lims[0], lb)), max((old_lims[1], ub))
    return new_lims


def _scale_limits(ax, include_points):
    ax.set_xlim(
        _rescale_one_axis(
            ax.get_xlim(),
            include_points[:, 0],
        )
    )
    ax.set_ylim(
        _rescale_one_axis(
            ax.get_ylim(),
            include_points[:, 1],
        )
    )


def points_to_pivot_space(points, pivots, dist_func):
    n_points = points.shape[0]
    n_dim = len(pivots)
    transformed = np.empty([n_points, n_dim])
    for dim, pv in enumerate(pivots):
        transformed[:, dim] = dist_func(pv, points)
    return transformed


def mask_forbidden(pivots, dist_func, ax=None):
    """Plot a gray area where the pivot space is inaccessible.

    This region does not have an preimage, as can easily be seen
    by using the triangle inequality, with both pivots being two
    points of the triangle.
    """
    if ax is None:
        ax = plt.gca()

    def mask(*args, **kwargs):
        return ax.fill_between(
            *args,
            **kwargs,
            color="xkcd:light grey",
        )

    p_1, p_2 = points_to_pivot_space(pivots, pivots, dist_func)
    b = p_2[0]
    m = -1
    inacc_below = lambda x: m * x + b

    # we don't want to influence the existing limits
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    x = np.linspace(-10, 10)
    y = inacc_below(x)
    mask(x, y, y - 100)

    m = +1
    inacc_above = lambda x: m * x + b
    y = inacc_above(x)
    mask(x, y, y + 100)

    inacc_below = lambda x: m * x - b
    y = inacc_below(x)
    mask(x, y - 100, y)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylim(ylim)


class PivotSpaceVisualizer:
    def __init__(self, piv_space: PivotSpace, save_path=""):
        self.piv = piv_space
        self.save_path = save_path
        self.scale_to_include_pivots = True

    def plot(self, fname, plot_in_metric_space_func, plot_in_pivot_space_func):
        fig = plt.figure(figsize=(10, 5), dpi=300)

        with self._plot_metric_space(fig) as ax:
            plot_in_metric_space_func(ax)

        with self._plot_pivot_space(fig) as ax:
            plot_in_pivot_space_func(ax)

        fig.tight_layout()
        fig.savefig(self.save_path + fname)

    @contextmanager
    def _plot_metric_space(self, fig):
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title(r"metric space $(\mathbb{R}^2, " + self.piv.metric.name + ")$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        yield ax

        ax.plot(*self.piv.pivots.T, "o", color="C2", label="pivots")
        ax.legend()
        ax.set_aspect("equal")

    @contextmanager
    def _plot_pivot_space(self, fig):
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title(r"pivot space $(\mathbb{R}_+)^2$")
        ax.set_xlabel("$\Phi_1$")
        ax.set_ylabel("$\Phi_2$")

        yield ax

        mask_forbidden(self.piv.pivots, self.piv.metric)
        pivots_t = self.piv.transform_points(self.piv.pivots)
        ax.plot(*pivots_t, "o", color="C2", label="pivots")
        ax.set_aspect("equal")

        if self.scale_to_include_pivots:
            _scale_limits(ax, pivots_t * 0.8)
