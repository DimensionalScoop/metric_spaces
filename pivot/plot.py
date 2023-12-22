import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import matplotlib as mpl
import scipy.stats as stats
from scipy.spatial import ConvexHull

from matplotlib.collections import LineCollection
from dataclasses import dataclass


def plot_grid(xx,yy, ax=None, xcolor=None, ycolor=None, **kwargs):
    """plot a grid from a meshgrid input"""
    # from https://stackoverflow.com/questions/47295473/how-to-plot-using-matplotlib-python-colahs-deformed-grid
    ax = ax or plt.gca()
    segs1 = np.stack((xx,yy), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, color=xcolor, **kwargs))
    ax.add_collection(LineCollection(segs2, color=ycolor, **kwargs))
    ax.autoscale()

def plot_hull(points, ax=None, color="gray", **kwargs):
    hull = ConvexHull(points)
    hull_idx = np.hstack((hull.vertices, hull.vertices[0])) # to close the polygon
    hull_points = points[hull_idx,:]

    line_segments = hull_points.reshape(1,-1,2) 

    ax = ax or plt.gca()
    ax.add_collection(LineCollection(line_segments, colors = color, facecolors=color, **kwargs))
    ax.autoscale()



@dataclass
class GridPlot:
    x:np.ndarray
    y:np.ndarray

    resolution:int = 10
    cmap = mpl.colormaps["plasma"]
