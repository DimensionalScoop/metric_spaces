import numpy as np


def hyper_unit_grid(dimensions, samples):
    """generate a grid with `samples` points between [-0.5, 0.5] in any dims"""
    edge_length_in_samples = int(samples ** (1 / dimensions))
    _check_sample_size(edge_length_in_samples, dimensions)
    lin = np.linspace(-0.5, 0.5, edge_length_in_samples)
    grid_coors = [lin] * dimensions
    ggrid = np.meshgrid(*grid_coors)
    points = np.array([d.flatten() for d in ggrid]).T
    return points


def fill_hypercube(dimensions, samples):
    """fill a hypercube with random points"""
    return np.random.rand(int(samples), dimensions) - 0.5


def _check_sample_size(edge_length_in_samples, dimensions):
    if edge_length_in_samples < 2:
        reasonable_samples = 3**dimensions
        raise ValueError(
            f"""
        Too few samples to generate a grid in {dimensions} dimensions!
        >= {reasonable_samples:e} is a reasonable minimum"""
        )


def fill_rectangle(x_lim, y_lim, n):
    """Return about `n` points that are inside the given rectangle"""
    samples = int(np.sqrt(n))
    x = np.linspace(*x_lim, samples)
    y = np.linspace(*y_lim, samples)
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T
    return points


def fill_circle(radius, n):
    bounds = (-radius, radius)
    points = fill_rectangle(bounds, bounds, n)
    distance_to_center = np.sum(points**2, axis=1)
    return points[distance_to_center <= radius**2]


def move_points(points, delta_x, delta_y):
    points = points.copy()
    points[:, 0] += delta_x
    points[:, 1] += delta_y
    return points


def circle_around_point(radius, point, n):
    c = fill_circle(radius, n)
    return move_points(c, *point)
