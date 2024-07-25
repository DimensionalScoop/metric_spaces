import numpy as np
from scipy import stats
from tqdm.auto import tqdm


def random_PDF(rng: np.random.Generator, size=(1, 10), n_samples=100):
    assert len(size) == 2, "size should be of the form (sampels, bins)!"
    dist = stats.norm(loc=rng.random(size[0]), std=rng.random(size[0]))
    pdf = dist.rvs(size=n_samples, random_state=rng) / n_samples
    return pdf


def _order_axis(train):
    data = train[b"data"]
    data = data.reshape(-1, 3, 32, 32)
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 2, 1)
    return data


def load_cifar_100_train():
    """Return images as (n_image, pixel_x, pixel_y, n_color)"""

    def unpickle():
        train_path = "../data/cifar-100/train"
        import pickle

        with open(train_path, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    train = unpickle()
    train.keys()

    labels = np.asarray(train[b"coarse_labels"])
    data = _order_axis(train)
    return data


def color_histogram_1D(images, n_bins=256):
    """Generate three histogram per image, one for each primary color."""
    assert np.issubdtype(images.dtype, np.integer), "Input images must be integers"
    assert np.all(
        (images >= 0) & (images <= 255)
    ), "Pixel values must be between 0 and 255"

    n_images = images.shape[0]
    n_colors = images.shape[-1]
    images = images.reshape(n_images, -1, n_colors)
    hists = np.empty((n_images, n_colors, n_bins))

    for image_idx in tqdm(range(n_images)):
        for color in range(n_colors):
            hists[image_idx, color, :], _ = np.histogram(
                images[image_idx, :, color], bins=n_bins, range=(0, 255)
            )
    hists = hists / hists.sum(axis=-1, keepdims=True)
    return hists


def color_histogram_3D(images, n_bins=10):
    """Generate one histogram per image, treating each pixle as a 3D color vector."""
    assert np.issubdtype(images.dtype, np.integer), "Input images must be integers"
    assert np.all(
        (images >= 0) & (images <= 255)
    ), "Pixel values must be between 0 and 255"

    n_images = images.shape[0]
    n_colors = images.shape[-1]
    images = images.reshape(n_images, -1, n_colors)
    hists = np.empty((n_images, n_bins, n_bins, n_bins))

    for image_idx in tqdm(range(n_images)):
        hists[image_idx], _ = np.histogramdd(
            images[image_idx], bins=[n_bins] * 3, range=[(0, 255), (0, 255), (0, 255)]
        )

    normalization_factor = hists.reshape(n_images, -1).sum(axis=-1)
    normalization_factor = normalization_factor.reshape(n_images, 1, 1, 1)
    hists = hists / normalization_factor
    return hists
