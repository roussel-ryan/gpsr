from typing import Callable
from skimage.measure import block_reduce
from skimage.filters import threshold_triangle
from scipy.ndimage import median_filter
import numpy as np


def process_images(
    images: np.ndarray,
    screen_resolution: float,
    image_fitter: Callable,
    pool_size: int = None,
    median_filter_size: int = None,
    threshold: float = None,
    n_stds: int = 8,
):
    """
    Process a batch of images for use in GPSR. The images are cropped, thresholded, pooled, median filtered, and normalized.

    Parameters
    ----------
    images : np.ndarray
        A batch of images with shape (..., H, W).
    screen_resolution : float
        The resolution of the screen in microns.
    image_fitter : Callable
        A function that fits an image and returns the rms size and centroid as a tuple in px coordinates.
    threshold : float, optional
        The threshold to apply to the images before pooling and filters, by default None. If None, the threshold is calculated via the triangle method.
    pool_size : int, optional
        The size of the pooling window, by default None. If None, no pooling is applied.
    median_filter_size : int, optional
        The size of the median filter, by default None. If None, no median filter is applied.
    n_stds : int, optional
        The number of standard deviations to crop the images, by default 8.

    Returns
    -------
    np.ndarray
        The processed images with shape (..., H', W').
    np.ndarray
        The meshgrid for the processed images.

    """

    batch_shape = images.shape[:-2]
    batch_dims = tuple(range(len(batch_shape)))

    total_image = np.sum(images, axis=batch_dims)

    rms_size, centroid = image_fitter(total_image)

    crop_ranges = [
        (centroid - n_stds * rms_size).astype("int"),
        (centroid + n_stds * rms_size).astype("int"),
    ]
    processed_images = images[
        ...,
        crop_ranges[0][1] : crop_ranges[1][1],
        crop_ranges[0][0] : crop_ranges[1][0],
    ]

    # apply threshold if provided -- otherwise calculate threshold using triangle method
    if threshold is None:
        avg_image = np.mean(processed_images, axis=batch_dims)
        threshold = threshold_triangle(avg_image)
    processed_images = np.clip(processed_images - threshold, 0, None)

    # pooling
    if pool_size is not None:
        block_size = (1,) * len(batch_shape) + (pool_size,) * 2
        processed_images = block_reduce(
            processed_images, block_size=block_size, func=np.mean
        )

    # median filter
    if median_filter_size is not None:
        processed_images = median_filter(
            processed_images,
            size=median_filter_size,
            axes=[-2, -1],
        )

    # normalize image intensities such that the peak intensity image has a sum of 1
    total_intensities = np.sum(processed_images, axis=(-2, -1))
    scale_factor = np.max(total_intensities)
    processed_images = processed_images / scale_factor

    # compute meshgrids for screens
    bins = []
    pool_size = 1 if pool_size is None else pool_size

    # returns left sided bins
    for j in [-2, -1]:
        img_bins = np.arange(processed_images.shape[j])
        img_bins = img_bins - len(img_bins) / 2
        img_bins = img_bins * screen_resolution * 1e-6 * pool_size
        bins += [img_bins]

    return processed_images, np.meshgrid(*bins)
