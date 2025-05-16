from typing import Callable
import scipy
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
    center_images: bool = False,
):
    """
    Process a batch of images for use in GPSR. 
    The images are cropped, thresholded, pooled, median filtered, and normalized.
    An image_fitter function is used to fit the images and return the rms size and centroid to crop the images.

    Optionally, the images can be centered using the image_fitter function.

    Parameters
    ----------
    images : np.ndarray
        A batch of images with shape (..., H, W).
    screen_resolution : float
        The resolution of the screen in microns.
    image_fitter : Callable
        A function that fits an image and returns the rms size and centroid as a tuple in px coordinates.
        Example: <rms size>, (<x_center>, <y_center>) = image_fitter(image)
    threshold : float, optional
        The threshold to apply to the images before pooling and filters, by default None. If None, the threshold is calculated via the triangle method.
    pool_size : int, optional
        The size of the pooling window, by default None. If None, no pooling is applied.
    median_filter_size : int, optional
        The size of the median filter, by default None. If None, no median filter is applied.
    n_stds : int, optional
        The number of standard deviations to crop the images, by default 8.
    center_images : bool, optional
        Whether to center the images before processing, by default False. 
        If True, the images are centered using the image_fitter function.

    Returns
    -------
    np.ndarray
        The processed images with cropped shape (..., H', W').
    np.ndarray
        The meshgrid for the processed images.

    """

    batch_shape = images.shape[:-2]
    batch_dims = tuple(range(len(batch_shape)))
    center_location = np.array(images.shape[-2:]) // 2

    # center the images
    if center_images:
        # flatten batch dimensions
        images = np.reshape(images, (-1,) + images.shape[-2:])

        centered_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            # fit the image centers
            rms_size, centroid = image_fitter(images[i])

            # shift the images to center them
            centered_images[i] = scipy.ndimage.shift(
                images[i],
                -(centroid - center_location),
                order=1,
                mode="nearest",
            )

        # reshape back to original shape
        images = np.reshape(centered_images, batch_shape + images.shape[-2:])

    total_image = np.sum(images, axis=batch_dims)
    rms_size, centroid = image_fitter(total_image)

    crop_ranges = np.array(
        [
            (centroid - n_stds * rms_size).astype("int"),
            (centroid + n_stds * rms_size).astype("int"),
        ]
    )

    # transpose crop ranges temporarily to clip on image size
    crop_ranges = crop_ranges.T
    crop_ranges[0] = np.clip(crop_ranges[0], 0, images.shape[-2])
    crop_ranges[1] = np.clip(crop_ranges[1], 0, images.shape[-1])
    crop_ranges = crop_ranges.T

    processed_images = images[
        ...,
        crop_ranges[0][0] : crop_ranges[1][0],
        crop_ranges[0][1] : crop_ranges[1][1],
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
