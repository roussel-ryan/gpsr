from typing import Callable, Literal
import scipy
from skimage.measure import block_reduce
from skimage.filters import threshold_triangle
from scipy.ndimage import median_filter
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def compute_blob_stats(image):
    """
    Compute the center (centroid) and RMS size of a blob in a 2D image
    using intensity-weighted averages.

    Parameters:
        image (np.ndarray): 2D array representing the image.

    Returns:
        center (tuple): (x_center, y_center)
        rms_size (tuple): (x_rms, y_rms)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array")

    # Get coordinate grids
    y_indices, x_indices = np.indices(image.shape)

    # Flatten everything
    x = x_indices.ravel()
    y = y_indices.ravel()
    weights = image.ravel()

    # Total intensity
    total_weight = np.sum(weights)
    if total_weight == 0:
        raise ValueError(
            "Total image intensity is zero — can't compute centroid or RMS size."
        )

    # Weighted centroid
    x_center = np.sum(x * weights) / total_weight
    y_center = np.sum(y * weights) / total_weight

    # Weighted RMS size
    x_rms = np.sqrt(np.sum(weights * (x - x_center) ** 2) / total_weight)
    y_rms = np.sqrt(np.sum(weights * (y - y_center) ** 2) / total_weight)

    return np.array((x_rms, y_rms)), np.array((x_center, y_center))


def process_images(
    images: np.ndarray,
    pixel_size: float,
    image_fitter: Callable = compute_blob_stats,
    pool_size: Optional[int] = None,
    median_filter_size: Optional[int] = None,
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    n_stds: int = 8,
    normalization: Literal["independent", "max_intensity_image"] = "independent",
    center_images: bool = False,
    visualize: bool = False,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Process a batch of images for use in GPSR.
    The images are cropped, thresholded, pooled, median filtered, and normalized.
    An image_fitter function is used to fit the images and return the rms size and centroid to crop the images.

    Optionally, the images can be centered using the image_fitter function.

    Parameters
    ----------
    images : np.ndarray
        A batch of images with shape (..., H, W).
    pixel_size : float
        Pixel size of the screen in microns.
    image_fitter : Callable
        A function that fits an image and returns the rms size and centroid as a tuple in px coordinates.
        Example: <rms size>, (<x_center>, <y_center>) = image_fitter(image)
    threshold : float, optional
        The threshold to apply to the images before pooling and filters, by default None. If None, the threshold is calculated via the triangle method.
    threshold_multiplier : float, optional
        The multiplier for the threshold, by default 1.0.
    pool_size : int, optional
        The size of the pooling window, by default None. If None, no pooling is applied.
    median_filter_size : int, optional
        The size of the median filter, by default None. If None, no median filter is applied.
    n_stds : int, optional
        The number of standard deviations to crop the images, by default 8.
    normalization : str, optional
        Normalization method: 'independent' (default) or 'max_intensity_image'.
    center_images : bool, optional
        Whether to center the images before processing, by default False.
        If True, the images are centered using the image_fitter function.
    visualize : bool, optional
        Whether to visualize the images at each step of the processing, by default False.
        If True, the images are displayed using matplotlib.

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
    center_location = center_location[::-1]

    if visualize:
        plt.figure()
        plt.imshow(images[(0,) * len(batch_shape)])

    # median filter
    if median_filter_size is not None:
        images = median_filter(
            images,
            size=median_filter_size,
            axes=[-2, -1],
        )

    # apply threshold if provided -- otherwise calculate threshold using triangle method
    if threshold is None:
        avg_image = np.mean(images, axis=batch_dims)
        threshold = threshold_triangle(avg_image)
    images = np.clip(images - threshold_multiplier * threshold, 0, None)

    if visualize:
        plt.figure()
        plt.title("post filtering and thresholding")
        plt.imshow(images[(0,) * len(batch_shape)])

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
                -(centroid - center_location)[::-1],
                order=1,  # linear interpolation to avoid artifacts
            )

            if visualize:
                fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
                ax[0].imshow(images[i])
                ax[0].plot(*centroid, "r+")
                ax[1].imshow(centered_images[i])
                ax[1].plot(*center_location, "r+")

        # reshape back to original shape
        images = np.reshape(centered_images, batch_shape + images.shape[-2:])

    if visualize:
        plt.figure()
        plt.title("post image centering")
        plt.imshow(images[(0,) * len(batch_shape)])

    total_image = np.mean(images, axis=batch_dims)
    rms_size, centroid = image_fitter(total_image)
    centroid = centroid[::-1]
    rms_size = rms_size[::-1]

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

    print("crop ranges", crop_ranges)
    print(images.shape)

    if visualize:
        plt.figure()
        plt.imshow(total_image)
        plt.plot(*centroid[::-1], "+r")
        rect = plt.Rectangle(
            (crop_ranges[1][0], crop_ranges[0][0]),
            crop_ranges[1][1] - crop_ranges[1][0],
            crop_ranges[0][1] - crop_ranges[0][0],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rect)

    images = images[
        ...,
        crop_ranges[0][0] : crop_ranges[0][1],
        crop_ranges[1][0] : crop_ranges[1][1],
    ]

    if visualize:
        plt.figure()
        plt.title("post cropping")
        plt.imshow(images[(0,) * len(batch_shape)])

    # pooling
    if pool_size is not None:
        block_size = (1,) * len(batch_shape) + (pool_size,) * 2
        images = block_reduce(images, block_size=block_size, func=np.mean)

    # normalize image intensities
    if normalization == "independent":
        # normalize each image independently
        scale_factor = np.sum(images, axis=(-2, -1), keepdims=True)
    elif normalization == "max_intensity_image":
        # normalize by the maximum intensity image
        scale_factor = np.max(np.sum(images, axis=(-2, -1)))
    else:
        raise ValueError(
            "Normalization must be 'independent' or 'max_intensity_image'."
        )

    images = images / scale_factor

    # compute meshgrids for screens
    bins = []
    pool_size = 1 if pool_size is None else pool_size

    # returns left sided bins
    for j in [-2, -1]:
        img_bins = np.arange(images.shape[j])
        img_bins = img_bins - len(img_bins) / 2
        img_bins = img_bins * pixel_size * 1e-6 * pool_size
        bins += [img_bins]

    return images, np.meshgrid(*bins)
