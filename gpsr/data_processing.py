from typing import Callable, Literal
from math import gcd

import scipy
from skimage.measure import block_reduce
from skimage.filters import threshold_triangle
from scipy.ndimage import median_filter
from skimage.transform import resize
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def compute_blob_stats(image):
    """
    Compute the RMS size and centroid of a blob in a 2D image using intensity-weighted averages.

    Parameters
    ----------
    image : np.ndarray
        2D array representing the image. Shape should be (height (y size), width (x size)).

    Returns
    -------
    centroid : np.ndarray
        Array containing the centroid coordinates (x_center, y_center).
    rms_size : np.ndarray
        Array containing the RMS size along (x, y) axes.
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

    return np.array((x_center, y_center)), np.array((x_rms, y_rms))


def calc_image_centroids(
    images: np.ndarray, image_fitter: Callable = compute_blob_stats
) -> np.ndarray:
    """
    Calculate centroids for a batch of images using the provided image_fitter function.

    Parameters
    ----------
    images : np.ndarray
        Batch of images with shape (..., height (y size), width (x size)).
    image_fitter : Callable, optional
        Function that returns (centroid, rms) for a single image.

    Returns
    -------
    np.ndarray
        Array of centroids with shape (..., 2), where the last dimension is (x, y) coordinates.
    """

    batch_shape = images.shape[:-2]
    flattened_images = images.reshape((-1,) + images.shape[-2:])
    flattened_centroids = np.zeros((flattened_images.shape[0], 2))

    for i in range(flattened_images.shape[0]):
        centroid, _ = image_fitter(flattened_images[i])
        flattened_centroids[i] = centroid

    return flattened_centroids.reshape(batch_shape + (2,))


def center_images(
    images: np.ndarray,
    image_centroids: np.ndarray,
    visualize: bool = False,
) -> np.ndarray:
    """
    Centers a batch of images based on provided centroid coordinates.

    Each image in the batch is shifted such that its centroid aligns with the center of the image.
    Optionally, visualizes the centering process for each image.

    Parameters
    ----------
    images : np.ndarray
        Batch of images with shape (..., height (y size), width (x size)).
    image_centroids : np.ndarray
        Array of centroid coordinates for each image, shape (..., 2).
    visualize : bool, optional
        If True, displays before and after centering for each image. Default is False.

    Returns
    -------
    np.ndarray
        Batch of centered images with the same shape as the input.
    """

    batch_shape = images.shape[:-2]
    center_location = np.array(images.shape[-2:]) // 2
    center_location = center_location[::-1]

    # Flatten batch dimensions
    flattened_images = images.reshape((-1,) + images.shape[-2:])
    flattened_centroids = image_centroids.reshape((flattened_images.shape[0], 2))
    centered_images = np.zeros_like(flattened_images)

    for i in range(flattened_images.shape[0]):
        # Shift the images to center them
        centered_images[i] = scipy.ndimage.shift(
            flattened_images[i],
            -(flattened_centroids[i] - center_location)[::-1],
            order=1,  # Linear interpolation to avoid artifacts
        )

        if visualize:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].imshow(flattened_images[i])
            ax[0].plot(*flattened_centroids[i], "r+")
            ax[1].imshow(centered_images[i])
            ax[1].plot(*center_location, "r+")

    # Reshape back to original shape
    centered_images = centered_images.reshape(images.shape)

    if visualize:
        plt.figure()
        plt.title("post image centering")
        plt.imshow(centered_images[(0,) * len(batch_shape)])

    return centered_images


def calc_crop_ranges(
    images,
    n_stds: int = 8,
    image_fitter=compute_blob_stats,
    filter_size: int = 5,
    visualize: bool = False,
) -> np.ndarray:
    """
    Calculate crop ranges for a batch of images based on the centroid and RMS size of the mean image.

    Parameters
    ----------
    images : np.ndarray
        Batch of images with shape (..., height (y size), width (x size)).
    n_stds : int, optional
        Number of standard deviations (RMS size) to include in the crop range. Default is 8.
    image_fitter : Callable, optional
        Function to compute centroid and RMS size from an image. Default is compute_blob_stats.
    visualize : bool, optional
        If True, displays a visualization of the crop range on the mean image. Default is False.

    Returns
    -------
    np.ndarray
        Array of shape (2, 2) containing crop ranges for each axis:
        [[start_x, end_x], [start_y, end_y]].
    """

    batch_shape = images.shape[:-2]
    batch_dims = tuple(range(len(batch_shape)))

    test_images = np.copy(images)
    total_image = np.mean(test_images, axis=batch_dims)

    # apply a strong median filter to remove noise
    total_image = median_filter(total_image, size=filter_size)

    # apply a threshold to remove background noise
    threshold = threshold_triangle(total_image)

    total_image[total_image < threshold] = 0

    centroid, rms_size = image_fitter(total_image)
    centroid = centroid[::-1]
    rms_size = rms_size[::-1]

    crop_ranges = np.array(
        [
            (centroid - n_stds * rms_size).astype("int"),
            (centroid + n_stds * rms_size).astype("int"),
        ]
    )
    crop_ranges = crop_ranges.T  # Transpose to match (start_x, end_x), (start_y, end_y)

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

    return crop_ranges


def crop_images(
    images: np.ndarray,
    crop_ranges: np.ndarray,
    visualize: bool = False,
) -> np.ndarray:
    """
    Crops a batch of images according to specified crop ranges.

    Parameters
    ----------
    images : np.ndarray
        A batch of images to be cropped. The shape should be (..., height (y_size), width (x_size)).
    crop_ranges : np.ndarray
        An array specifying the crop ranges for x,y. Should be of shape (2, 2),
        where crop_ranges[0] is [start_x, end_x] and crop_ranges[1] is [start_y, end_y].
    visualize : bool, optional
        If True, displays the first cropped image using matplotlib for visualization. Default is False.

    Returns
    -------
    np.ndarray
        The cropped images as a numpy array with the same batch dimensions as the input.
    """
    if crop_ranges.shape != (2, 2):
        raise ValueError("crop_ranges must be of shape (2, 2)")

    if images.ndim < 2:
        raise ValueError(
            "images must have at least 2 dimensions (batch, height, width)"
        )

    batch_shape = images.shape[:-2]

    crop_ranges[0] = np.clip(crop_ranges[0], 0, images.shape[-2])
    crop_ranges[1] = np.clip(crop_ranges[1], 0, images.shape[-1])

    cropped_images = images[
        ...,
        crop_ranges[0][0] : crop_ranges[0][1],
        crop_ranges[1][0] : crop_ranges[1][1],
    ]
    print("images shape: ", images.shape)
    print("cropped images shape: ", cropped_images.shape)

    if visualize:
        plt.figure()
        plt.title("post cropping")
        plt.imshow(cropped_images[(0,) * len(batch_shape)])

    return cropped_images


def pool_images(images: np.ndarray, pool_size) -> np.ndarray:
    """
    Pools (downsamples) the input images by applying mean pooling over non-overlapping blocks.

    Parameters
    ----------
    images : np.ndarray
        Input array of images. The last two dimensions are assumed to be spatial (height, width).
    pool_size : Optional[int], optional
        Size of the pooling window along each spatial dimension. If None, no pooling is applied.

    Returns
    -------
    np.ndarray
        Array of pooled images with reduced spatial dimensions.
    """

    batch_shape = images.shape[:-2]
    block_size = (1,) * len(batch_shape) + (pool_size,) * 2
    pooled_images = block_reduce(images, block_size=block_size, func=np.mean)
    return pooled_images


def adaptive_reduce(x: np.ndarray, max_elems: int, min_images: int = 10):
    """
    Reduce array size adaptively while keeping at least `min_images` images
    (if available) and maximum resolution under the element budget.

    Subsamples along the last batch axis (axis=-3).
    Tries block_reduce with exact divisors first, falls back to interpolation resize
    if pooling cannot meet the constraint.

    Args:
        x: (..., B, N, M) numpy array of images
        max_elems: maximum allowed total elements
        min_images: minimum number of images to keep (if available)

    Returns:
        (reduced_x, k)
        reduced_x: Reduced array (..., B', N', M')
        k: pooling factor if block_reduce / resize
        idxs: indicies of subsampled data along the batch axis
    """
    if x.ndim < 3:
        raise ValueError("Input must have at least 3 dims (..., B, N, M)")

    *batch_prefix, B, N, M = x.shape
    prefix_size = int(np.prod(batch_prefix, dtype=int)) if batch_prefix else 1

    total = x.size
    if total <= max_elems:
        return x, 1, np.arange(B)  # already under budget

    # --- Step 1: Determine how many images we can afford at full resolution ---
    max_fullres_images = max_elems // (N * M * prefix_size)
    if B <= min_images:
        B_target = B
    else:
        B_target = max(min_images, min(B, max_fullres_images))

    # --- Step 2: Subsample evenly along axis=-3 ---
    if B > B_target:
        idxs = np.linspace(0, B - 1, B_target).round().astype(int)
        x = np.take(x, idxs, axis=-3)
        B = B_target
    else:
        idxs = np.arange(B)

    # --- Step 3: Interpolation resize  ---
    # Compute target resolution
    target_area = max_elems / (B * prefix_size)
    scale = np.sqrt(target_area / (N * M))
    N_target = max(1, int(N * scale))
    M_target = max(1, int(M * scale))

    # Resize each image (handling arbitrary batch prefix)
    new_shape = (*batch_prefix, B, N_target, M_target)
    reduced = np.zeros(new_shape, dtype=x.dtype)
    flat = x.reshape((-1, N, M))  # flatten batch dims
    for i in range(flat.shape[0]):
        reduced.reshape((-1, N_target, M_target))[i] = resize(
            flat[i], (N_target, M_target), anti_aliasing=True, preserve_range=True
        )

    return reduced, 1 / scale, idxs


def normalize_images(
    images: np.ndarray,
    normalization: Literal["independent", "max_intensity_image"] = "independent",
) -> np.ndarray:
    """
    Normalize a batch of images using the specified normalization method.

    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (..., H, W), where H and W are image height and width.
    normalization : Literal["independent", "max_intensity_image"], optional
        Method for normalization:
        - "independent": Each image is normalized by its own sum of pixel intensities.
        - "max_intensity_image": All images are normalized by the maximum sum of pixel intensities across the batch.
        Default is "independent".

    Returns
    -------
    np.ndarray
        Array of normalized images with the same shape as the input.

    Raises
    ------
    ValueError
        If an unsupported normalization method is provided.
    """

    if normalization == "independent":
        scale_factor = np.sum(images, axis=(-2, -1), keepdims=True)
    elif normalization == "max_intensity_image":
        scale_factor = np.max(np.sum(images, axis=(-2, -1)))
    else:
        raise ValueError(
            "Normalization must be 'independent' or 'max_intensity_image'."
        )

    return images / scale_factor


def process_images(
    images: np.ndarray,
    pixel_size: float,
    image_fitter: Callable = compute_blob_stats,
    pool_size: Optional[int] = None,
    max_pixels: Optional[int] = None,
    median_filter_size: Optional[int] = None,
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    n_stds: int = 8,
    normalization: Literal["independent", "max_intensity_image"] = "independent",
    center: bool = False,
    crop: bool = False,
    image_centroids: Optional[np.ndarray] = None,
    crop_ranges: Optional[np.ndarray] = None,
    visualize: bool = False,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Process a batch of images for use in GPSR.

    Applies a series of processing steps to a batch of images:
    - Median filtering (optional)
    - Thresholding (using a provided value or the triangle method)
    - Centering (optional, using an image fitter function)
    - Cropping (optional, based on fitted centroid and RMS size)
    - Normalization (independent or max intensity image)

    If max_pixels is not None, images are subsampled / pooled to ensure
    they contain at most max_pixels pixels. This ignores values specified by
    `pool_size`.

    Parameters
    ----------
    images : np.ndarray
        Batch of images with shape (..., height (y size), width (x size)).
    pixel_size : float
        Pixel size of the screen in microns.
    image_fitter : Callable, optional
        Function that fits an image and returns (centroid, rms) in pixel coordinates.
    pool_size : int, optional
        Size of the pooling window. If None, no pooling is applied.
    max_pixels : int, optional
        Maximum number of pixels per image after pooling. If specified, `pool_size` is ignored.
    median_filter_size : int, optional
        Size of the median filter. If None, no median filter is applied.
    threshold : float, optional
        Threshold to apply before filtering. If None, calculated via triangle method.
    threshold_multiplier : float, optional
        Multiplier for the threshold value. Default is 1.0.
    n_stds : int, optional
        Number of standard deviations for cropping. Default is 8.
    normalization : Literal["independent", "max_intensity_image"], optional
        Normalization method. Default is "independent".
    center : bool, optional
        If True, center images using the image fitter. Default is False.
    crop : bool, optional
        If True, crop images using fitted centroid and RMS size. Default is False.
    image_centroids : np.ndarray, optional
        Precomputed centroids for centering. If None, computed internally.
    crop_ranges : np.ndarray, optional
        Precomputed crop ranges. If None, computed internally.
    visualize : bool, optional
        If True, visualize images at each processing step. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
            - "images": processed images (np.ndarray)
            - "pixel_size": pixel size after pooling (float)
            - "centroids": centroids used for centering (np.ndarray)
            - "crop_ranges": crop ranges used for cropping (np.ndarray)
    """

    batch_shape = images.shape[:-2]
    batch_dims = tuple(range(len(batch_shape)))

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
    if center:
        if image_centroids is None:
            image_centroids = calc_image_centroids(images, image_fitter=image_fitter)
        centered_images = center_images(images, image_centroids, visualize=visualize)
    else:
        centered_images = images

    # crop the images
    if crop:
        if crop_ranges is None:
            crop_ranges = calc_crop_ranges(
                centered_images,
                n_stds=n_stds,
                image_fitter=image_fitter,
                visualize=visualize,
            )

        cropped_images = crop_images(
            centered_images,
            crop_ranges=crop_ranges,
            visualize=visualize,
        )
    else:
        cropped_images = centered_images

    # pooling
    if max_pixels is not None:
        pooled_images, pool_size, subsample_idx = adaptive_reduce(
            cropped_images, max_elems=max_pixels
        )
        pixel_size = pixel_size * pool_size
    elif pool_size is not None:
        pooled_images = pool_images(cropped_images, pool_size=pool_size)
        pixel_size = pixel_size * pool_size
        subsample_idx = np.arange(cropped_images.shape[-3])
    else:
        pooled_images = cropped_images
        pixel_size = pixel_size
        subsample_idx = np.arange(cropped_images.shape[-3])

    # normalize image intensities
    normalized_images = normalize_images(pooled_images, normalization=normalization)

    post_processing_results = {
        "images": normalized_images,
        "pixel_size": pixel_size,
        "centroids": image_centroids,
        "crop_ranges": crop_ranges,
        "subsample_idx": subsample_idx,
    }
    return post_processing_results


def mask_and_normalize(
    images,
    masks,
    normalization: Literal["independent", "max_intensity_image"] = "independent",
):
    """
    Apply a mask to input images and normalize the result.

    Parameters
    ----------
    images : np.ndarray
        Array of images to be masked and normalized. Shape should be (..., height (y size), width (x size)).
    masks : np.ndarray
        Array of mask to apply to the images. Non-zero values indicate regions to keep. Shape should match images.
    normalization : Literal["independent", "max_intensity_image"], optional
        Method for normalizing the masked images.

    Returns
    -------
    np.ndarray
        Masked and normalized images.
    """
    if images.shape != masks.shape:
        raise ValueError("Images and masks must have the same shape.")
    masked_images = images * (masks > 0)
    return normalize_images(masked_images, normalization=normalization)
