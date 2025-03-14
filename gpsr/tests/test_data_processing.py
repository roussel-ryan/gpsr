import numpy as np
from gpsr.data_processing import process_images
import pytest


class TestProcessImages:
    image_shapes = [(5, 100, 100), (10, 10, 100, 100), (5, 2, 10, 100, 100)]
    rms_size = np.array([5, 5])
    centroid = np.array([50, 50])

    def mock_image_fitter(self, image):
        # Mock image fitter that returns fixed rms size and centroid
        return self.rms_size, self.centroid

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_no_pooling_no_filter(self, image_shape):
        images = np.random.rand(*image_shape)
        screen_resolution = 1.0
        processed_images, meshgrid = process_images(
            images, screen_resolution, self.mock_image_fitter
        )

        assert processed_images.shape == image_shape[:-2] + tuple(8 * 2 * self.rms_size)
        assert meshgrid[0].shape == tuple(8 * 2 * self.rms_size)
        assert meshgrid[1].shape == tuple(8 * 2 * self.rms_size)

        assert np.allclose(
            meshgrid[0][0],
            np.linspace(
                -8 * self.rms_size[0],
                8 * self.rms_size[0] - 1.0,
                8 * 2 * self.rms_size[0],
            )
            * 1.0e-6,
        )

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_with_pooling(self, image_shape):
        images = np.random.rand(*image_shape)
        screen_resolution = 1.0
        pool_size = 2
        processed_images, meshgrid = process_images(
            images, screen_resolution, self.mock_image_fitter, pool_size=pool_size
        )

        assert processed_images.shape == image_shape[:-2] + tuple(
            8 * 2 * self.rms_size // pool_size
        )
        assert len(meshgrid) == 2
        assert meshgrid[0].shape == tuple(8 * 2 * self.rms_size // pool_size)
        assert meshgrid[1].shape == tuple(8 * 2 * self.rms_size // pool_size)

        size = 8 * 2 * self.rms_size[0] // pool_size
        assert np.allclose(
            meshgrid[0][0],
            pool_size * np.linspace(-size / 2, size / 2 - 1.0, size) * 1.0e-6,
        )

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_with_median_filter(self, image_shape):
        images = np.random.rand(*image_shape)
        screen_resolution = 1.0
        median_filter_size = 3
        process_images(
            images,
            screen_resolution,
            self.mock_image_fitter,
            median_filter_size=median_filter_size,
        )

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_with_threshold(self, image_shape):
        images = np.random.rand(*image_shape)
        screen_resolution = 1.0
        threshold = 0.5
        process_images(
            images, screen_resolution, self.mock_image_fitter, threshold=threshold
        )
