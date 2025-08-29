import numpy as np
from gpsr.data_processing import process_images
import pytest


class TestProcessImages:
    image_shapes = [
        (5, 100, 100),
        (10, 10, 100, 100),
        (5, 2, 10, 100, 100),
        (20, 100, 100),
        (1, 7, 7)
    ]
    rms_size = np.array([5, 5])
    centroid = np.array([55, 55])

    def mock_image_fitter(self, image):
        # Mock image fitter that returns fixed centroid and rms size
        return self.centroid, self.rms_size

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_no_pooling_no_filter(self, image_shape):
        images = np.random.rand(*image_shape)
        screen_resolution = 1.0
        processed_images_dict = process_images(
            images, screen_resolution, self.mock_image_fitter
        )

        assert processed_images_dict["images"].shape == image_shape

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_crop(self, image_shape):
        images = np.random.rand(*image_shape)
        screen_resolution = 1.0
        processed_images_dict = process_images(
            images, screen_resolution, self.mock_image_fitter, crop=True
        )

        assert processed_images_dict["images"].shape == image_shape[:-2] + tuple(
            8 * 2 * self.rms_size
        )

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_with_pooling(self, image_shape):
        images = np.random.rand(*image_shape)
        screen_resolution = 1.0
        pool_size = 2
        processed_images_dict = process_images(
            images,
            screen_resolution,
            self.mock_image_fitter,
            pool_size=pool_size,
            crop=True,
        )

        assert processed_images_dict["images"].shape == image_shape[:-2] + tuple(
            8 * 2 * self.rms_size // pool_size
        )
        assert processed_images_dict["pixel_size"] == screen_resolution * pool_size

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

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_with_center_images(self, image_shape):
        images = np.zeros(image_shape)
        images[..., 50:60, 50:60] = 1.0

        process_images(
            images,
            pixel_size=1.0,
            image_fitter=self.mock_image_fitter,
            center=True,
        )

    @pytest.mark.parametrize("image_shape", image_shapes)
    def test_process_images_with_adaptive_reduction(self, image_shape):
        images = np.zeros(image_shape)
        images[..., 50:60, 50:60] = 1.0

        processed_images = process_images(
            images,
            pixel_size=1.0,
            image_fitter=self.mock_image_fitter,
            max_pixels=1000,
        )
        pimages = processed_images["images"]
        psize = processed_images["pixel_size"]
        subsample_idx = processed_images["subsample_idx"]

        assert pimages.size <= 1000
        if image_shape == (5, 2, 10, 100, 100):
            assert pimages.shape == (5, 2, 10, 3, 3)
            assert np.isclose(psize, 31.62, rtol=1e-2)
            assert np.all(subsample_idx == np.arange(0, 10))
        elif image_shape == (10, 10, 100, 100):
            assert pimages.shape == (10, 10, 3, 3)
            assert np.isclose(psize, 31.62, rtol=1e-2)
            assert np.all(subsample_idx == np.arange(0, 10))
        elif image_shape == (5, 100, 100):
            assert np.isclose(psize, 7.07, rtol=1e-2)
            assert pimages.shape == (5, 14, 14)
            assert np.all(subsample_idx == np.arange(0, 5))
        elif image_shape == (20, 100, 100):
            assert np.isclose(psize, 10.0, rtol=1e-2)
            assert pimages.shape == (10, 10, 10)
            assert np.all(subsample_idx == np.linspace(0, 19, 10).round().astype(int))
        elif image_shape == (1, 75, 75):
            assert np.isclose(psize, 1.0, rtol=1e-2)
            assert pimages.shape == (1, 75, 75)
            assert np.all(subsample_idx == np.arange(0))
