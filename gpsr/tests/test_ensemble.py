import pytest
import torch
import numpy as np
import copy
from unittest.mock import Mock, patch, MagicMock
from gpsr.ensemble import (
    reinitialize_weights,
    train_ensemble,
    train_ensemble_distributed,
    compute_mean_and_confidence_interval,
    compute_distribution_statistics,
    plot_2d_distribution,
    _train_single_model_thread,
)
from gpsr.modeling import GPSR
from gpsr.beams import NNParticleBeamGenerator
from cheetah.particles import ParticleBeam
import matplotlib.pyplot as plt


@pytest.fixture
def device_list():
    """Fixture to provide available devices for testing"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    return devices


@pytest.fixture
def mock_gpsr_model():
    """Create a mock GPSR model for testing"""
    mock_model = Mock(spec=GPSR)
    mock_model.apply = Mock()
    return mock_model


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing"""
    mock_loader = Mock()
    mock_loader.__iter__ = Mock(
        return_value=iter([(torch.randn(2, 4), [torch.randn(2, 100, 100)])])
    )
    return mock_loader


@pytest.fixture
def sample_histograms():
    """Create sample histogram data for testing"""
    np.random.seed(42)
    # Create 5 histograms of shape (10, 10)
    histograms = torch.tensor(np.random.rand(5, 10, 10))
    return histograms


@pytest.fixture
def sample_particle_beams():
    """Create sample ParticleBeam objects for testing"""
    beams = []
    for i in range(3):
        beam = Mock(spec=ParticleBeam)
        beam.x = torch.randn(1000) * 0.001 + i * 0.0001
        beam.px = torch.randn(1000) * 0.001 + i * 0.0001
        beam.y = torch.randn(1000) * 0.001 + i * 0.0001
        beam.py = torch.randn(1000) * 0.001 + i * 0.0001
        beam.tau = torch.randn(1000) * 0.001
        beam.p = torch.ones(1000) * 1e6 + torch.randn(1000) * 1000
        beams.append(beam)
    return beams


class TestReinitializeWeights:
    """Test the reinitialize_weights function"""

    def test_reinitialize_linear_layer(self):
        """Test reinitializing a linear layer"""
        layer = torch.nn.Linear(10, 5)
        original_weight = layer.weight.clone()
        original_bias = layer.bias.clone()

        reinitialize_weights(layer)

        # Weights should be different after reinitialization
        assert not torch.allclose(original_weight, layer.weight)
        assert not torch.allclose(original_bias, layer.bias)
        # Bias should be zeros
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_reinitialize_conv_layer(self):
        """Test reinitializing a convolutional layer"""
        layer = torch.nn.Conv2d(3, 16, 3)
        original_weight = layer.weight.clone()
        original_bias = layer.bias.clone()

        reinitialize_weights(layer)

        assert not torch.allclose(original_weight, layer.weight)
        assert not torch.allclose(original_bias, layer.bias)
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_reinitialize_layer_with_reset_parameters(self):
        """Test reinitializing a layer that has reset_parameters method"""
        layer = Mock()
        layer.reset_parameters = Mock()

        reinitialize_weights(layer)

        layer.reset_parameters.assert_called_once()

    def test_reinitialize_layer_without_weights(self):
        """Test reinitializing a layer without weights (should not raise error)"""
        layer = torch.nn.ReLU()

        # Should not raise an exception
        reinitialize_weights(layer)


class TestTrainEnsemble:
    """Test the train_ensemble function"""

    @patch("gpsr.ensemble.train_gpsr")
    @patch("gpsr.ensemble.CSVLogger")
    def test_train_ensemble_sequential_cpu(
        self, mock_logger, mock_train_gpsr, mock_gpsr_model, mock_dataloader
    ):
        """Test sequential training on CPU"""
        # Mock train_gpsr to return a model
        mock_trained_model = Mock()
        mock_train_gpsr.return_value = mock_trained_model

        models = train_ensemble(
            mock_gpsr_model,
            mock_dataloader,
            n_models=2,
            n_epochs=5,
            parallel_training=False,
        )

        assert len(models) == 2
        assert all(model == mock_trained_model for model in models)
        assert mock_train_gpsr.call_count == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Multiple GPUs not available"
    )
    @patch("gpsr.ensemble._train_single_model_thread")
    def test_train_ensemble_parallel_gpu(
        self, mock_train_thread, mock_gpsr_model, mock_dataloader
    ):
        """Test parallel training with multiple GPUs"""

        # Mock the thread function to simulate successful training
        def mock_thread_func(model_idx, *args, **kwargs):
            results_dict = args[6]  # results_dict is the 7th argument
            results_dict[model_idx] = Mock()  # Mock trained model

        mock_train_thread.side_effect = mock_thread_func

        models = train_ensemble(
            mock_gpsr_model,
            mock_dataloader,
            n_models=2,
            n_epochs=5,
            parallel_training=True,
        )

        assert len(models) == 2
        assert mock_train_thread.call_count == 2

    @patch("gpsr.ensemble.train_gpsr")
    @patch("gpsr.ensemble.CSVLogger")
    def test_train_ensemble_auto_mode_single_gpu(
        self, mock_logger, mock_train_gpsr, mock_gpsr_model, mock_dataloader
    ):
        """Test auto mode with single GPU (should fall back to sequential)"""
        mock_trained_model = Mock()
        mock_train_gpsr.return_value = mock_trained_model

        with patch("torch.cuda.device_count", return_value=1):
            models = train_ensemble(
                mock_gpsr_model,
                mock_dataloader,
                n_models=2,
                n_epochs=5,
                parallel_training="auto",
            )

        assert len(models) == 2
        assert mock_train_gpsr.call_count == 2

    def test_train_ensemble_invalid_model_type(self, mock_dataloader):
        """Test that invalid model type raises ValueError"""
        invalid_model = "not_a_gpsr_model"

        with pytest.raises(ValueError, match="gpsr_model must be an instance of GPSR"):
            train_ensemble(
                invalid_model, mock_dataloader, n_models=1, parallel_training=False
            )

    @pytest.mark.skipif(torch.cuda.is_available(), reason="Test for CUDA not available")
    def test_train_ensemble_parallel_no_cuda(self, mock_gpsr_model, mock_dataloader):
        """Test parallel training fails when CUDA is not available"""
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            train_ensemble(
                mock_gpsr_model, mock_dataloader, n_models=2, parallel_training=True
            )


class TestTrainEnsembleDistributed:
    """Test the train_ensemble_distributed function"""

    @patch("gpsr.ensemble.train_gpsr")
    @patch("gpsr.ensemble.CSVLogger")
    @patch("gpsr.ensemble.GPSR")
    @patch("gpsr.ensemble.NNParticleBeamGenerator")
    def test_train_ensemble_distributed(
        self, mock_beam_gen, mock_gpsr, mock_logger, mock_train_gpsr, mock_dataloader
    ):
        """Test distributed ensemble training"""
        mock_lattice = Mock()
        mock_trained_model = Mock()
        mock_train_gpsr.return_value = mock_trained_model
        mock_gpsr.return_value = Mock()

        models = train_ensemble_distributed(
            mock_lattice, mock_dataloader, n_models=2, n_epochs=5
        )

        assert len(models) == 2
        assert all(model == mock_trained_model for model in models)
        assert mock_train_gpsr.call_count == 2


class TestComputeMeanAndConfidenceInterval:
    """Test the compute_mean_and_confidence_interval function"""

    def test_compute_mean_and_confidence_interval(self, sample_histograms):
        """Test computing mean and confidence interval"""
        mean_hist, conf_width = compute_mean_and_confidence_interval(
            sample_histograms, lower_quantile=0.1, upper_quantile=0.9
        )

        # Check shapes
        assert mean_hist.shape == (10, 10)
        assert conf_width.shape == (10, 10)

        # Check that mean is approximately correct
        expected_mean = torch.mean(sample_histograms, axis=0)
        assert torch.allclose(mean_hist, expected_mean)

        # Check that confidence width values are reasonable
        assert torch.all(conf_width >= 0)

    def test_compute_mean_and_confidence_interval_edge_cases(self):
        """Test edge cases for mean and confidence interval computation"""
        # Single histogram
        single_hist = torch.ones(1, 5, 5)
        mean_hist, conf_width = compute_mean_and_confidence_interval(single_hist)

        assert torch.allclose(mean_hist, torch.ones(5, 5))
        # Confidence width should be inf when lower and upper bounds are the same
        assert torch.all(torch.isinf(conf_width) | (conf_width == 0))

    def test_compute_mean_and_confidence_interval_different_quantiles(
        self, sample_histograms
    ):
        """Test different quantile values"""
        mean_hist1, conf_width1 = compute_mean_and_confidence_interval(
            sample_histograms, 0.05, 0.95
        )
        mean_hist2, conf_width2 = compute_mean_and_confidence_interval(
            sample_histograms, 0.25, 0.75
        )

        # Mean should be the same
        assert torch.allclose(mean_hist1, mean_hist2)
        # Confidence width should be different (narrower range should have larger normalized width)
        assert not torch.allclose(conf_width1, conf_width2)


class TestComputeDistributionStatistics:
    """Test the compute_distribution_statistics function"""

    def test_compute_distribution_statistics(self, sample_particle_beams):
        """Test computing distribution statistics from particle beams"""
        x_centers, y_centers, mean_hist, conf_width = compute_distribution_statistics(
            sample_particle_beams, x_dimension="x", y_dimension="px", bins=20
        )

        # Check output shapes
        assert len(x_centers) == 20
        assert len(y_centers) == 20
        assert mean_hist.shape == (20, 20)
        assert conf_width.shape == (20, 20)

        # Check that centers are reasonable
        assert x_centers[0] < x_centers[-1]  # Should be increasing
        assert y_centers[0] < y_centers[-1]  # Should be increasing

    def test_compute_distribution_statistics_with_bin_ranges(
        self, sample_particle_beams
    ):
        """Test computing distribution statistics with specified bin ranges"""
        bin_ranges = ((-0.01, 0.01), (-0.01, 0.01))

        x_centers, y_centers, mean_hist, conf_width = compute_distribution_statistics(
            sample_particle_beams,
            x_dimension="y",
            y_dimension="py",
            bins=10,
            bin_ranges=bin_ranges,
        )

        # Check that centers are within specified ranges
        assert x_centers[0] >= bin_ranges[0][0]
        assert x_centers[-1] <= bin_ranges[0][1]
        assert y_centers[0] >= bin_ranges[1][0]
        assert y_centers[-1] <= bin_ranges[1][1]

    def test_compute_distribution_statistics_with_smoothing(
        self, sample_particle_beams
    ):
        """Test computing distribution statistics with smoothing"""
        x_centers1, y_centers1, mean_hist1, conf_width1 = (
            compute_distribution_statistics(
                sample_particle_beams,
                x_dimension="tau",
                y_dimension="p",
                bins=15,
                smoothing_factor=None,
            )
        )

        x_centers2, y_centers2, mean_hist2, conf_width2 = (
            compute_distribution_statistics(
                sample_particle_beams,
                x_dimension="tau",
                y_dimension="p",
                bins=15,
                smoothing_factor=1.0,
            )
        )

        # Centers should be the same
        assert np.allclose(x_centers1, x_centers2)
        assert np.allclose(y_centers1, y_centers2)

        # Histograms should be different due to smoothing
        assert not torch.allclose(mean_hist1, mean_hist2)

    def test_compute_distribution_statistics_different_dimensions(
        self, sample_particle_beams
    ):
        """Test with different dimension combinations"""
        dimensions = ["x", "px", "y", "py", "tau", "p"]

        for x_dim in dimensions[:3]:  # Test a few combinations
            for y_dim in dimensions[3:]:
                x_centers, y_centers, mean_hist, conf_width = (
                    compute_distribution_statistics(
                        sample_particle_beams,
                        x_dimension=x_dim,
                        y_dimension=y_dim,
                        bins=5,
                    )
                )

                assert len(x_centers) == 5
                assert len(y_centers) == 5
                assert mean_hist.shape == (5, 5)
                assert conf_width.shape == (5, 5)


class TestPlot2DDistribution:
    """Test the plot_2d_distribution function"""

    def test_plot_2d_distribution_basic(self, sample_particle_beams):
        """Test basic 2D distribution plotting"""
        fig, ax = plot_2d_distribution(
            sample_particle_beams, x_dimension="x", y_dimension="px", bins=10
        )

        assert fig is not None
        assert len(ax) == 2  # Should return two axes
        assert ax[0].get_xlabel() == "x"
        assert ax[0].get_ylabel() == "px"

        plt.close(fig)

    def test_plot_2d_distribution_with_custom_axes(self, sample_particle_beams):
        """Test plotting with custom axes"""
        fig, custom_ax = plt.subplots(1, 2, figsize=(10, 5))

        result_fig, result_ax = plot_2d_distribution(
            sample_particle_beams,
            x_dimension="y",
            y_dimension="py",
            bins=8,
            ax=custom_ax,
        )

        # Should use the provided axes
        assert result_ax is custom_ax

        plt.close(fig)

    def test_plot_2d_distribution_with_kwargs(self, sample_particle_beams):
        """Test plotting with custom keyword arguments"""
        density_kws = {"cmap": "viridis", "alpha": 0.8}
        ci_kws = {"cmap": "plasma"}

        fig, ax = plot_2d_distribution(
            sample_particle_beams,
            x_dimension="tau",
            y_dimension="p",
            bins=6,
            density_kws=density_kws,
            ci_kws=ci_kws,
        )

        assert fig is not None
        assert len(ax) == 2

        plt.close(fig)

    def test_plot_2d_distribution_with_smoothing(self, sample_particle_beams):
        """Test plotting with smoothing factor"""
        fig, ax = plot_2d_distribution(
            sample_particle_beams,
            x_dimension="x",
            y_dimension="y",
            bins=8,
            smoothing_factor=0.5,
        )

        assert fig is not None
        assert len(ax) == 2

        plt.close(fig)


class TestTrainSingleModelThread:
    """Test the _train_single_model_thread function"""

    @patch("gpsr.ensemble.train_gpsr")
    @patch("gpsr.ensemble.CSVLogger")
    @patch("copy.deepcopy")
    def test_train_single_model_thread_success(
        self,
        mock_deepcopy,
        mock_logger,
        mock_train_gpsr,
        mock_gpsr_model,
        mock_dataloader,
    ):
        """Test successful single model training in thread"""
        results_dict = {}
        mock_trained_model = Mock()
        mock_train_gpsr.return_value = mock_trained_model
        mock_deepcopy.return_value = mock_gpsr_model

        _train_single_model_thread(
            model_idx=0,
            gpsr_model=mock_gpsr_model,
            train_dataloader=mock_dataloader,
            n_epochs=5,
            lr=1e-3,
            log_name="test",
            checkpoint_period_epochs=100,
            results_dict=results_dict,
            gpu_id=None,
        )

        assert 0 in results_dict
        assert results_dict[0] == mock_trained_model

    @patch("gpsr.ensemble.train_gpsr")
    @patch("gpsr.ensemble.CSVLogger")
    @patch("copy.deepcopy")
    def test_train_single_model_thread_exception(
        self,
        mock_deepcopy,
        mock_logger,
        mock_train_gpsr,
        mock_gpsr_model,
        mock_dataloader,
    ):
        """Test exception handling in single model training thread"""
        results_dict = {}
        test_exception = Exception("Test exception")
        mock_train_gpsr.side_effect = test_exception
        mock_deepcopy.return_value = mock_gpsr_model

        _train_single_model_thread(
            model_idx=1,
            gpsr_model=mock_gpsr_model,
            train_dataloader=mock_dataloader,
            n_epochs=5,
            lr=1e-3,
            log_name="test",
            checkpoint_period_epochs=100,
            results_dict=results_dict,
            gpu_id=None,
        )

        assert 1 in results_dict
        assert isinstance(results_dict[1], Exception)
        assert str(results_dict[1]) == "Test exception"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("gpsr.ensemble.train_gpsr")
    @patch("gpsr.ensemble.CSVLogger")
    @patch("copy.deepcopy")
    def test_train_single_model_thread_with_gpu(
        self,
        mock_deepcopy,
        mock_logger,
        mock_train_gpsr,
        mock_gpsr_model,
        mock_dataloader,
    ):
        """Test single model training with GPU specification"""
        results_dict = {}
        mock_trained_model = Mock()
        mock_train_gpsr.return_value = mock_trained_model
        mock_deepcopy.return_value = mock_gpsr_model

        _train_single_model_thread(
            model_idx=0,
            gpsr_model=mock_gpsr_model,
            train_dataloader=mock_dataloader,
            n_epochs=5,
            lr=1e-3,
            log_name="test",
            checkpoint_period_epochs=100,
            results_dict=results_dict,
            gpu_id=0,
        )

        assert 0 in results_dict
        assert results_dict[0] == mock_trained_model
        # Check that GPU-related kwargs were added
        call_kwargs = mock_train_gpsr.call_args[1]
        assert call_kwargs["accelerator"] == "gpu"
        assert call_kwargs["devices"] == [0]


class TestEnsemble:
    """Main ensemble test class with comprehensive GPU/CPU testing"""

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_reinitialize_weights_device_compatibility(self, device):
        """Test weight reinitialization works on different devices"""
        layer = torch.nn.Linear(10, 5).to(device)
        original_weight = layer.weight.clone()

        reinitialize_weights(layer)

        assert not torch.allclose(original_weight, layer.weight)
        assert layer.weight.device.type == device
        assert layer.bias.device.type == device

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_compute_mean_and_confidence_interval_device_compatibility(self, device):
        """Test statistics computation on different devices"""
        histograms = torch.randn(5, 10, 10).to(device)

        mean_hist, conf_width = compute_mean_and_confidence_interval(histograms)

        assert mean_hist.device.type == device
        assert conf_width.device.type == device
        assert mean_hist.shape == (10, 10)
        assert conf_width.shape == (10, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_management(self, sample_histograms):
        """Test that GPU memory is properly managed during computation"""
        if torch.cuda.is_available():
            # Move to GPU
            histograms_gpu = sample_histograms.cuda()

            # Clear cache
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Compute statistics
            mean_hist, conf_width = compute_mean_and_confidence_interval(histograms_gpu)

            # Check results are on GPU
            assert mean_hist.is_cuda
            assert conf_width.is_cuda

            # Memory should not have leaked excessively
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024

    def test_large_histogram_computation(self):
        """Test computation with large histograms"""
        # Create large histograms
        large_histograms = torch.randn(20, 200, 200)

        mean_hist, conf_width = compute_mean_and_confidence_interval(large_histograms)

        assert mean_hist.shape == (200, 200)
        assert conf_width.shape == (200, 200)
        assert torch.all(torch.isfinite(mean_hist))

    def test_single_histogram_edge_case(self):
        """Test edge case with single histogram"""
        single_hist = torch.ones(1, 5, 5) * 2.0

        mean_hist, conf_width = compute_mean_and_confidence_interval(single_hist)

        assert torch.allclose(mean_hist, torch.ones(5, 5) * 2.0)
        # With single histogram, quantiles are identical, so division by zero
        # Results in inf or nan values
        assert torch.all(
            torch.isinf(conf_width) | torch.isnan(conf_width) | (conf_width == 0)
        )

    def test_extreme_quantiles(self, sample_histograms):
        """Test with extreme quantile values"""
        mean_hist, conf_width = compute_mean_and_confidence_interval(
            sample_histograms, lower_quantile=0.01, upper_quantile=0.99
        )

        assert mean_hist.shape == (10, 10)
        assert conf_width.shape == (10, 10)
        assert torch.all(torch.isfinite(mean_hist))

    @pytest.mark.parametrize("n_beams,n_particles", [(1, 100), (5, 1000), (10, 500)])
    def test_distribution_statistics_scaling(self, n_beams, n_particles):
        """Test distribution statistics with different beam/particle counts"""
        beams = []
        for i in range(n_beams):
            beam = Mock(spec=ParticleBeam)
            beam.x = torch.randn(n_particles) * 0.001
            beam.px = torch.randn(n_particles) * 0.001
            beam.y = torch.randn(n_particles) * 0.001
            beam.py = torch.randn(n_particles) * 0.001
            beams.append(beam)

        x_centers, y_centers, mean_hist, conf_width = compute_distribution_statistics(
            beams, x_dimension="x", y_dimension="px", bins=20
        )

        assert len(x_centers) == 20
        assert len(y_centers) == 20
        assert mean_hist.shape == (20, 20)
        assert conf_width.shape == (20, 20)

    def test_thread_safety_simulation(self, mock_gpsr_model, mock_dataloader):
        """Simulate thread safety by running multiple operations"""
        results_dict = {}

        # Simulate multiple threads writing to results_dict
        for i in range(5):
            with patch("gpsr.ensemble.train_gpsr") as mock_train:
                mock_train.return_value = Mock()

                # This simulates what happens in the threading version
                _train_single_model_thread(
                    model_idx=i,
                    gpsr_model=mock_gpsr_model,
                    train_dataloader=mock_dataloader,
                    n_epochs=1,
                    lr=1e-3,
                    log_name="test",
                    checkpoint_period_epochs=100,
                    results_dict=results_dict,
                    gpu_id=None,
                )

        # All results should be stored
        assert len(results_dict) == 5
        for i in range(5):
            assert i in results_dict

    def test_reproducibility_with_seeds(self):
        """Test that setting seeds produces reproducible results"""
        # Test with identical histograms and seeds
        torch.manual_seed(42)
        histograms1 = torch.randn(3, 10, 10)
        mean1, conf1 = compute_mean_and_confidence_interval(histograms1)

        torch.manual_seed(42)
        histograms2 = torch.randn(3, 10, 10)
        mean2, conf2 = compute_mean_and_confidence_interval(histograms2)

        # Should be identical due to same seed
        assert torch.allclose(mean1, mean2)
        assert torch.allclose(conf1, conf2)


class TestEnsembleIntegration:
    """Integration tests for ensemble functionality"""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing"""
        from gpsr.datasets import QuadScanDataset

        # Create dummy data
        parameters = torch.randn(10, 2)
        observations = (torch.randn(10, 20, 20),)

        # Mock screen
        mock_screen = type(
            "MockScreen",
            (),
            {
                "method": "charge_deposition",
                "bins": (20, 20),
                "x_range": (-1, 1),
                "y_range": (-1, 1),
            },
        )()

        dataset = QuadScanDataset(parameters, observations, mock_screen)
        return dataset

    @pytest.fixture
    def simple_gpsr_model(self):
        """Create a simple GPSR model for testing"""
        from gpsr.modeling import GPSR
        from gpsr.beams import NNParticleBeamGenerator

        p0c = 43.36e6
        beam_gen = NNParticleBeamGenerator(100, p0c, n_dim=4)

        # Mock lattice
        mock_lattice = type("MockLattice", (), {"l_quad": 0.1, "l_drift": 1.0})()

        # Create minimal GPSR model
        model = GPSR(beam_gen, mock_lattice)
        return model

    def test_reinitialize_weights_on_real_model(self, simple_gpsr_model):
        """Test weight reinitialization on a real GPSR model"""
        # Get original weights
        original_weights = []
        for param in simple_gpsr_model.parameters():
            original_weights.append(param.clone())

        # Reinitialize
        simple_gpsr_model.apply(reinitialize_weights)

        # Check that weights changed
        weight_changed = False
        for original, current in zip(original_weights, simple_gpsr_model.parameters()):
            if not torch.allclose(original, current):
                weight_changed = True
                break

        assert weight_changed, "Weights should have changed after reinitialization"

    def test_compute_statistics_with_real_particle_beams(self):
        """Test statistics computation with real ParticleBeam objects"""
        # Create real particle beams with different distributions
        beams = []
        for i in range(3):
            # Create particles with slightly different distributions
            n_particles = 500
            x = torch.randn(n_particles) * 0.001 + i * 0.0005
            px = torch.randn(n_particles) * 0.001
            y = torch.randn(n_particles) * 0.001 + i * 0.0005
            py = torch.randn(n_particles) * 0.001
            tau = torch.zeros(n_particles)
            p = torch.ones(n_particles) * 1e6
            particles = torch.stack(
                [x, px, y, py, tau, p, torch.ones(n_particles)], dim=1
            )

            beam = ParticleBeam(particles, energy=torch.tensor(1e6))
            beams.append(beam)

        # Test computation
        x_centers, y_centers, mean_hist, conf_width = compute_distribution_statistics(
            beams, x_dimension="x", y_dimension="px", bins=15
        )

        # Validate results
        assert len(x_centers) == 15
        assert len(y_centers) == 15
        assert mean_hist.shape == (15, 15)
        assert conf_width.shape == (15, 15)

        # Check that the histogram captures the distribution differences
        assert torch.sum(mean_hist) > 0, "Mean histogram should have non-zero values"
        assert torch.all(torch.nan_to_num(conf_width, nan=0.1) >= 0), (
            "Non-nan confidence width should be non-negative"
        )

    def test_mean_confidence_interval_consistency(self):
        """Test that mean and confidence interval computation is consistent"""
        # Create histograms with known statistics
        n_hist = 10
        hist_size = (20, 20)

        # Create identical histograms (no variance)
        identical_hists = torch.ones(n_hist, *hist_size)
        mean_hist, conf_width = compute_mean_and_confidence_interval(identical_hists)

        # Mean should be 1, confidence width should be inf (due to zero variance)
        assert torch.allclose(mean_hist, torch.ones(hist_size))
        assert torch.all(torch.isinf(conf_width) | (conf_width == 0))

        # Create histograms with known variance
        varying_hists = torch.randn(n_hist, *hist_size) + 2.0
        mean_hist, conf_width = compute_mean_and_confidence_interval(
            varying_hists, lower_quantile=0.25, upper_quantile=0.75
        )

        # Mean should be around 2, confidence width should be finite
        assert torch.abs(torch.mean(mean_hist) - 2.0) < 0.5
        assert torch.all(torch.isfinite(conf_width))
        assert torch.all(conf_width > 0)

    @pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.device_count() > 1,
        reason="Skip when multiple GPUs available to avoid parallel training",
    )
    def test_ensemble_training_deterministic_behavior(
        self, simple_gpsr_model, simple_dataset
    ):
        """Test that ensemble training produces deterministic results with fixed seeds"""
        # This is a simplified test since actual training requires more setup
        # We'll test the model copying and reinitialization behavior

        dataloader = torch.utils.data.DataLoader(simple_dataset, batch_size=2)

        # Create two copies and reinitialize
        model1 = copy.deepcopy(simple_gpsr_model)
        model2 = copy.deepcopy(simple_gpsr_model)

        # They should start identical
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())

        for p1, p2 in zip(params1, params2):
            assert torch.allclose(p1, p2), "Copied models should start identical"

        # After reinitialization, they should be different
        model1.apply(reinitialize_weights)
        model2.apply(reinitialize_weights)

        params1_after = list(model1.parameters())
        params2_after = list(model2.parameters())

        # Should be different from each other and from original
        different = False
        for p1, p2 in zip(params1_after, params2_after):
            if not torch.allclose(p1, p2, rtol=1e-6):
                different = True
                break

        assert different, "Reinitialized models should have different weights"


class TestEnsembleErrorHandling:
    """Test error handling and edge cases in ensemble functions"""

    def test_compute_statistics_empty_beams(self):
        """Test error handling with empty beam list"""
        with pytest.raises(ValueError):
            compute_distribution_statistics([], x_dimension="x", y_dimension="px")

    def test_compute_statistics_invalid_dimensions(self):
        """Test error handling with invalid dimension names"""
        # Create a mock beam with minimal attributes
        mock_beam = type(
            "MockBeam", (), {"x": torch.randn(100), "px": torch.randn(100)}
        )()

        with pytest.raises(AttributeError):
            compute_distribution_statistics(
                [mock_beam], x_dimension="invalid_dim", y_dimension="px"
            )

    def test_mean_confidence_interval_invalid_quantiles(self):
        """Test error handling with invalid quantile values"""
        hist = torch.randn(5, 10, 10)

        # Test with invalid quantile range (lower > upper)
        with pytest.raises(ValueError):
            compute_mean_and_confidence_interval(
                hist, lower_quantile=0.9, upper_quantile=0.1
            )

    def test_ensemble_training_invalid_parallel_config(self):
        """Test error handling with invalid parallel training configuration"""
        mock_model = Mock(spec=GPSR)
        mock_dataloader = Mock()

        # Force parallel training when not possible
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                train_ensemble(
                    mock_model, mock_dataloader, n_models=2, parallel_training=True
                )


class TestEnsemblePerformance:
    """Performance and scaling tests for ensemble functions"""

    def test_compute_statistics_scaling(self):
        """Test that statistics computation scales reasonably with input size"""
        import time

        # Create beams of different sizes
        small_beams = []
        large_beams = []

        for i in range(3):
            # Small beams (100 particles)
            small_beam = type(
                "MockBeam",
                (),
                {
                    "x": torch.randn(100),
                    "px": torch.randn(100),
                    "y": torch.randn(100),
                    "py": torch.randn(100),
                    "tau": torch.randn(100),
                    "p": torch.randn(100) + 1e6,
                },
            )()
            small_beams.append(small_beam)

            # Large beams (10000 particles)
            large_beam = type(
                "MockBeam",
                (),
                {
                    "x": torch.randn(10000),
                    "px": torch.randn(10000),
                    "y": torch.randn(10000),
                    "py": torch.randn(10000),
                    "tau": torch.randn(10000),
                    "p": torch.randn(10000) + 1e6,
                },
            )()
            large_beams.append(large_beam)

        # Time small computation
        start_time = time.time()
        compute_distribution_statistics(small_beams, "x", "px", bins=50)
        small_time = time.time() - start_time

        # Time large computation
        start_time = time.time()
        compute_distribution_statistics(large_beams, "x", "px", bins=50)
        large_time = time.time() - start_time

        # Large computation should be slower but not excessively so
        assert large_time > small_time
        assert large_time < small_time * 1000  # Should not be 1000x slower

    def test_memory_usage_reasonable(self):
        """Test that ensemble functions don't use excessive memory"""
        # Create moderate-sized test data
        n_beams = 5
        n_particles = 1000
        beams = []

        for i in range(n_beams):
            beam = type(
                "MockBeam",
                (),
                {
                    "x": torch.randn(n_particles),
                    "px": torch.randn(n_particles),
                    "y": torch.randn(n_particles),
                    "py": torch.randn(n_particles),
                    "tau": torch.randn(n_particles),
                    "p": torch.randn(n_particles) + 1e6,
                },
            )()
            beams.append(beam)

        # This should complete without memory errors
        try:
            x_centers, y_centers, mean_hist, conf_width = (
                compute_distribution_statistics(beams, "x", "px", bins=100)
            )

            # Verify results are reasonable size
            assert len(x_centers) == 100
            assert len(y_centers) == 100
            assert mean_hist.shape == (100, 100)
            assert conf_width.shape == (100, 100)

        except MemoryError:
            pytest.fail("Function used too much memory for reasonable input size")
