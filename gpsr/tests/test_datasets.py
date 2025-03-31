from unittest.mock import Mock
import pytest
import torch
from gpsr.custom_cheetah.screen import Screen
from gpsr.datasets import (
    ObservableDataset,
    QuadScanDataset,
    SixDReconstructionDataset,
)


class TestDatasets:
    def test_observable_dataset_initialization(self):
        # Valid initialization
        parameters = torch.rand((3, 2, 5))  # B = 3, M = 2, N = 5
        observations = tuple((torch.rand((3, 200, 200)), torch.rand((3, 150, 150))))
        dataset = ObservableDataset(parameters, observations)

        assert dataset.parameters.shape == parameters.shape
        assert len(dataset.observations) == len(observations)

        # Invalid observations
        with pytest.raises(ValueError):
            ObservableDataset(parameters, torch.rand((3, 200, 200)))  # Not a tuple

    def test_observable_dataset_len(self):
        parameters = torch.rand((3, 2, 5))  # B = 3,, M = 2, N = 5
        observations = (torch.rand((3, 200, 200)), torch.rand((3, 150, 150)))
        dataset = ObservableDataset(parameters, observations)

        assert len(dataset) == 3

    def test_observable_dataset_getitem(self):
        parameters = torch.rand((3, 2, 5))  # B = 3,, M = 2, N = 5
        observations = (torch.rand((3, 200, 200)), torch.rand((3, 150, 150)))
        dataset = ObservableDataset(parameters, observations)

        sample = dataset[1]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape == (2, 5)
        assert len(sample[1]) == 2
        assert sample[1][0].shape == (200, 200)

    def test_four_d_reconstruction_dataset_initialization(self):
        parameters = torch.rand((5, 3))  # K = 5, N = 3
        observations = torch.rand((5, 100, 100))  # K = 5, bins x bins = 100 x 100
        dataset = QuadScanDataset(parameters, observations, screen=Mock(Screen))

        assert dataset.parameters.shape == (5, 3)
        assert dataset.observations[0].shape == (5, 100, 100)

    def test_six_d_reconstruction_dataset_initialization(self):
        parameters = torch.rand((5, 2, 2, 3))  # (n_g, n_v, n_k, n_params)
        observations = (
            torch.rand((5, 2, 100, 100)),
            torch.rand((5, 2, 150, 150)),
        )
        bins = (torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 150))
        dataset = SixDReconstructionDataset(parameters, observations, bins)

        assert dataset.parameters.shape == (10, 2, 3)
        assert len(dataset.observations) == 2
        assert dataset.observations[0].shape == (10, 100, 100)

        # Invalid initialization
        with pytest.raises(ValueError):
            SixDReconstructionDataset(torch.rand((3, 3, 5, 3)), observations, bins)
