from unittest.mock import Mock
import pytest
import torch
from cheetah import Screen
from gpsr.datasets import (
    ObservableDataset,
    QuadScanDataset,
    SixDReconstructionDataset,
)


class TestDatasets:
    def test_observable_dataset_initialization(self):
        # Valid initialization
        parameters = torch.rand((3, 2, 5))  # B = 3, M = 2, N = 5
        observations = tuple((torch.rand((3, 200, 150)), torch.rand((3, 150, 100))))
        screen1 = Mock(Screen)
        screen2 = Mock(Screen)
        screen1.resolution = (observations[0].shape[-1], observations[0].shape[-2])
        screen2.resolution = (observations[1].shape[-1], observations[1].shape[-2])
        screens = (screen1, screen2)
        dataset = ObservableDataset(parameters, observations, screens)

        assert dataset.parameters.shape == parameters.shape
        assert len(dataset.observations) == len(observations)

        # Invalid observations
        with pytest.raises(ValueError):
            ObservableDataset(
                parameters, torch.rand((3, 200, 150)), screens
            )  # Not a tuple

    def test_observable_dataset_len(self):
        parameters = torch.rand((3, 2, 5))  # B = 3,, M = 2, N = 5
        observations = (torch.rand((3, 200, 150)), torch.rand((3, 150, 100)))
        screen1 = Mock(Screen)
        screen2 = Mock(Screen)
        screen1.resolution = (observations[0].shape[-1], observations[0].shape[-2])
        screen2.resolution = (observations[1].shape[-1], observations[1].shape[-2])
        screens = (screen1, screen2)
        dataset = ObservableDataset(parameters, observations, screens)

        assert len(dataset) == 3

    def test_observable_dataset_getitem(self):
        parameters = torch.rand((3, 2, 5))  # B = 3,, M = 2, N = 5
        observations = (torch.rand((3, 200, 150)), torch.rand((3, 150, 100)))
        screen1 = Mock(Screen)
        screen2 = Mock(Screen)
        screen1.resolution = (observations[0].shape[-1], observations[0].shape[-2])
        screen2.resolution = (observations[1].shape[-1], observations[1].shape[-2])
        screens = (screen1, screen2)
        dataset = ObservableDataset(parameters, observations, screens)

        sample = dataset[1]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape == (2, 5)
        assert len(sample[1]) == 2
        assert sample[1][0].shape == (200, 150)

    def test_four_d_reconstruction_dataset_initialization(self):
        parameters = torch.rand((5, 3))  # K = 5, N = 3
        observations = tuple(
            [torch.rand((5, 150, 100))]
        )  # K = 5, bins x bins = 100 x 100
        screen1 = Mock(Screen)
        screen1.resolution = (observations[0].shape[-1], observations[0].shape[-2])
        screens = (screen1,)
        dataset = QuadScanDataset(parameters, observations, screens)

        assert dataset.parameters.shape == (5, 3)
        assert dataset.observations[0].shape == (5, 150, 100)

    def test_six_d_reconstruction_dataset_initialization(self):
        parameters = torch.rand((5, 2, 2, 3))  # (n_g, n_v, n_k, n_params)
        observations = (
            torch.rand((5, 2, 200, 150)),
            torch.rand((5, 2, 150, 100)),
        )
        screen1 = Mock(Screen)
        screen2 = Mock(Screen)
        screen1.resolution = (observations[0].shape[-1], observations[0].shape[-2])
        screen2.resolution = (observations[1].shape[-1], observations[1].shape[-2])
        screens = (screen1, screen2)
        dataset = SixDReconstructionDataset(parameters, observations, screens)

        assert dataset.parameters.shape == (10, 2, 3)
        assert len(dataset.observations) == 2
        assert dataset.observations[0].shape == (10, 200, 150)

        # Invalid initialization
        with pytest.raises(ValueError):
            SixDReconstructionDataset(torch.rand((3, 3, 5, 3)), observations, screens)
