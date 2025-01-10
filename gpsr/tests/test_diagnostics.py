import pytest
import torch
from unittest.mock import MagicMock
from cheetah.particles import ParticleBeam
from gpsr.diagnostics import ImageDiagnostic


def test_image_diagnostic_initialization():
    bins_x = torch.linspace(-5, 5, 50)
    bins_y = torch.linspace(-5, 5, 50)
    bandwidth = torch.tensor([1.0])

    diagnostic = ImageDiagnostic(bins_x, bins_y, bandwidth)

    assert diagnostic.x == "x"
    assert diagnostic.y == "y"
    assert torch.equal(diagnostic.bins_x, bins_x)
    assert torch.equal(diagnostic.bins_y, bins_y)
    assert torch.equal(diagnostic.bandwidth, bandwidth)


def test_image_diagnostic_forward_valid_input():
    bins_x = torch.linspace(-5, 5, 50)
    bins_y = torch.linspace(-5, 5, 50)
    bandwidth = torch.tensor(1.0)

    diagnostic = ImageDiagnostic(bins_x, bins_y, bandwidth)

    # Mock the ParticleBeam object
    beam = MagicMock(spec=ParticleBeam)
    beam.x = torch.randn(10, 100)  # Batch of 10 samples, each with 100 points
    beam.y = torch.randn(10, 100)

    result = diagnostic.forward(beam)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 50, 50)


def test_image_diagnostic_forward_mismatched_shapes():
    bins_x = torch.linspace(-5, 5, 50)
    bins_y = torch.linspace(-5, 5, 50)
    bandwidth = torch.tensor(1.0)

    diagnostic = ImageDiagnostic(bins_x, bins_y, bandwidth)

    beam = MagicMock(spec=ParticleBeam)
    beam.x = torch.randn(10, 100)
    beam.y = torch.randn(10, 101)  # Mismatched shape

    with pytest.raises(ValueError, match="x,y coords must be the same shape"):
        diagnostic.forward(beam)


def test_image_diagnostic_forward_insufficient_dimensions():
    bins_x = torch.linspace(-5, 5, 50)
    bins_y = torch.linspace(-5, 5, 50)
    bandwidth = torch.tensor(1.0)

    diagnostic = ImageDiagnostic(bins_x, bins_y, bandwidth)

    beam = MagicMock(spec=ParticleBeam)
    beam.x = torch.randn(100)  # 1D tensor
    beam.y = torch.randn(100)

    with pytest.raises(ValueError, match="coords must be at least 2D"):
        diagnostic.forward(beam)


def test_image_diagnostic_forward_with_non_default_axes():
    bins_x = torch.linspace(-5, 5, 50)
    bins_y = torch.linspace(-5, 5, 50)
    bandwidth = torch.tensor(1.0)

    diagnostic = ImageDiagnostic(bins_x, bins_y, bandwidth, x="z", y="px")

    beam = MagicMock(spec=ParticleBeam)
    beam.z = torch.randn(10, 100)
    beam.px = torch.randn(10, 100)

    result = diagnostic.forward(beam)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 50, 50)
