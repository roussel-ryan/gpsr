import matplotlib.pyplot as plt
import pytest
import torch
from torch import Size

from phase_space_reconstruction.histogram import marginal_pdf, histogram2d


class TestHistogram:
    def test_marginal_pdf_basic(self):
        # test basic usage
        data = torch.randn((5, 100, 1))  # 5 beamline states, 100 particles in 1D
        bins = torch.linspace(0, 1, 10)  # a single histogram
        sigma = torch.tensor(0.1)  # a single bandwidth

        pdf, _ = marginal_pdf(data, bins, sigma)

        assert pdf.shape == Size([5, 10])  # 5 histograms at 10 points

        # test bad bins
        with pytest.raises(ValueError):
            marginal_pdf(data, bins, torch.rand(3) + 0.1)

    def test_marginal_pdf_batched(self):
        data = torch.randn((5, 100, 1))  # 5 beamline states, 100 particles in 1D

        bins = torch.linspace(0, 1, 10).unsqueeze(dim=0).repeat((5, 1))
        bins[0] = torch.linspace(-1, 1, 10)
        assert bins.shape == Size([5, 10])  # 5 histograms at 10 points (n histograms
        # must equal the batch shape of the particle data

        sigma = torch.rand(5) + 0.1  # a single bandwidth

        pdf, _ = marginal_pdf(data, bins, sigma)
        assert pdf.shape == Size([5, 10])  # 5 histograms at 10 points

        # test bad sigma arg
        with pytest.raises(ValueError):
            marginal_pdf(data, bins, torch.rand(3) + 0.1)

        # test bad bins arg
        with pytest.raises(ValueError):
            marginal_pdf(
                data,
                torch.linspace(0, 1, 10).unsqueeze(dim=0).repeat((3, 1)),
                torch.tensor(1.0)
            )

    def test_histogram_2d_batched(self):
        data = torch.randn((5, 10000, 2))  # 5 beamline states, 100 particles in 1D

        bins = torch.linspace(0, 1, 10).unsqueeze(dim=0).repeat((5, 1))
        bins[0] = torch.linspace(-1, 1, 10)
        bins[-1] = torch.linspace(-10, 10, 10)

        assert bins.shape == Size([5, 10])
        sigma = torch.tensor(0.1)  # a single bandwidth

        pdf = histogram2d(
            data[..., 0],
            data[..., 1],
            bins,
            bins,
            sigma
        )

        assert pdf.shape == Size([5, 10, 10])
        for ele in pdf:
            assert torch.isclose(ele.sum(), torch.tensor(1.0))
            #fig, ax = plt.subplots()
            #ax.imshow(ele)

        # test bad bins
        with pytest.raises(ValueError):
            bins = torch.randn(6, 5)
            histogram2d(
                data[..., 0],
                data[..., 1],
                bins,
                bins,
                sigma
            )


