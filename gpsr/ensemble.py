from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from gpsr.modeling import GPSR
from gpsr.train import train_gpsr
from gpsr.beams import NNParticleBeamGenerator, NNTransform
from lightning.pytorch.loggers import CSVLogger
from scipy.ndimage import gaussian_filter

from cheetah.particles import ParticleBeam


def train_ensemble(
    gpsr_lattice,
    p0c,
    train_dset,
    n_particles=10000,
    n_models=5,
    n_epochs=100,
    lr=1e-3,
    log_name="gpsr_ensemble",
    checkpoint_period_epochs=100,
    **kwargs,
):
    gpsr_model = GPSR(NNParticleBeamGenerator(n_particles, p0c), gpsr_lattice)
    train_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=len(train_dset))
    for i in range(n_models):
        print(f"Training model {i + 1}/{n_models}...")

        # define logger
        logger = CSVLogger("logs", name=log_name + f"/model_{i}")

        model = train_gpsr(
            model=gpsr_model,
            train_dataloader=train_dataloader,
            n_epochs=n_epochs,
            lr=lr,
            logger=logger,
            checkpoint_period_epochs=checkpoint_period_epochs,
            **kwargs,
        )
        reconstructed_beam = model.gpsr_model.beam_generator()
        torch.save(reconstructed_beam, f"{log_name}_beam_{i}.pt")
        print(f'Saved reconstructed beam to {log_name}_beam_{i}.pt')


def compute_mean_and_confidence_interval(
    histograms: np.ndarray,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an array of histograms, compute the mean and confidence interval over the first axis.

    Parameters
    ----------
    histograms: np.ndarray
        A 2D array of histograms, where each row is a histogram.
    lower_quantile: float, optional
        The lower percentile for the confidence interval. Default is 0.05.
    upper_quantile: float, optional
        The upper percentile for the confidence interval. Default is 0.95.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the mean histogram and the normalized confidence width.

    """
    mean_histogram = torch.mean(histograms, axis=0)
    lower_bound = torch.quantile(histograms, lower_quantile, axis=0)
    upper_bound = torch.quantile(histograms, upper_quantile, axis=0)
    normalized_confidence_width = mean_histogram / (upper_bound - lower_bound)
    normalized_confidence_width[normalized_confidence_width == torch.nan] = 0.0

    return mean_histogram, normalized_confidence_width


def compute_distribution_statistics(
    beams: list[ParticleBeam],
    x_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    y_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    bins: int = 100,
    bin_ranges: tuple[tuple[float]] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    smoothing_factor: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a list of beams, compute the mean and confidence interval for the specified dimensions.

    Parameters
    ----------
    beams: list[ParticleBeam]
        List of ParticleBeam objects.
    x_dimension: str
        The x dimension to compute the mean and confidence interval for.
        Options are "x", "px", "y", "py", "tau", "p".
    y_dimension: str
        The y dimension to compute the mean and confidence interval for.
        Options are "x", "px", "y", "py", "tau", "p".
    bins: int, optional
        Number of bins to use for the histogram. Default is 100.
    bin_ranges: tuple[tuple[float]], optional
        The ranges for the x and y dimensions. Default is None, which means the ranges will be computed from the data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the x and y centers of the histogram, the mean, and the confidence interval.

    """
    # get maximum and minimum values for the x and y dimensions if not provided
    if bin_ranges is None:
        x_min = min([getattr(beam, x_dimension).min().detach() for beam in beams])
        x_max = max([getattr(beam, x_dimension).max().detach() for beam in beams])
        y_min = min([getattr(beam, y_dimension).min().detach() for beam in beams])
        y_max = max([getattr(beam, y_dimension).max().detach() for beam in beams])
        bin_ranges = ((x_min, x_max), (y_min, y_max))

    histograms = []

    for beam in beams:
        # get the histogram for each beam
        histogram, x_edges, y_edges = np.histogram2d(
            getattr(beam, x_dimension).cpu().detach().numpy(),
            getattr(beam, y_dimension).cpu().detach().numpy(),
            bins=bins,
            range=bin_ranges,
        )
        if smoothing_factor is not None:
            histogram = gaussian_filter(histogram, sigma=smoothing_factor)

        histograms.append(histogram)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    mean_histogram, normalized_confidence_width = compute_mean_and_confidence_interval(
        torch.tensor(histograms),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    return (
        x_centers,
        y_centers,
        mean_histogram,
        normalized_confidence_width,
    )


def plot_2d_distribution(
    beams: list[ParticleBeam],
    x_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    y_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    bins: int = 100,
    bin_ranges: tuple[tuple[float]] | None = None,
    ci_kws: dict | None = None,
    density_kws: dict | None = None,
    ax: plt.Axes | None = None,
    smoothing_factor: float = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

    x_centers, y_centers, mean_histogram, normalized_confidence_width = (
        compute_distribution_statistics(
            beams,
            x_dimension,
            y_dimension,
            bins=bins,
            bin_ranges=bin_ranges,
            smoothing_factor=smoothing_factor,
        )
    )

    c = ax[0].pcolormesh(
        x_centers,
        y_centers,
        mean_histogram.T,
        shading="auto",
        **(density_kws or {}),
    )
    ax[0].set_title("Mean Histogram")
    ax[0].set_xlabel(x_dimension)
    ax[0].set_ylabel(y_dimension)
    fig.colorbar(c, ax=ax[0], label="Density")

    c = ax[1].pcolormesh(
        x_centers,
        y_centers,
        normalized_confidence_width.T,
        **(ci_kws or {}),
    )
    fig.colorbar(c, ax=ax[1], label="Normalized CI")

    return fig, ax
