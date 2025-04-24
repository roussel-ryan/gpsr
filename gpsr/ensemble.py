
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from gpsr.modeling import GPSR
from gpsr.train import train_gpsr
from gpsr.beams import NNParticleBeamGenerator, NNTransform
from lightning.pytorch.loggers import CSVLogger

from cheetah.particles import ParticleBeam

def train_ensemble(
    gpsr_lattice,
    train_dataloader,
    n_models=5,
    n_epochs=100,
    lr=1e-3,
    log_name="gpsr_ensemble",
    checkpoint_period_epochs=100,
    **kwargs,
):
    models = []

    for i in range(n_models):
        print(f"Training model {i + 1}/{n_models}...")

        # define logger
        logger = CSVLogger("logs", name=log_name + f"/model_{i}")

        p0c = 43.36e6  # reference momentum in eV/c

        model = train_gpsr(
            GPSR(NNParticleBeamGenerator(
                10000, p0c, transformer=NNTransform(2, 20, output_scale=1e-1)
            ), gpsr_lattice),
            train_dataloader,
            n_epochs=n_epochs,
            lr=lr,
            logger=logger,
            checkpoint_period_epochs=checkpoint_period_epochs,
            **kwargs,
        )
        models.append(model)

    return models


def compute_mean_and_confidence_interval(
    beams: list[ParticleBeam],
    x_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    y_dimension: Literal["x", "px", "y", "py", "tau", "p"],
    bins: int = 100,
    bin_ranges: tuple[tuple[float]] | None = None,
    lower_percentile: float = 0.05,
    upper_percentile: float = 0.95,
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
        histograms.append(histogram)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    histograms = np.array(histograms)
    mean_histogram = np.mean(histograms, axis=0)
    lower_bound = np.percentile(histograms, lower_percentile * 100, axis=0)
    upper_bound = np.percentile(histograms, upper_percentile * 100, axis=0)
    normalized_confidence_width = mean_histogram / (upper_bound - lower_bound)
    normalized_confidence_width[normalized_confidence_width == np.nan] = 0.0

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
        style: Literal["histogram", "contour"] = "histogram",
        bins: int = 100,
        bin_ranges: tuple[tuple[float]] | None = None,
        histogram_smoothing: float = 0.0,
        contour_smoothing: float = 3.0,
        pcolormesh_kws: dict | None = None,
        contour_kws: dict | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

    x_centers, y_centers, mean_histogram, normalized_confidence_width = compute_mean_and_confidence_interval(
        beams,
        x_dimension,
        y_dimension,
        bins=bins,
        bin_ranges=bin_ranges,
    )

    c = ax[0].pcolormesh(
        x_centers,
        y_centers,
        mean_histogram.T,
        shading="auto",
        **(pcolormesh_kws or {}),
    )
    ax[0].set_title("Mean Histogram")
    ax[0].set_xlabel(x_dimension)
    ax[0].set_ylabel(y_dimension)
    fig.colorbar(c, ax=ax[0], label="Counts")
    #ax[0].contour(
    #    x_centers,
    #    y_centers,
    #    normalized_confidence_width.T,
    #    cmap="plasma",
    #    levels=[1.0,2.0],
    #    **(pcolormesh_kws or {}),
    #)
    c = ax[1].pcolor(
        x_centers,
        y_centers,
        normalized_confidence_width.T,
        **(pcolormesh_kws or {}),
    )
    #ax[1].set_title("Normalized Confidence Width")
    #ax[1].set_xlabel(x_dimension)
    fig.colorbar(c, ax=ax[1], label="Normalized Width")


    return fig, ax
        