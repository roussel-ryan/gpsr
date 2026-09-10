"""
Ensemble (multi-beam) utilities: build an ensemble beam and compute statistics.

- **Construction** -- :func:`list_to_beam` stacks a list of single Cheetah
  ``ParticleBeam`` s along a leading draw axis into one vectorized ensemble beam
  (``particles`` of shape ``(n_draws, n_particles, 7)``), ready to feed to
  :func:`gpsr.lume.predicting.predict_images` (which tracks it as an ensemble).
- **Statistics** -- vectorized 1D/2D histograms, Gaussian filters, and
  ``compute_statistics_*`` / :func:`compute_mean_and_bounds`, which reduce an
  ensemble of histograms (leading dim indexes the draws) to a mean plus a two-sided
  confidence band. Ported from the ``jp-temp-branch`` of Cheetah's
  ``ensemble_utils.py``.

The statistics back the ensemble corner plots and confidence-band screen images in
:mod:`gpsr.lume.plotting` (``plot_ensemble_distribution``,
``plot_beam_distribution``, ``plot_ensemble_images``), which imports them from
here. Prediction (tracking a beam to observations) lives in
:mod:`gpsr.lume.predicting`.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from scipy import stats

from cheetah.particles import ParticleBeam

UncertaintyType = Literal["percentile", "std_error"]


# ---------------------------------------------------------------------------
# Ensemble beam construction
# ---------------------------------------------------------------------------
def list_to_beam(beam_list: list[ParticleBeam]) -> ParticleBeam:
    """Stack a list of single beams into one vectorized ensemble beam.

    Stacks the per-beam ``particles`` along a new leading draw axis to build an
    ensemble beam with ``particles`` of shape ``(n_draws, n_particles, 7)``, reusing
    the first beam's ``energy``. Only ``particles`` is stacked -- the per-particle
    buffers (``particle_charges`` / ``survival_probabilities``) are left at their
    defaults, which broadcast fine; :func:`gpsr.lume.predicting.predict_images`
    inserts the size-1 scan-step axis at track time.

    Parameters
    ----------
    beam_list : list[ParticleBeam]
        Single (non-vectorized) ``ParticleBeam`` s, each with ``particles`` of
        shape ``(n_particles, 7)`` and a shared ``energy``.

    Returns
    -------
    ParticleBeam
        One vectorized ensemble ``ParticleBeam``.
    """
    all_particles = torch.stack([beam.particles for beam in beam_list], dim=0)
    return ParticleBeam(particles=all_particles, energy=beam_list[0].energy)


# ---------------------------------------------------------------------------
# Statistics helpers (ported from ensemble_utils.py)
# ---------------------------------------------------------------------------
def compute_mean_and_bounds(
    histograms: torch.Tensor,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute elementwise mean and two-sided confidence bounds over an ensemble
    of histograms.

    Parameters
    ----------
    histograms : torch.Tensor
        Ensemble of histograms with shape ``(n_draws, ...)``. Statistics are
        computed across the first dimension.
    uncertainty_type : "percentile" | "std_error"
        ``"percentile"`` uses empirical two-sided percentiles at
        ``(1 - confidence_level) / 2`` and its complement; ``"std_error"`` uses a
        parametric two-sided Student-t interval based on the sample mean and
        standard error.
    confidence_level : float
        Nominal confidence level in ``(0, 1)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(mean_histogram, lower_bound, upper_bound)``, each with shape
        ``histograms.shape[1:]``.
    """
    alpha = 1 - confidence_level
    mean_histogram = torch.mean(histograms, dim=0)

    if uncertainty_type == "percentile":
        lower_bound = torch.quantile(histograms, alpha / 2, dim=0)
        upper_bound = torch.quantile(histograms, 1 - alpha / 2, dim=0)
    elif uncertainty_type == "std_error":
        n = torch.tensor(
            histograms.shape[0], dtype=histograms.dtype, device=histograms.device
        )
        std_error = torch.std(histograms, dim=0, correction=1) / torch.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2, df=float(n) - 1)
        margin = t_crit * std_error

        lower_bound = mean_histogram - margin
        upper_bound = mean_histogram + margin
    else:
        raise ValueError(
            "Invalid uncertainty_type: "
            f"{uncertainty_type}. Must be 'percentile' or 'std_error'."
        )

    return mean_histogram, lower_bound, upper_bound


def vectorized_histogram_1d(
    x: torch.Tensor,
    bins: int = 100,
    bin_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``n`` 1D histograms for an ``(n, m)`` tensor using
    ``torch.bincount`` with per-row offsets.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(n, m)`` or ``(m,)`` (promoted to ``(1, m)``).
    bins : int
        Number of histogram bins.
    bin_range : tuple[float, float] | None
        Optional ``(min, max)`` bin-edge range; inferred if None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(hist, bin_edges)`` where ``hist`` has shape ``(n, bins)`` (or
        ``(bins,)`` for 1D input) and ``bin_edges`` has length ``bins + 1``.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # shape (1, m)
    x = x.contiguous()

    n = x.shape[0]
    device = x.device
    dtype = x.dtype

    if bin_range is None:
        bin_range = (x.min(), x.max())
    bin_edges = torch.linspace(
        bin_range[0], bin_range[1], bins + 1, device=device, dtype=dtype
    )

    idx = torch.bucketize(x, bin_edges) - 1
    idx = idx.clamp(0, bins - 1).long()  # (n, m)

    offset = torch.arange(n, device=device) * bins  # (n,)
    idx_flat = (idx + offset.unsqueeze(1)).flatten()  # (n*m,)

    hist_flat = torch.bincount(idx_flat, minlength=n * bins).to(dtype)
    hist = hist_flat.view(n, bins)  # (n, bins)

    if hist.size(0) == 1:
        hist = hist.squeeze(0)

    return hist, bin_edges


def vectorized_gaussian_filter_1d(
    hist: torch.Tensor, sigma: float, truncate: float = 4.0
) -> torch.Tensor:
    """Apply a 1D Gaussian smoothing filter to one or many histograms (vectorized).

    Parameters
    ----------
    hist : torch.Tensor
        ``(m,)`` for a single histogram or ``(n, m)`` for a batch.
    sigma : float
        Standard deviation of the Gaussian kernel, in bins.
    truncate : float
        Truncate the filter at this many standard deviations.

    Returns
    -------
    torch.Tensor
        Smoothed histogram(s) with the same shape/dtype/device as input.
    """
    if hist.dim() == 1:
        hist = hist.unsqueeze(0)  # (1, m)

    device = hist.device
    dtype = hist.dtype

    radius = int(truncate * sigma + 0.5)
    pos = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (pos / sigma) ** 2)
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, -1)  # (1, 1, k)

    hist = hist.unsqueeze(1)  # (n, 1, m)
    smoothed = F.conv1d(hist, kernel, padding=radius).squeeze(1)  # (n, m)

    if smoothed.size(0) == 1:
        smoothed = smoothed.squeeze(0)

    return smoothed


def compute_statistics_1d(
    x: torch.Tensor,
    bins: int = 100,
    bin_range: tuple[float, float] | None = None,
    smoothing: float = 0.0,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the mean 1D histogram and a two-sided confidence interval for an
    ensemble of samples.

    Parameters
    ----------
    x : torch.Tensor
        ``(n_samples,)`` for a single set or ``(n_draws, n_samples)`` for an
        ensemble (statistics computed across the first dimension).
    bins : int
        Number of histogram bins.
    bin_range : tuple[float, float] | None
        ``(min, max)`` range; inferred from the data if None.
    smoothing : float
        If > 0, sigma (in bins) of a Gaussian applied to each histogram before
        computing mean/bounds.
    uncertainty_type : "percentile" | "std_error"
        Passed to :func:`compute_mean_and_bounds`.
    confidence_level : float
        Nominal confidence level in ``(0, 1)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(bin_centers, mean_hist, lower_bound, upper_bound)``.
    """
    histogram, bin_edges = vectorized_histogram_1d(x, bins=bins, bin_range=bin_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if smoothing:
        histogram = vectorized_gaussian_filter_1d(histogram, sigma=smoothing)

    mean_hist, lower_bound, upper_bound = compute_mean_and_bounds(
        histogram, confidence_level=confidence_level, uncertainty_type=uncertainty_type
    )

    return bin_centers, mean_hist, lower_bound, upper_bound


def vectorized_histogram_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: tuple[int, int] = (100, 100),
    bin_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute batched 2D histograms for coordinate pairs ``(x, y)``.

    Parameters
    ----------
    x : torch.Tensor
        ``(N,)`` for a single set or ``(B, N)`` for a batch.
    y : torch.Tensor
        Same shape as ``x``.
    bins : tuple[int, int]
        ``(bins_x, bins_y)``.
    bin_ranges : tuple[tuple[float, float], tuple[float, float]] | None
        ``((x_min, x_max), (y_min, y_max))``; inferred if None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(hist, x_edges, y_edges)`` with ``hist`` of shape
        ``(B, bins_x, bins_y)``.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.contiguous()
    if y.dim() == 1:
        y = y.unsqueeze(0)
    y = y.contiguous()

    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 1D or 2D tensors")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    bins_x, bins_y = bins

    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    if bin_ranges is None:
        range_x = (float(x.min()), float(x.max()))
        range_y = (float(y.min()), float(y.max()))
    else:
        range_x = bin_ranges[0]
        range_y = bin_ranges[1]

    x_edges = torch.linspace(
        range_x[0], range_x[1], bins_x + 1, device=device, dtype=dtype
    )
    y_edges = torch.linspace(
        range_y[0], range_y[1], bins_y + 1, device=device, dtype=dtype
    )

    ix = (torch.bucketize(x, x_edges) - 1).clamp(0, bins_x - 1).long()  # (B, N)
    iy = (torch.bucketize(y, y_edges) - 1).clamp(0, bins_y - 1).long()  # (B, N)

    idx_flat = ix * bins_y + iy  # (B, N)

    offset = torch.arange(B, device=device, dtype=idx_flat.dtype) * (bins_x * bins_y)
    idx_flat_offset = (idx_flat + offset.unsqueeze(1)).view(-1)

    hist_flat = torch.bincount(idx_flat_offset, minlength=B * bins_x * bins_y).to(dtype)
    hist = hist_flat.view(B, bins_x, bins_y)

    return hist, x_edges, y_edges


def vectorized_gaussian_filter_2d(
    x: torch.Tensor, sigma: float, truncate: float = 4.0
) -> torch.Tensor:
    """Apply a 2D Gaussian filter to a batch of 2D tensors (vectorized).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, H, W)`` (or ``(H, W)``).
    sigma : float
        Standard deviation of the Gaussian kernel in bins.
    truncate : float
        Truncate the filter at this many standard deviations.

    Returns
    -------
    torch.Tensor
        Smoothed tensor with the same shape as the input.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, H, W)

    device, dtype = x.device, x.dtype

    radius = int(truncate * sigma + 0.5)
    kernel_size = 2 * radius + 1
    pos = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (pos / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    # kernel_1d already sums to 1, so its outer product does too -- no need to
    # renormalize the 2D kernel.
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    x = x.unsqueeze(1)  # (B, 1, H, W)
    y = F.conv2d(x, kernel_2d, padding=radius).squeeze(1)  # (B, H, W)
    if y.size(0) == 1:
        y = y.squeeze(0)

    return y


def compute_statistics_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: tuple[int, int] = (100, 100),
    bin_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    smoothing: float = 0.0,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the mean 2D histogram and uncertainty bounds for paired samples.

    Parameters
    ----------
    x : torch.Tensor
        ``(B, N)`` or ``(N,)`` values for the x dimension.
    y : torch.Tensor
        ``(B, N)`` or ``(N,)`` values for the y dimension.
    bins : tuple[int, int]
        ``(nx, ny)``.
    bin_ranges : tuple[tuple[float, float], tuple[float, float]] | None
        ``((x_min, x_max), (y_min, y_max))``; inferred if None.
    smoothing : float
        Sigma of an optional Gaussian applied to the mean histogram; 0.0 disables
        it.
    uncertainty_type : "percentile" | "std_error"
        Passed to :func:`compute_mean_and_bounds`.
    confidence_level : float
        Confidence level in ``(0, 1)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x_centers, y_centers, mean_hist, lower_bound, upper_bound)``.
    """
    hist, x_edges, y_edges = vectorized_histogram_2d(
        x, y, bins=bins, bin_ranges=bin_ranges
    )

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    mean_hist, lower_bound, upper_bound = compute_mean_and_bounds(
        hist, confidence_level=confidence_level, uncertainty_type=uncertainty_type
    )

    if smoothing:
        mean_hist = vectorized_gaussian_filter_2d(mean_hist, sigma=smoothing)

    return x_centers, y_centers, mean_hist, lower_bound, upper_bound
