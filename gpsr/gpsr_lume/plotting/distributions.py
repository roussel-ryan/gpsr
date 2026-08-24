"""Phase-space (corner) distribution plotting for GPSRLUME.

One of the two halves of :mod:`gpsr.gpsr_lume.plotting` (the other,
:mod:`gpsr.gpsr_lume.plotting.images`, draws screen images). Ported from Cheetah's
``particle_beam.py`` as free functions.

:func:`plot_ensemble_distribution` and :func:`plot_beam_distribution` draw
triangle / corner plots of a Cheetah ``ParticleBeam`` -- 1D histograms on the
diagonal, 2D projections in the lower triangle -- for an ensemble (mean +
confidence bands) or a single beam (plain histograms, single contours),
respectively. The ensemble statistics they rely on live in
:mod:`gpsr.gpsr_lume.ensemble`.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from scipy.ndimage import gaussian_filter

from cheetah.utils import format_axis_with_prefixed_unit

from gpsr.gpsr_lume.ensemble import (
    UncertaintyType,
    compute_statistics_1d,
    compute_statistics_2d,
)

Dimension = Literal["x", "px", "y", "py", "tau", "p"]

PRETTY_DIMENSION_LABELS = {
    "x": r"$x$",
    "px": r"$p_x$",
    "y": r"$y$",
    "py": r"$p_y$",
    "tau": r"$\tau$",
    "p": r"$\delta$",
}
SPATIAL_DIMENSIONS = ("x", "y", "tau")
# Dimensionless momenta: tick display multiplied by 1e3 (0.001 shows as "1").
SCALED_DIMENSIONS = ("px", "py", "p")


# ---------------------------------------------------------------------------
# Phase-space (corner) distribution plots (ported from Cheetah's
# particle_beam.py, as free functions). Ensemble statistics come from
# gpsr.gpsr_lume.ensemble.
# ---------------------------------------------------------------------------
def _scale_axis_ticks_1e3(axis) -> str:
    """Multiply an axis's tick display by 1e3; return the label's inner math text.

    Applies a ``FuncFormatter`` that renders ``value * 1e3`` on both major and
    minor ticks, and returns the axis's existing label with any surrounding ``$``
    stripped so callers can re-wrap it with a unit or a ``10^3`` prefix.
    """
    formatter = mticker.FuncFormatter(lambda value, _pos: f"{value * 1e3:g}")
    axis.set_major_formatter(formatter)
    axis.set_minor_formatter(formatter)
    label = axis.get_label_text().strip()
    return label[1:-1] if label.startswith("$") and label.endswith("$") else label


def _format_spatial_axis_mm(axis) -> None:
    """Force a spatial axis to display in millimetres (m x 10^3)."""
    inner = _scale_axis_ticks_1e3(axis)
    axis.set_label_text(f"${inner}$ (mm)")


def _format_axis_scaled_1e3(axis) -> None:
    """Scale a dimensionless axis's ticks by 1e3 and annotate its label.

    Used for ``px``/``py``, which have no physical unit for
    ``format_axis_with_prefixed_unit`` to prefix; instead we multiply the
    displayed tick values by 1000 and label the axis with the plotted quantity
    ``10^3 <quantity>`` (e.g. ``$10^{3}\\,p_x$``). This avoids a multiplication
    sign against a unit / an isolated power of ten, both of which many journal
    style guides forbid.
    """
    inner = _scale_axis_ticks_1e3(axis)
    axis.set_label_text(r"$10^{3}\," + inner + "$")


def _label_and_format_axis(
    axis, dimension: Dimension, centers, force_spatial_mm: bool = False
) -> None:
    """Set a phase-space axis's label and unit formatting for ``dimension``.

    Sets the pretty label first (``_format_axis_scaled_1e3`` reads it back), then
    applies unit prefixing for spatial dims (metres) or the 1e3 scaling for the
    dimensionless momenta ``px``/``py``. Dimensions in neither group (e.g. ``p``)
    are labelled but left unformatted.
    """
    axis.set_label_text(PRETTY_DIMENSION_LABELS[dimension])
    if dimension in SPATIAL_DIMENSIONS:
        if force_spatial_mm:
            _format_spatial_axis_mm(axis)
        else:
            format_axis_with_prefixed_unit(axis, "m", centers)
    elif dimension in SCALED_DIMENSIONS:
        _format_axis_scaled_1e3(axis)


def _padded_range(values, pad: float = 0.1) -> tuple[float, float]:
    """Return ``(min, max)`` of ``values`` expanded by ``pad`` of their span."""
    low = float(values.min())
    high = float(values.max())
    margin = (high - low) * pad
    return low - margin, high + margin


def _resolve_bin_ranges(
    bin_ranges,
    dimensions: tuple[str, ...],
    full_tensor: np.ndarray,
) -> list[tuple[float, float]]:
    """Normalize a `_corner_plot` ``bin_ranges`` arg to one ``(min, max)`` per dim.

    Accepts ``None`` (infer per-dim), ``"unit_same"`` (one shared range across
    spatial dims and another across unitless dims), a single ``(min, max)``
    (broadcast to all dims), or an already-per-dim list. ``full_tensor`` is the
    stacked per-dim data used to infer ranges. Raises ``ValueError`` if the
    resolved list does not match ``dimensions`` in length or pair shape.
    """
    if bin_ranges is None:
        bin_ranges = [_padded_range(full_tensor[i]) for i in range(len(dimensions))]
    elif isinstance(bin_ranges, str) and bin_ranges == "unit_same":
        spatial_idxs = [
            i for i, dim in enumerate(dimensions) if dim in SPATIAL_DIMENSIONS
        ]
        unitless_idxs = [
            i for i, dim in enumerate(dimensions) if dim in SCALED_DIMENSIONS
        ]
        spatial_bin_range = _padded_range(full_tensor[spatial_idxs])
        unitless_bin_range = _padded_range(full_tensor[unitless_idxs])
        bin_ranges = [
            spatial_bin_range if dim in SPATIAL_DIMENSIONS else unitless_bin_range
            for dim in dimensions
        ]
    elif np.asarray(bin_ranges).shape == (2,):
        bin_ranges = [bin_ranges] * len(dimensions)

    if len(bin_ranges) != len(dimensions) or not all(len(e) == 2 for e in bin_ranges):
        raise ValueError(
            f"bin_ranges must resolve to {len(dimensions)} (min, max) pairs; "
            f"got {bin_ranges!r}."
        )
    return bin_ranges


def plot_1d_distribution(
    beam,
    dimension: Dimension,
    bins: int = 100,
    bin_range: tuple[float, float] | None = None,
    smoothing: float = 0.0,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.9,
    plot_kwargs: dict | None = None,
    fill_between_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
    force_spatial_mm: bool = False,
) -> plt.Axes:
    """Plot the mean 1D histogram of ``dimension`` with an ensemble uncertainty band.

    For a non-vectorized beam (single draw) this reduces to a plain histogram.

    Parameters
    ----------
    beam : ParticleBeam
        A Cheetah ``ParticleBeam``. Its ``dimension`` accessor should return
        ``(n_particles,)`` (single) or ``(n_draws, n_particles)`` (ensemble;
        extra leading dims are flattened).
    dimension : Dimension
        One of ``('x', 'px', 'y', 'py', 'tau', 'p')``.
    bins : int
        Number of histogram bins.
    bin_range : tuple[float, float] | None
        ``(min, max)`` histogram range, or None to infer.
    smoothing : float
        Sigma of a Gaussian kernel applied to the histogram.
    uncertainty_type : "percentile" | "std_error"
        Passed to :func:`gpsr.gpsr_lume.ensemble.compute_statistics_1d`.
    confidence_level : float
        Confidence level for the band.
    plot_kwargs : dict | None
        Extra kwargs forwarded to ``Axes.plot`` for the mean-histogram line.
    fill_between_kwargs : dict | None
        Extra kwargs forwarded to ``Axes.fill_between`` for the ensemble
        confidence band. Merged over the defaults (``color="C1"``,
        ``alpha=0.75``), so user-supplied keys win. Ignored for a single beam
        (no band is drawn).
    ax : plt.Axes | None
        Axes to draw on; a new one is created if None.
    force_spatial_mm : bool
        If True, spatial axes are labelled in mm regardless of the auto-prefix
        that ``format_axis_with_prefixed_unit`` would pick.

    Returns
    -------
    plt.Axes
        The Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    x_array = getattr(beam, dimension)

    if x_array.dim() == 1:
        histogram, edges = np.histogram(
            x_array.cpu().detach().numpy(), bins=bins, range=bin_range
        )
        centers = (edges[:-1] + edges[1:]) / 2
        if smoothing:
            histogram = gaussian_filter(histogram, smoothing)
    else:
        centers, histogram, lower_bound, upper_bound = compute_statistics_1d(
            x=x_array.flatten(start_dim=0, end_dim=-2),
            bins=bins,
            bin_range=bin_range,
            smoothing=smoothing,
            confidence_level=confidence_level,
            uncertainty_type=uncertainty_type,
        )
        centers = centers.cpu().numpy()
        histogram = histogram.cpu().numpy()
        lower_bound = lower_bound.cpu().numpy()
        upper_bound = upper_bound.cpu().numpy()
        fill_defaults = {"color": "C1", "alpha": 0.75}
        ax.fill_between(
            centers,
            lower_bound,
            upper_bound,
            **(fill_defaults | (fill_between_kwargs or {})),
        )

    ax.plot(centers, histogram, **(plot_kwargs or {}))

    _label_and_format_axis(
        ax.xaxis, dimension, centers, force_spatial_mm=force_spatial_mm
    )

    return ax


def plot_2d_distribution(
    beam,
    x_dimension: Dimension,
    y_dimension: Dimension,
    bins: int = 100,
    bin_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    histogram_smoothing: float = 0.0,
    contour_smoothing: float = 1.0,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.9,
    image_cmap: str = "Greys",
    contour_cmap: str = "plasma",
    contour_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    image_alpha: float = 1.0,
    pcolormesh_kwargs: dict | None = None,
    contour_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
    force_spatial_mm: bool = False,
) -> plt.Axes:
    """Plot the mean 2D projection of ``(x_dimension, y_dimension)``.

    A filled ``pcolormesh`` image with overlaid contours (and, for an ensemble,
    additional lower / upper confidence-band contours).

    Parameters
    ----------
    beam : ParticleBeam
        A Cheetah ``ParticleBeam`` (single or ensemble-vectorized).
    x_dimension, y_dimension : Dimension
        Dimensions for the x- and y-axes.
    bins : int
        Number of bins per dimension.
    bin_ranges : tuple[tuple[float, float], tuple[float, float]] | None
        ``((xmin, xmax), (ymin, ymax))`` or None to infer.
    histogram_smoothing : float
        Sigma applied to the mean histogram.
    contour_smoothing : float
        Sigma applied to contour data (after ``histogram_smoothing``).
    uncertainty_type : "percentile" | "std_error"
        Passed to :func:`gpsr.gpsr_lume.ensemble.compute_statistics_2d`.
    confidence_level : float
        Confidence level for the bands.
    image_cmap : str
        Colormap for the pcolormesh fill. Defaults to ``"Greys"``.
    contour_cmap : str
        Colormap for the contour lines. Defaults to ``"plasma"``.
    contour_levels : tuple[float, ...]
        Contour levels as fractions of each histogram's peak. Defaults to
        ``(0.1, 0.5, 0.9)``.
    image_alpha : float
        Opacity of the filled ``pcolormesh``, in ``[0, 1]``. Defaults to 1.0.
    pcolormesh_kwargs : dict | None
        Extra kwargs forwarded to ``Axes.pcolormesh``. Merged over the defaults
        built from ``image_cmap`` / ``image_alpha`` (user keys win).
    contour_kwargs : dict | None
        Extra kwargs forwarded to ``Axes.contour``. Merged over the defaults
        built from ``contour_cmap`` / ``contour_levels`` / the internal
        ``linestyles`` / ``alpha`` (user keys win).
    ax : plt.Axes | None
        Axes to draw on; a new one is created if None.
    force_spatial_mm : bool
        If True, spatial axes are labelled in mm regardless of the auto-prefix.

    Returns
    -------
    plt.Axes
        The Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    x_full = getattr(beam, x_dimension)
    y_full = getattr(beam, y_dimension)

    is_ensemble = x_full.dim() > 1

    if not is_ensemble:
        histogram, x_edges, y_edges = np.histogram2d(
            x_full.cpu().detach().numpy(),
            y_full.cpu().detach().numpy(),
            bins=bins,
            range=bin_ranges,
        )
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        smoothed_histogram = gaussian_filter(histogram, histogram_smoothing)
    else:
        x_array = x_full.flatten(start_dim=0, end_dim=-2)
        y_array = y_full.flatten(start_dim=0, end_dim=-2)
        x_centers, y_centers, smoothed_histogram, lower_bound, upper_bound = (
            compute_statistics_2d(
                x=x_array,
                y=y_array,
                bins=(bins, bins),
                bin_ranges=bin_ranges,
                smoothing=histogram_smoothing,
                uncertainty_type=uncertainty_type,
                confidence_level=confidence_level,
            )
        )
        x_centers = x_centers.cpu().numpy()
        y_centers = y_centers.cpu().numpy()
        smoothed_histogram = smoothed_histogram.cpu().numpy()
        lower_bound = lower_bound.cpu().numpy()
        upper_bound = upper_bound.cpu().numpy()

        lower_bound = gaussian_filter(lower_bound, contour_smoothing)
        upper_bound = gaussian_filter(upper_bound, contour_smoothing)

    contour_histogram = gaussian_filter(smoothed_histogram, contour_smoothing)

    pcolormesh_defaults = {"cmap": image_cmap, "alpha": image_alpha}
    ax.pcolormesh(
        x_centers,
        y_centers,
        smoothed_histogram.T,
        **(pcolormesh_defaults | (pcolormesh_kwargs or {})),
    )
    contour_defaults = {
        "levels": list(contour_levels),
        "cmap": contour_cmap,
        "linestyles": "dashed",
        "alpha": 0.75,
    }

    def draw_contour(data):
        peak = data.max()
        normalized = data.T / peak if peak > 0 else data.T
        ax.contour(
            x_centers,
            y_centers,
            normalized,
            **(contour_defaults | (contour_kwargs or {})),
        )

    draw_contour(contour_histogram)
    if is_ensemble:
        draw_contour(lower_bound)
        draw_contour(upper_bound)

    _label_and_format_axis(
        ax.xaxis, x_dimension, x_centers, force_spatial_mm=force_spatial_mm
    )
    _label_and_format_axis(
        ax.yaxis, y_dimension, y_centers, force_spatial_mm=force_spatial_mm
    )

    return ax


def _corner_plot(
    beam,
    dimensions: tuple[str, ...],
    bins: int,
    bin_ranges: Literal["unit_same"]
    | tuple[float, float]
    | list[tuple[float, float]]
    | None,
    image_cmap: str,
    contour_cmap: str,
    plot_1d_kwargs: dict | None,
    plot_2d_kwargs: dict | None,
    axs: np.ndarray | None,
) -> tuple[plt.Figure, np.ndarray]:
    """Shared triangle / corner-plot scaffolding for single-beam and ensemble beams.

    Lays out the ``(N, N)`` grid -- 1D histograms on the diagonal, 2D histograms
    in the lower triangle, upper triangle hidden -- and delegates each panel to
    :func:`plot_1d_distribution` / :func:`plot_2d_distribution`. Those functions
    auto-detect whether the beam is ensemble-vectorized (draws indexed by a leading
    dim -> mean histogram plus confidence band / bounds contours) or a single beam
    (``particles`` of shape ``(n_particles, 7)`` -> a plain histogram with single
    contours and no bounds). Ensemble-only options (``uncertainty_type``,
    ``confidence_level``) are threaded in via ``plot_1d_kwargs`` / ``plot_2d_kwargs`` by
    the caller, so this helper stays agnostic to them.
    """
    if axs is None:
        fig, axs = plt.subplots(
            len(dimensions),
            len(dimensions),
            figsize=(2 * len(dimensions), 2 * len(dimensions)),
        )
    else:
        fig = axs[0, 0].figure
        assert axs.shape == (len(dimensions), len(dimensions)), (
            "If `axs` is provided, it must have the shape "
            f"`({len(dimensions)}, {len(dimensions)})`."
        )

    # Determine bin ranges for all plots in the grid at once.
    full_tensor = (
        torch.stack([getattr(beam, dim) for dim in dimensions], dim=0)
        .cpu()
        .detach()
        .numpy()
    )
    bin_ranges = _resolve_bin_ranges(bin_ranges, dimensions, full_tensor)

    # Diagonal: 1D histograms.
    diagonal_axs = [axs[i, i] for i, _ in enumerate(dimensions)]
    for dimension, bin_range, ax in zip(dimensions, bin_ranges, diagonal_axs):
        plot_1d_distribution(
            beam,
            dimension=dimension,
            bins=bins,
            bin_range=bin_range,
            ax=ax,
            **(plot_1d_kwargs or {}),
        )

    # Off-diagonal: 2D histograms in the lower triangle, upper triangle hidden.
    for i, j in itertools.combinations(range(len(dimensions)), 2):
        plot_2d_distribution(
            beam,
            x_dimension=dimensions[i],
            y_dimension=dimensions[j],
            bins=bins,
            bin_ranges=(bin_ranges[i], bin_ranges[j]),
            ax=axs[j, i],
            **(
                {"image_cmap": image_cmap, "contour_cmap": contour_cmap}
                | (plot_2d_kwargs or {})
            ),
        )
        axs[i, j].set_visible(False)

    # Clean up shared axes / labels.
    for ax_column in axs.T:
        for ax in ax_column[0:-1]:
            ax.sharex(ax_column[0])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_xlabel(None)
    for i, ax_row in enumerate(axs):
        for ax in ax_row[1:i]:
            ax.sharey(ax_row[0])
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_ylabel(None)
    for i, _ in enumerate(dimensions):
        axs[i, i].sharey(axs[0, 0])
        axs[i, i].set_yticks([])
        axs[i, i].set_ylabel(None)

    return fig, axs


def plot_ensemble_distribution(
    beam,
    dimensions: tuple[str, ...] = ("x", "px", "y", "py", "tau", "p"),
    bins: int = 100,
    bin_ranges: Literal["unit_same"]
    | tuple[float, float]
    | list[tuple[float, float]]
    | None = None,
    uncertainty_type: UncertaintyType = "percentile",
    confidence_level: float = 0.9,
    image_cmap: str = "Greys",
    contour_cmap: str = "plasma",
    plot_1d_kwargs: dict | None = None,
    plot_2d_kwargs: dict | None = None,
    axs: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Corner / triangle plot of 1D and 2D projections for a beam ensemble.

    The diagonal shows mean 1D histograms with confidence bands; the lower
    triangle shows mean 2D histograms with lower / mean / upper contours. When the
    beam is vectorized (multiple draws), uncertainty is estimated across the
    leading (draw) dimension -- e.g. across the 16 beams of a ``(16, 100000, 7)``
    ensemble. For a single beam (``particles`` of shape ``(n_particles, 7)``), use
    :func:`plot_beam_distribution`, which draws plain histograms with single
    contours and no bands.

    Parameters
    ----------
    beam : ParticleBeam
        A Cheetah ``ParticleBeam``, ideally ensemble-vectorized.
    dimensions : tuple[str, ...]
        Dimension names to include (subset of
        ``('x', 'px', 'y', 'py', 'tau', 'p')``).
    bins : int
        Number of bins for both 1D and 2D histograms.
    bin_ranges : "unit_same" | tuple[float, float] | list[tuple[float, float]] | None
        Bin-range spec. ``None`` infers per-dimension; ``"unit_same"`` shares a
        range across spatial dims and another across unitless dims; a single
        ``(min, max)`` applies to all; a list gives one pair per dimension.
    uncertainty_type : "percentile" | "std_error"
        Passed to :func:`gpsr.gpsr_lume.ensemble.compute_statistics_1d` /
        :func:`gpsr.gpsr_lume.ensemble.compute_statistics_2d`.
    confidence_level : float
        Confidence level for the bands.
    image_cmap : str
        Colormap for the pcolormesh fill on the off-diagonal panels. Defaults to
        ``"Greys"``. Overridden by ``plot_2d_kwargs["image_cmap"]`` if given.
    contour_cmap : str
        Colormap for the contour lines on the off-diagonal panels. Defaults to
        ``"plasma"``. Overridden by ``plot_2d_kwargs["contour_cmap"]`` if given.
    plot_1d_kwargs : dict | None
        Extra kwargs forwarded to :func:`plot_1d_distribution`.
    plot_2d_kwargs : dict | None
        Extra kwargs forwarded to :func:`plot_2d_distribution`.
    axs : np.ndarray | None
        Optional pre-made ``(N, N)`` Axes array.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        ``(fig, axs)``.
    """
    # `force_spatial_mm` isn't uncertainty-related, but it's threaded through the
    # same kwargs channel to both per-panel plotters.
    shared_kwargs = {
        "uncertainty_type": uncertainty_type,
        "confidence_level": confidence_level,
        "force_spatial_mm": True,
    }
    return _corner_plot(
        beam,
        dimensions=dimensions,
        bins=bins,
        bin_ranges=bin_ranges,
        image_cmap=image_cmap,
        contour_cmap=contour_cmap,
        plot_1d_kwargs=shared_kwargs | (plot_1d_kwargs or {}),
        plot_2d_kwargs=shared_kwargs | (plot_2d_kwargs or {}),
        axs=axs,
    )


def plot_beam_distribution(
    beam,
    dimensions: tuple[str, ...] = ("x", "px", "y", "py", "tau", "p"),
    bins: int = 100,
    bin_ranges: Literal["unit_same"]
    | tuple[float, float]
    | list[tuple[float, float]]
    | None = None,
    image_cmap: str = "Greys",
    contour_cmap: str = "plasma",
    plot_1d_kwargs: dict | None = None,
    plot_2d_kwargs: dict | None = None,
    axs: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Corner / triangle plot of 1D and 2D projections for a single beam.

    The single-beam analogue of :func:`plot_ensemble_distribution`: same layout
    (1D histograms on the diagonal, 2D histograms in the lower triangle), but for a
    beam with ``particles`` of shape ``(n_particles, 7)``. Because there is only one
    draw, there is no uncertainty to estimate -- the diagonal shows plain 1D
    histograms (no confidence band) and each off-diagonal panel shows a single set
    of contours (one line per level) rather than the lower / mean / upper triplet.

    This just calls the shared corner-plot scaffolding without any uncertainty
    options; :func:`plot_1d_distribution` / :func:`plot_2d_distribution` already
    fall back to single-beam rendering when the per-dimension accessor returns a 1D
    ``(n_particles,)`` tensor.

    Parameters
    ----------
    beam : ParticleBeam
        A Cheetah ``ParticleBeam`` with ``particles`` of shape
        ``(n_particles, 7)`` (a single, non-vectorized beam).
    dimensions : tuple[str, ...]
        Dimension names to include (subset of
        ``('x', 'px', 'y', 'py', 'tau', 'p')``).
    bins : int
        Number of bins for both 1D and 2D histograms.
    bin_ranges : "unit_same" | tuple[float, float] | list[tuple[float, float]] | None
        Bin-range spec (see :func:`plot_ensemble_distribution`).
    image_cmap : str
        Colormap for the pcolormesh fill on the off-diagonal panels. Defaults to
        ``"Greys"``. Overridden by ``plot_2d_kwargs["image_cmap"]`` if given.
    contour_cmap : str
        Colormap for the contour lines on the off-diagonal panels. Defaults to
        ``"plasma"``. Overridden by ``plot_2d_kwargs["contour_cmap"]`` if given.
    plot_1d_kwargs : dict | None
        Extra kwargs forwarded to :func:`plot_1d_distribution`.
    plot_2d_kwargs : dict | None
        Extra kwargs forwarded to :func:`plot_2d_distribution`.
    axs : np.ndarray | None
        Optional pre-made ``(N, N)`` Axes array.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        ``(fig, axs)``.
    """
    return _corner_plot(
        beam,
        dimensions=dimensions,
        bins=bins,
        bin_ranges=bin_ranges,
        image_cmap=image_cmap,
        contour_cmap=contour_cmap,
        plot_1d_kwargs={"force_spatial_mm": True} | (plot_1d_kwargs or {}),
        plot_2d_kwargs={"force_spatial_mm": True} | (plot_2d_kwargs or {}),
        axs=axs,
    )
