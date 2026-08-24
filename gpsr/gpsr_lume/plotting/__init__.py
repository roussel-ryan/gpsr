"""Plotting for GPSRLUME: screen images and phase-space (corner) distributions.

All plotting lives here (not on the dataset/datamodule) and is driven by plain
dicts / beams. The module is split into two independent halves, re-exported here
so ``from gpsr.gpsr_lume.plotting import plot_images`` (and the package-level
``from gpsr.gpsr_lume import plot_images``) keep working unchanged:

- :mod:`gpsr.gpsr_lume.plotting.images` -- screen-image plots (:func:`plot_images`,
  :func:`plot_ensemble_images`, :func:`plot_multi_source_images`,
  :func:`plot_multi_source_ensemble_images`).
- :mod:`gpsr.gpsr_lume.plotting.distributions` -- phase-space corner plots
  (:func:`plot_beam_distribution`, :func:`plot_ensemble_distribution`, and the
  per-panel :func:`plot_1d_distribution` / :func:`plot_2d_distribution`).

The ensemble statistics both halves rely on live in :mod:`gpsr.gpsr_lume.ensemble`.
"""

from gpsr.gpsr_lume.plotting.images import (
    plot_images,
    plot_ensemble_images,
    plot_multi_source_images,
    plot_multi_source_ensemble_images,
)
from gpsr.gpsr_lume.plotting.distributions import (
    Dimension,
    PRETTY_DIMENSION_LABELS,
    SPATIAL_DIMENSIONS,
    SCALED_DIMENSIONS,
    plot_1d_distribution,
    plot_2d_distribution,
    plot_beam_distribution,
    plot_ensemble_distribution,
)

__all__ = [
    # screen images
    "plot_images",
    "plot_ensemble_images",
    "plot_multi_source_images",
    "plot_multi_source_ensemble_images",
    # phase-space distributions
    "plot_1d_distribution",
    "plot_2d_distribution",
    "plot_beam_distribution",
    "plot_ensemble_distribution",
    # phase-space dimension metadata
    "Dimension",
    "PRETTY_DIMENSION_LABELS",
    "SPATIAL_DIMENSIONS",
    "SCALED_DIMENSIONS",
]
