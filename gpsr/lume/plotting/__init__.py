"""Plotting for GPSRLUME, driven by plain dicts and beams.

Two independent halves, both re-exported here:
``gpsr.lume.plotting.images`` for screen images, and
``gpsr.lume.plotting.distributions`` for phase-space corner plots.
"""

from gpsr.lume.plotting.images import (
    plot_images,
    plot_ensemble_images,
    plot_multi_source_images,
    plot_multi_source_ensemble_images,
)
from gpsr.lume.plotting.distributions import (
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
