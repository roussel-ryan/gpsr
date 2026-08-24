"""GPSRLUME: Phase-space reconstruction with GPSR + a frozen LUME-Cheetah
virtual accelerator.

This ``__init__`` curates the package's public API so callers can import the
common entry points directly from ``gpsr.gpsr_lume`` (e.g.
``from gpsr.gpsr_lume import LitGPSRLUME, predict_images``) rather than reaching into
submodules. Names not re-exported here (``gpsr.gpsr_lume._imports``, the low-level
histogram / Gaussian-kernel helpers in :mod:`gpsr.gpsr_lume.ensemble`) are internal
and may move; import them from their submodule if you need them.

The typical flow, module by module:

- :mod:`gpsr.gpsr_lume.data` -- wrap a scan in :class:`GPSRLUMEDataset` /
  :class:`GPSRLUMEDataModule`; persist one as plain tensors with
  :func:`save_dataset` / :func:`load_dataset` (one scan per file) or
  :func:`save_datamodule` / :func:`load_datamodule` (a whole set in one).
- :mod:`gpsr.gpsr_lume.builders` -- assemble a :class:`GPSRLUMEModel` from a JSON-pure
  spec (:func:`model_spec_from_files` -> :func:`build_gpsr_lume_model`).
- :mod:`gpsr.gpsr_lume.training` -- fit / checkpoint via :class:`LitGPSRLUME`.
- :mod:`gpsr.gpsr_lume.predicting` -- track a beam to predicted images
  (:func:`predict_images` and its ensemble / multi-source variants).
- :mod:`gpsr.gpsr_lume.ensemble` -- build an ensemble beam (:func:`list_to_beam`) and
  reduce ensembles to mean + confidence bands.
- :mod:`gpsr.gpsr_lume.plotting` -- screen-image and phase-space (corner) plots.
"""

from gpsr.gpsr_lume.data import (
    GPSRLUMEDataset,
    GPSRLUMEDataModule,
    save_dataset,
    load_dataset,
    save_datamodule,
    load_datamodule,
)
from gpsr.gpsr_lume.model import GPSRLUMEModel
from gpsr.gpsr_lume.training import LitGPSRLUME
from gpsr.gpsr_lume.builders import (
    build_gpsr_lume_model,
    model_spec_from_files,
    serialize_gpsr_lume_model,
)
from gpsr.gpsr_lume.predicting import (
    predict_images,
    predict_ensemble_images,
    predict_multi_source_images,
    predict_multi_source_ensemble_images,
)
from gpsr.gpsr_lume.ensemble import (
    UncertaintyType,
    list_to_beam,
    compute_mean_and_bounds,
    compute_statistics_1d,
    compute_statistics_2d,
)
from gpsr.gpsr_lume.plotting import (
    plot_images,
    plot_ensemble_images,
    plot_multi_source_images,
    plot_multi_source_ensemble_images,
    plot_beam_distribution,
    plot_ensemble_distribution,
)

__all__ = [
    # data
    "GPSRLUMEDataset",
    "GPSRLUMEDataModule",
    "save_dataset",
    "load_dataset",
    "save_datamodule",
    "load_datamodule",
    # model
    "GPSRLUMEModel",
    # training
    "LitGPSRLUME",
    # builders
    "build_gpsr_lume_model",
    "model_spec_from_files",
    "serialize_gpsr_lume_model",
    # predicting
    "predict_images",
    "predict_ensemble_images",
    "predict_multi_source_images",
    "predict_multi_source_ensemble_images",
    # ensemble
    "UncertaintyType",
    "list_to_beam",
    "compute_mean_and_bounds",
    "compute_statistics_1d",
    "compute_statistics_2d",
    # plotting
    "plot_images",
    "plot_ensemble_images",
    "plot_multi_source_images",
    "plot_multi_source_ensemble_images",
    "plot_beam_distribution",
    "plot_ensemble_distribution",
]
