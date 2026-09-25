"""GPSRLUME: Phase-space reconstruction with GPSR + a frozen LUME-Cheetah
virtual accelerator.

Names not re-exported here are internal and may move.

The typical flow, module by module:

- ``data`` -- wrap a scan in ``GPSRLUMEDataset`` / ``GPSRLUMEDataModule``, and
  persist it as plain tensors (``save_dataset`` / ``save_datamodule``).
- ``builders`` -- assemble a ``GPSRLUMEModel`` from a JSON-pure spec.
- ``training`` -- fit and checkpoint via ``LitGPSRLUME``.
- ``predicting`` -- track a beam to predicted images.
- ``ensemble`` -- build an ensemble beam and reduce it to mean + bands.
- ``plotting`` -- screen-image and phase-space plots.
"""

from gpsr.lume.data import (
    GPSRLUMEDataset,
    GPSRLUMEDataModule,
    save_dataset,
    load_dataset,
    save_datamodule,
    load_datamodule,
)
from gpsr.lume.model import GPSRLUMEModel
from gpsr.lume.training import LitGPSRLUME
from gpsr.lume.builders import (
    build_gpsr_lume_model,
    model_spec_from_files,
    serialize_gpsr_lume_model,
)
from gpsr.lume.predicting import (
    predict_images,
    predict_ensemble_images,
    predict_multi_source_images,
    predict_multi_source_ensemble_images,
)
from gpsr.lume.ensemble import (
    UncertaintyType,
    list_to_beam,
    compute_mean_and_bounds,
    compute_statistics_1d,
    compute_statistics_2d,
)
from gpsr.lume.plotting import (
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
