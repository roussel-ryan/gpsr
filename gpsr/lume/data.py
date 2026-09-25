import pickle

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tensordict import TensorDict
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader

DATASET_FORMAT = "gpsr_lume_dataset"
DATASET_VERSION = 1
DATAMODULE_FORMAT = "gpsr_lume_datamodule"
DATAMODULE_VERSION = 1


class GPSRLUMEDataset(torch.utils.data.Dataset):
    """A Dataset for Generative Phase Space Reconstruction (GPSR).

    Stores beamline scan data as a batched TensorDict with two sub-dicts:
    - "beamline_settings": parameters varied during the scan (e.g. quad strengths),
      one value per sample.
    - "observations": measured data at each scan step (e.g. screen images),
      one tensor per sample.

    Non-batched metadata is stored as plain attributes:
    - observations_metadata: per-observation-key dict with type and resolution info.
    - beamline_constants: parameters held fixed during the scan.

    Parameters
    ----------
    beamline_settings : dict[str, Tensor]
        Scan parameters keyed by PV name. Each tensor has leading dim = n_samples.
    observations : dict[str, Tensor]
        Measured observations keyed by PV name. Each tensor has leading dim = n_samples.
    observations_metadata : dict[str, dict]
        Metadata for each observation key (must have the same keys as observations).
        Typical entries include "type" (e.g. "screen") and "pixel_size".
    n_samples : int | None
        Number of scan steps (samples) in the dataset. If None (default), it is
        inferred from the leading dimension of the beamline_settings tensors.
    beamline_constants : dict[str, Tensor] | None
        Fixed beamline parameters keyed by PV name. Keys must not overlap with
        beamline_settings.

    Examples
    --------
    >>> dataset = GPSRLUMEDataset(
    ...     beamline_settings={"QUAD:IN10:525:BCTRL": quad_tensor},
    ...     observations={"PROF:IN10:571:Image:ArrayData": image_tensor},
    ...     observations_metadata={
    ...         "PROF:IN10:571:Image:ArrayData": {
    ...             "type": "screen", "shape": (231, 212), "pixel_size": res
    ...         }
    ...     },
    ...     beamline_constants={"QUAD:IN10:511:BCTRL": torch.tensor(5.064)},
    ... )
    >>> batch = dataset[0:3]  # returns a TensorDict slice
    """

    def __init__(
        self,
        beamline_settings: dict[str, Tensor],
        observations: dict[str, Tensor],
        observations_metadata: dict[str, dict],
        n_samples: int | None = None,
        beamline_constants: dict[str, Tensor] | None = None,
    ):
        if not isinstance(observations_metadata, dict):
            raise TypeError(
                f"observations_metadata must be a dict, got {type(observations_metadata)}"
            )

        if beamline_constants is not None and not isinstance(beamline_constants, dict):
            raise TypeError(
                f"beamline_constants must be a dict, got {type(beamline_constants)}"
            )

        if beamline_constants is None:
            beamline_constants = {}

        if n_samples is None:
            if not beamline_settings:
                raise ValueError(
                    "Cannot infer n_samples from empty beamline_settings; pass n_samples explicitly."
                )
            n_samples = next(iter(beamline_settings.values())).shape[0]

        obs_keys = set(observations.keys())
        meta_keys = set(observations_metadata.keys())
        if obs_keys != meta_keys:
            raise ValueError(
                f"observations and observations_metadata must have the same keys. "
                f"observations keys: {obs_keys}, observations_metadata keys: {meta_keys}"
            )

        all_param_keys = list(beamline_settings.keys()) + list(
            beamline_constants.keys()
        )
        if len(all_param_keys) != len(set(all_param_keys)):
            duplicates = {k for k in all_param_keys if all_param_keys.count(k) > 1}
            raise ValueError(
                f"Parameter keys must be unique across beamline_settings and beamline_constants. "
                f"Duplicates: {duplicates}"
            )

        for key, tensor in observations.items():
            meta = observations_metadata[key]
            if meta.get("type") == "screen":
                if "shape" not in meta:
                    raise ValueError(
                        f"Screen observation '{key}' is missing 'shape' in its metadata."
                    )
                expected = tuple(meta["shape"])
                actual = tuple(tensor.shape[-2:])
                if actual != expected:
                    raise ValueError(
                        f"Screen observation '{key}' has spatial dims {actual} "
                        f"but metadata 'shape' is {expected}."
                    )

        self.observations_metadata = observations_metadata
        self.beamline_constants = beamline_constants
        self.data = TensorDict(
            {
                "beamline_settings": TensorDict(
                    beamline_settings, batch_size=n_samples
                ),
                "observations": TensorDict(observations, batch_size=n_samples),
            },
            batch_size=n_samples,
        )

    def __len__(self):
        return self.data.batch_size[0]

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        repr_str = f"GPSRLUMEDataset(n_samples={len(self)})\n"
        repr_str += f"  Data: {self.data}\n"
        repr_str += f"  Observations Metadata: {self.observations_metadata}"
        if self.beamline_constants:
            repr_str += f"\n  Beamline Constants: {self.beamline_constants}"
        return repr_str

    def to_dict(self) -> dict[str, TensorDict | dict]:
        """Return this dataset's contents as a flat dict.

        The dict has three keys -- ``beamline_settings``, ``images`` and
        ``observations_metadata``. Its keys match the first three (keyword)
        arguments of ``plot_images`` (the dataset's
        stored ``observations`` land under the plotter's role-neutral
        ``images`` slot), so it can be splatted straight in:

            from gpsr.lume.plotting import plot_images
            plot_images(**dataset.to_dict())
        """
        return {
            "beamline_settings": self.data["beamline_settings"],
            "images": self.data["observations"],
            "observations_metadata": self.observations_metadata,
        }

    def save(self, path) -> None:
        """Write this dataset to ``path``. See ``save_dataset``."""
        save_dataset(self, path)

    @classmethod
    def load(cls, path) -> "GPSRLUMEDataset":
        """Read a dataset from ``path``. See ``load_dataset``."""
        return load_dataset(path)


class GPSRLUMEDataModule(L.LightningDataModule):
    """A LightningDataModule wrapping one set of GPSRLUME datasets.

    A datamodule carries a *single* set of per-source datasets and exposes them
    two ways: shuffled (``train_dataloader``, for ``Trainer.fit``) and unshuffled
    (``test_dataloader``, for ``Trainer.test``). "Train vs test" is a role assigned
    by which dataloader the Trainer calls, not two separate dataset pools -- there
    is no validation loop, and a held-out set is simply another datamodule.

    One DataLoader is returned per source, so ``training_step`` receives

    ``batch[source_name] -> subbatch``

    where each ``subbatch`` is a stacked ``TensorDict`` with keys
    ``"beamline_settings"`` and ``"observations"``.

    Parameters
    ----------
    datasets : GPSRLUMEDataset | dict[str, GPSRLUMEDataset]
        A single dataset or a mapping of source name to dataset. A single dataset
        is promoted to ``{"default": dataset}``.
    batch_size : int | None, default=None
        Batch size shared by all source dataloaders. If ``None``, each source is
        loaded as one full-dataset batch.
    num_workers : int, default=0
        Number of worker processes used by each DataLoader.
    pin_memory : bool, default=False
        Whether DataLoaders should pin memory.

    Attributes
    ----------
    datasets : dict[str, GPSRLUMEDataset]
        Standardized source-to-dataset mapping.
    source_info : dict[str, dict]
        Per-source metadata registry containing:
        - ``observations_metadata``
        - ``beamline_constants``

        This is the module<->datamodule contract: ``LitGPSRLUME`` reads
        ``source_info`` from this datamodule in its ``setup`` hook (rather than
        through its constructor), so any datamodule paired with ``LitGPSRLUME``
        for ``fit`` *or* ``test`` must expose this attribute.

        It looks like a duplicate of what the datasets already carry, and it is --
        but by reference, not by copy, and ``save_datamodule`` writes nothing
        extra for it. It exists because neither entry can reach the model *through
        the batch*: both are per-*source* rather than per-sample, and
        ``observations_metadata`` holds plain Python objects, so
        ``GPSRLUMEDataset.__getitem__`` cannot index them and
        ``_collate_fn`` cannot stack them -- yet ``_shared_step`` needs the
        metadata to configure the screens it predicts with. Hence a side channel,
        read once in ``setup``. Exposing it as its own attribute (rather than
        letting the model reach into ``datasets``) is also what keeps
        ``LitGPSRLUME`` from naming ``GPSRLUMEDataset`` at all, so any
        datamodule that offers this mapping can stand in.

        Derived on access from ``datasets``, so replacing or adding a source
        after construction is reflected rather than silently stale.
    """

    def __init__(
        self,
        datasets: GPSRLUMEDataset | dict[str, GPSRLUMEDataset],
        batch_size: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.datasets = self._standardize_datasets(datasets)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @property
    def source_info(self) -> dict[str, dict]:
        """Per-source metadata registry, looked up by ``LitGPSRLUME`` in ``setup``.

        Both entries are held by reference to each dataset's own attributes, and
        neither is per-sample, so neither can ride in the batch.

        Derived on access rather than snapshotted in ``__init__`` so it cannot fall
        out of step with ``datasets``: a source assigned after construction would
        otherwise be missing here, and ``_shared_step`` would ``KeyError`` on the
        source its own dataloader had just yielded.
        """
        return {
            source_name: {
                "observations_metadata": dataset.observations_metadata,
                "beamline_constants": dataset.beamline_constants,
            }
            for source_name, dataset in self.datasets.items()
        }

    @staticmethod
    def _standardize_datasets(
        datasets: GPSRLUMEDataset | dict[str, GPSRLUMEDataset],
    ) -> dict[str, GPSRLUMEDataset]:
        """Return datasets as a source-name to dataset dictionary."""
        if isinstance(datasets, GPSRLUMEDataset):
            return {"default": datasets}
        return dict(datasets)

    @staticmethod
    def _collate_fn(samples):
        """Collate function that stacks TensorDict samples into one sub-batch."""
        return torch.stack(samples)

    def _resolve_batch_size(self, dataset: GPSRLUMEDataset) -> int:
        """Resolve the effective batch size for a dataset.

        ``None`` means "one full-dataset batch", resolved to ``len(dataset)``.
        """
        if self.batch_size is None:
            if len(dataset) == 0:
                raise ValueError(
                    "Cannot use full-dataset batching on an empty dataset."
                )
            return len(dataset)

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive int or None.")
        return self.batch_size

    def _make_combined_loader(
        self, datasets: dict[str, GPSRLUMEDataset], shuffle: bool
    ) -> CombinedLoader:
        """Build one DataLoader per source, wrapped in a ``CombinedLoader``.

        The ``"max_size_cycle"`` mode makes every batch a ``{source_name:
        subbatch}`` dict, which is the structure ``_shared_step`` expects. Both
        ``train_dataloader`` and ``test_dataloader`` go through here over the same
        ``self.datasets`` so they return the same type and yield the same batch
        shape -- the only difference is shuffling. (During ``fit`` Lightning would
        wrap a bare dict of loaders this way automatically; doing it explicitly
        keeps train and eval symmetric and makes the ``test`` path -- whose
        default mode is ``"sequential"`` -- behave identically.)
        """
        loaders = {
            source_name: DataLoader(
                dataset,
                batch_size=self._resolve_batch_size(dataset),
                collate_fn=self._collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=shuffle,
            )
            for source_name, dataset in datasets.items()
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def train_dataloader(self):
        """Return the per-source DataLoaders shuffled (for ``Trainer.fit``)."""
        return self._make_combined_loader(self.datasets, shuffle=True)

    def test_dataloader(self):
        """Return the per-source DataLoaders unshuffled (for ``Trainer.test`` / eval)."""
        return self._make_combined_loader(self.datasets, shuffle=False)

    def to_sources_spec(self) -> dict[str, dict]:
        """Return this module's per-source data as a ``sources`` spec.

        Projects each source into the observation-free mapping the multi-source
        predictors consume. The measured ``observations`` are dropped: prediction
        never needs them, and they are the heavy part of a dataset.

            preds = predict_multi_source_ensemble_images(model, dm.to_sources_spec(), beam)

        Returns
        -------
        dict[str, dict]
            ``{source_name: {"beamline_settings", "observations_metadata",
            "beamline_constants"}}``.
        """
        return {
            source_name: {
                "beamline_settings": dataset.data["beamline_settings"],
                "observations_metadata": dataset.observations_metadata,
                "beamline_constants": dataset.beamline_constants,
            }
            for source_name, dataset in self.datasets.items()
        }

    def to_observations(self) -> dict[str, TensorDict]:
        """Return this module's measured observations, keyed by source name.

        The images ``to_sources_spec`` drops, for use as the measured overlay:

            plot_multi_source_ensemble_images(
                preds, dm.to_sources_spec(), overlay_images=dm.to_observations()
            )

        Returns
        -------
        dict[str, TensorDict]
            ``{source_name: observations}``; each ``observations`` is keyed by
            observation PV with a leading n_samples dim.
        """
        return {
            source_name: dataset.data["observations"]
            for source_name, dataset in self.datasets.items()
        }

    def save(self, path) -> None:
        """Write this module's data to ``path``. See ``save_datamodule``."""
        save_datamodule(self, path)

    @classmethod
    def load(cls, path, **datamodule_kwargs) -> "GPSRLUMEDataModule":
        """Read a datamodule from ``path``. See ``load_datamodule``."""
        return load_datamodule(path, **datamodule_kwargs)

    def __repr__(self):
        """Return a string representation of the data module."""

        def _summarize(datasets):
            if not datasets:
                return "    None"
            lines = []
            for name, ds in datasets.items():
                ds_repr = repr(ds).replace("\n", "\n      ")
                lines.append(f"    [{name}] {ds_repr}")
            return "\n".join(lines)

        batch_size_repr = "full" if self.batch_size is None else self.batch_size
        repr_str = "GPSRLUMEDataModule(\n"
        repr_str += f"  batch_size={batch_size_repr}, num_workers={self.num_workers}, pin_memory={self.pin_memory}\n"
        repr_str += f"  sources: {list(self.source_info.keys())}\n"
        repr_str += f"  datasets:\n{_summarize(self.datasets)}\n"
        repr_str += ")"
        return repr_str


def _dataset_to_dict(dataset: GPSRLUMEDataset) -> dict:
    """Return ``dataset``'s contents as ``GPSRLUMEDataset`` constructor kwargs.

    Every key is a constructor parameter, so
    ``GPSRLUMEDataset(**_dataset_to_dict(ds))`` reconstructs it. Unlike
    ``GPSRLUMEDataset.to_dict``, which is shaped for the plotters and cannot
    round-trip.

    ``TensorDict.to_dict`` unwraps the stored batched TensorDicts into plain dicts
    of tensors, which is what makes the result loadable under
    ``weights_only=True``; a pickled ``TensorDict`` is not.
    """
    data = dataset.data
    return {
        "beamline_settings": data["beamline_settings"].to_dict(),
        "observations": data["observations"].to_dict(),
        "observations_metadata": dataset.observations_metadata,
        "beamline_constants": dataset.beamline_constants,
        # Stored rather than re-inferred: a source whose settings are all held
        # constant has empty `beamline_settings`, and the constructor cannot infer
        # a sample count from that.
        "n_samples": len(dataset),
    }


# The formats this module writes, mapped to the function that reads each. Used to
# redirect a file handed to the wrong loader -- the two are easy to confuse, since
# both are ``.pt`` files holding scan data.
_FORMAT_READERS = {
    DATASET_FORMAT: "load_dataset",
    DATAMODULE_FORMAT: "load_datamodule",
}


def _load_envelope(path, expected_format: str, current_version: int, what: str) -> dict:
    """Read and validate one of this module's format/version envelopes.

    Shared by ``load_dataset`` and ``load_datamodule``; returns the raw
    dict, leaving the caller to interpret its payload.

    Raises
    ------
    ValueError
        If ``path`` does not hold an envelope of ``expected_format``, or holds a
        newer layout version than this ``gpsr.lume`` understands.
    """
    try:
        raw = torch.load(path, weights_only=True)
    except pickle.UnpicklingError as err:
        # Everything this module writes is a tensor, string or number, so a
        # weights-only read of a genuine file always succeeds. Failing here means the
        # file holds pickled objects -- most often a `torch.save` of a beam, a
        # datamodule, or a checkpoint. Say so, rather than leaving the caller with
        # torch's advice to set `weights_only=False`, which would not help.
        raise ValueError(
            f"'{path}' is not a gpsr.lume {what} file: it contains pickled "
            f"objects, while gpsr.lume writes only tensors, strings and numbers. "
            f"It is most likely a torch.save of some other object."
        ) from err

    found_format = raw.get("format") if isinstance(raw, dict) else None
    if found_format != expected_format:
        if found_format in _FORMAT_READERS:
            # The other of this module's two formats. Name its loader instead of
            # just refusing: one scan per file and all sources in one file are both
            # normal ways to store this data.
            raise ValueError(
                f"'{path}' holds a '{found_format}' envelope, not "
                f"'{expected_format}'. Read it with "
                f"gpsr.lume.data.{_FORMAT_READERS[found_format]} instead."
            )
        found = (
            f"a dict with keys {sorted(raw)[:6]}"
            if isinstance(raw, dict)
            else f"a {type(raw).__name__}"
        )
        raise ValueError(
            f"'{path}' is not a gpsr.lume {what} file (expected a "
            f"'{expected_format}' envelope, found {found}). Files written by "
            f"gpsr.lume carry one; a bare scan dict or a torch.save of some other "
            f"object does not."
        )

    version = raw.get("version")
    if not isinstance(version, int) or version > current_version:
        raise ValueError(
            f"'{path}' has {what} layout version {version!r}, but this "
            f"gpsr.lume understands up to {current_version}. It was written by "
            f"a newer version; upgrade gpsr to read it."
        )
    return raw


def save_dataset(dataset: GPSRLUMEDataset, path) -> None:
    """Save a single ``GPSRLUMEDataset`` -- one scan -- to ``path``.

    Writes the *data* under a format/version envelope rather than a pickle of the
    object, so it loads with ``weights_only=True``. One file per scan is how scans
    are usually collected, and a set of them becomes a datamodule at load time:

        datasets = {p.stem: GPSRLUMEDataset.load(p) for p in sorted(dir.glob("*.pt"))}
        datamodule = GPSRLUMEDataModule(datasets)

    The source name is not stored -- a lone scan has none, and the file name is the
    natural carrier. Use ``save_datamodule`` to make the naming part of the payload.

    Parameters
    ----------
    dataset : GPSRLUMEDataset
        The dataset whose contents are written.
    path : str | os.PathLike
        Destination file. Its parent directory must already exist.

    See Also
    --------
    load_dataset : The inverse.
    save_datamodule : The multi-source sibling; one file for a whole set of scans.
    """
    torch.save(
        {
            "format": DATASET_FORMAT,
            "version": DATASET_VERSION,
            "dataset": _dataset_to_dict(dataset),
        },
        path,
    )


def load_dataset(path) -> GPSRLUMEDataset:
    """Load a ``GPSRLUMEDataset`` written by ``save_dataset``.

    The file's ``dataset`` payload is splatted into ``GPSRLUMEDataset``, so the
    dataset's own validation (observation and metadata keys must agree, screen
    ``shape`` must match the image dims) runs on load -- a malformed file is
    rejected here rather than at the first training step.

    Parameters
    ----------
    path : str | os.PathLike
        A file written by ``save_dataset``.

    Returns
    -------
    GPSRLUMEDataset

    Raises
    ------
    ValueError
        If ``path`` does not hold a dataset file, or holds a newer layout version
        than this ``gpsr.lume`` understands. A whole-datamodule file is reported as
        such, pointing at ``load_datamodule``.
    """
    raw = _load_envelope(path, DATASET_FORMAT, DATASET_VERSION, "dataset")
    return GPSRLUMEDataset(**raw["dataset"])


def save_datamodule(datamodule: GPSRLUMEDataModule, path) -> None:
    """Save a ``GPSRLUMEDataModule``'s data to ``path``.

    Writes the *data*, not the object: a dict of plain tensors, strings and numbers
    under a format/version envelope. A pickle of the datamodule would embed the
    class's module path, so the file would rot on any rename and would force
    ``weights_only=False`` on whoever read it.

    The loader knobs (``batch_size``, ``num_workers``, ``pin_memory``) are not
    saved, since they describe how to iterate the data rather than the data itself.
    Pass them to ``load_datamodule``; a round trip resets them to their defaults.

    Parameters
    ----------
    datamodule : GPSRLUMEDataModule
        The module whose datasets are written.
    path : str | os.PathLike
        Destination file. Its parent directory must already exist.

    See Also
    --------
    load_datamodule : The inverse.
    save_dataset : The one-scan sibling; one file per source instead of one for the
        set, which is how scans are usually collected.
    """
    torch.save(
        {
            "format": DATAMODULE_FORMAT,
            "version": DATAMODULE_VERSION,
            "sources": {
                source_name: _dataset_to_dict(dataset)
                for source_name, dataset in datamodule.datasets.items()
            },
        },
        path,
    )


def load_datamodule(path, **datamodule_kwargs) -> GPSRLUMEDataModule:
    """Load a ``GPSRLUMEDataModule`` written by ``save_datamodule``.

    Each entry of the file's ``sources`` mapping is splatted into
    ``GPSRLUMEDataset``, so the dataset's own validation (observation and
    metadata keys must agree, screen ``shape`` must match the image dims) runs on
    load -- a malformed file is rejected here rather than at the first training
    step.

    Parameters
    ----------
    path : str | os.PathLike
        A file written by ``save_datamodule``.
    **datamodule_kwargs
        Passed to ``GPSRLUMEDataModule`` (``batch_size``, ``num_workers``,
        ``pin_memory``), which the file does not carry.

    Returns
    -------
    GPSRLUMEDataModule

    Raises
    ------
    ValueError
        If ``path`` does not hold a datamodule file, or holds a newer layout
        version than this ``gpsr.lume`` understands. A single-scan file is reported
        as such, pointing at ``load_dataset``.
    """
    raw = _load_envelope(path, DATAMODULE_FORMAT, DATAMODULE_VERSION, "datamodule")
    datasets = {
        source_name: GPSRLUMEDataset(**fields)
        for source_name, fields in raw["sources"].items()
    }
    return GPSRLUMEDataModule(datasets, **datamodule_kwargs)
