import warnings
from typing import Callable

import lightning as L
import torch

from gpsr.losses import normalize_images

from gpsr.lume._imports import import_from_path, to_import_path
from gpsr.lume.builders import (
    build_gpsr_lume_model,
    serialize_gpsr_lume_model,
)
from gpsr.lume.model import GPSRLUMEModel


class LitGPSRLUME(L.LightningModule):
    """Trains a ``GPSRLUMEModel``: a frozen virtual accelerator plus a trainable
    beam generator.

    The model is injected, not built here, so the harness never names a concrete
    ``BeamGenerator`` subclass. ``from_spec`` builds one from a spec instead.

    ``loss_func`` has no default and may be ``None`` for a predict-only model;
    ``training_step`` / ``test_step`` raise if it was never supplied.

        lit = LitGPSRLUME(model, lr=1e-3, loss_func=kl_div_loss)

    ``on_save_checkpoint`` drops the frozen ``lume_cheetah_model.*`` state and
    embeds the construction spec, so a checkpoint loads either standalone
    (``load_self_contained``) or by re-supplying a model
    (``load_from_checkpoint(ckpt, gpsr_lume_model=model)``).
    """

    # Prefix of the fixed virtual accelerator's parameters/buffers within the state_dict.
    _LUME_PREFIX = "gpsr_lume_model.lume_cheetah_model."
    # Checkpoint key under which the self-contained construction spec is embedded.
    _SPEC_KEY = "gpsr_lume_spec"

    def __init__(
        self,
        gpsr_lume_model: GPSRLUMEModel,
        lr: float = 1e-3,
        loss_func: Callable | str | None = None,
    ):
        super().__init__()
        # Store as an import-path string *before* save_hyperparameters so the
        # hparams stay JSON-pure; a bare function would be pickled, forcing
        # weights_only=False at load.
        loss_func = to_import_path(loss_func) if loss_func is not None else None
        self.save_hyperparameters(ignore=["gpsr_lume_model"], logger=False)
        self.gpsr_lume_model = gpsr_lume_model
        self.lr = lr
        self.loss_func = import_from_path(loss_func) if loss_func is not None else None
        self.source_info = None  # `setup` pulls this from the datamodule

        # The frozen accelerator's keys are stripped at save, so tolerate their
        # absence; `on_load_checkpoint` backfills them on the everyday path.
        self.strict_loading = False

    @classmethod
    def from_spec(cls, spec: dict, **kwargs) -> "LitGPSRLUME":
        """Build a ``LitGPSRLUME`` with a fresh (untrained) model from a spec.

        ``model_spec_from_files`` builds a spec from lattice / name-map JSON.

        Parameters
        ----------
        spec : dict
            Canonical spec (see ``gpsr.lume.builders``).
        **kwargs
            Remaining ``LitGPSRLUME`` arguments (``lr``, ``loss_func``).
        """
        return cls(build_gpsr_lume_model(spec), **kwargs)

    @classmethod
    def load_self_contained(cls, checkpoint_path, **kwargs) -> "LitGPSRLUME":
        """Load a self-contained checkpoint, rebuilding the model from its spec.

        Rebuilds the ``GPSRLUMEModel`` from the embedded spec and restores the
        trained ``beam_generator`` weights over it.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to a checkpoint with an embedded spec (the usual case).
        **kwargs
            Forwarded to ``load_from_checkpoint`` (e.g. ``map_location``,
            ``weights_only``). The checkpoint is JSON-pure, so it loads with the
            PyTorch default ``weights_only=True``.

        Raises
        ------
        KeyError
            If the checkpoint has no embedded spec (its generator could not
            describe itself at save time); use ``load_from_checkpoint`` with a
            re-supplied ``gpsr_lume_model`` instead.
        """
        kwargs.setdefault("weights_only", True)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=kwargs.get("map_location"),
            weights_only=kwargs["weights_only"],
        )
        if cls._SPEC_KEY not in checkpoint:
            raise KeyError(
                f"Checkpoint {checkpoint_path!r} has no embedded "
                f"{cls._SPEC_KEY!r}; its beam generator could not describe "
                f"itself at save time. Use load_from_checkpoint(..., "
                f"gpsr_lume_model=...) with a re-supplied model instead."
            )
        gpsr_lume_model = build_gpsr_lume_model(checkpoint[cls._SPEC_KEY])
        return cls.load_from_checkpoint(
            checkpoint_path, gpsr_lume_model=gpsr_lume_model, **kwargs
        )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Score one batch, logged as ``test_loss``.

        Run with ``L.Trainer(..., inference_mode=False)``. The default
        ``inference_mode=True`` produces inference tensors that do not track a
        version counter, which breaks Cheetah's transfer-map caching during
        ``track``. ``inference_mode=False`` falls back to ``torch.no_grad()``,
        which still disables autograd without that issue.
        """
        loss = self._shared_step(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def setup(self, stage):
        """Pull per-source data metadata from the Trainer's datamodule.

        Sourced here rather than in ``__init__`` to keep it out of the checkpoint.
        The datamodule must expose ``source_info`` (``GPSRLUMEDataModule`` does).
        Bare ``load_from_checkpoint`` (no Trainer) leaves it ``None``.
        """
        self.source_info = self.trainer.datamodule.source_info

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Strip the frozen virtual accelerator's state; embed the construction spec.

        Drops the large ``lume_cheetah_model.*`` subtree, keeping only trained
        ``beam_generator.*`` weights, then embeds the construction spec under
        ``checkpoint[_SPEC_KEY]``. A generator whose ``get_config`` raises
        ``NotImplementedError`` is skipped with a warning: the checkpoint stays
        valid but must then be loaded by re-supplying a model.
        """
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith(self._LUME_PREFIX)
        }

        try:
            checkpoint[self._SPEC_KEY] = serialize_gpsr_lume_model(self.gpsr_lume_model)
        except NotImplementedError as err:
            warnings.warn(
                f"Could not embed a self-contained spec: {err} "
                f"The checkpoint holds only the trained beam_generator weights; "
                f"load it by re-supplying a model via "
                f"load_from_checkpoint(ckpt, gpsr_lume_model=...).",
                stacklevel=2,
            )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Backfill the frozen virtual accelerator's stripped state before loading.

        ``on_save_checkpoint`` dropped the ``lume_cheetah_model.*`` subtree, so
        the checkpoint is missing those keys by construction. This instance's
        accelerator (rebuilt from the spec, or injected) holds the values to keep,
        so copying them in completes the load and silences Lightning's
        missing-keys warning, which would otherwise list every element parameter.

        Only the frozen subtree is backfilled: a missing or unexpected
        ``beam_generator.*`` key still surfaces, since that would mean a genuinely
        mismatched generator.
        """
        checkpoint["state_dict"].update(
            {
                k: v
                for k, v in self.state_dict().items()
                if k.startswith(self._LUME_PREFIX)
            }
        )

    def _shared_step(self, batch):
        """Compute the multi-source reconstruction loss for one batch.

        Predicts via ``GPSRLUMEModel.predict_multi_source`` (one shared beam
        sample, joint across sources), then averages the per-PV loss over all
        observable PVs and all sources.

        Parameters
        ----------
        batch : dict[str, TensorDict]
            Multi-source batch. ``batch[source_name]`` -> TensorDict with keys
            ``"beamline_settings"`` and ``"observations"``.

        Returns
        -------
        torch.Tensor
            Scalar loss averaged across all observable PVs and all sources.

        Raises
        ------
        ValueError
            If no ``loss_func`` was supplied at construction -- the harness has no
            default reconstruction loss.
        """
        if self.loss_func is None:
            raise ValueError(
                "No 'loss_func' was supplied, so there is nothing to score this "
                "batch with. Pass one at construction, e.g. "
                "LitGPSRLUME(model, loss_func=gpsr.losses.kl_div_loss); it is only "
                "optional for a model built to predict rather than to fit."
            )

        predictions_by_source = self.gpsr_lume_model.predict_multi_source(
            batch, self.source_info
        )

        total_loss = 0.0
        for source_name, predictions in predictions_by_source.items():
            targets = dict(batch[source_name]["observations"])

            loss = 0.0
            for pv in targets:
                loss += self.loss_func(
                    normalize_images(targets[pv]), normalize_images(predictions[pv])
                )

            loss /= len(targets)
            total_loss += loss

        return total_loss / len(predictions_by_source)
