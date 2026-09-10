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
    """Trains a ``GPSRLUMEModel``: a frozen LUME-Cheetah virtual accelerator plus
    a trainable beam generator (the only trainable piece).

    The model is *injected*, not built here -- ``__init__`` takes an assembled
    ``GPSRLUMEModel``, so the harness stays decoupled from how it was constructed
    and never names a concrete ``BeamGenerator`` subclass (it is
    generator-agnostic). Use :meth:`from_spec` for the common "build from a spec"
    path (pair it with :func:`gpsr.lume.builders.model_spec_from_files` to build a
    spec from lattice / name-map JSON files).

    The reconstruction loss is a *training* argument with no library default:
    ``loss_func`` takes a callable or its dotted import path, and the harness has
    no opinion on which one you fit with. It stays ``None`` for a model built only
    to predict (the predictors in :mod:`gpsr.lume.predicting` never score
    anything); ``training_step`` / ``test_step`` raise if it was never supplied.
    :func:`gpsr.losses.kl_div_loss` is the usual choice::

        lit = LitGPSRLUME(model, lr=1e-3, loss_func=kl_div_loss)

    Checkpointing
    -------------
    The injected ``GPSRLUMEModel`` is an ``nn.Module``, not a hyperparameter, so
    it is not captured by ``save_hyperparameters``. ``on_save_checkpoint`` drops
    the frozen ``lume_cheetah_model.*`` state (large, transient beam buffers) and
    embeds the model's construction spec under ``checkpoint[_SPEC_KEY]`` when the
    generator can describe itself. Two ways to load:

    - **Standalone** -- rebuild from the embedded spec::

          lit = LitGPSRLUME.load_self_contained(ckpt)

    - **Inject** -- supply a (e.g. differently calibrated) model; embedded
      artifacts are ignored and trained ``beam_generator`` weights are restored::

          lit = LitGPSRLUME.load_from_checkpoint(ckpt, gpsr_lume_model=model)
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
        # Store loss_func as an import-path string *before* save_hyperparameters so
        # the checkpoint's hparams stay JSON-pure (a bare function would be pickled,
        # forcing weights_only=False at load). Accepts a callable or a path string.
        loss_func = to_import_path(loss_func) if loss_func is not None else None
        # gpsr_lume_model is an injected nn.Module, not a hyperparameter: it cannot
        # be captured by save_hyperparameters and is re-supplied at load.
        self.save_hyperparameters(ignore=["gpsr_lume_model"], logger=False)
        self.gpsr_lume_model = gpsr_lume_model
        self.lr = lr
        self.loss_func = import_from_path(loss_func) if loss_func is not None else None
        # source_info is data metadata, not a hyperparameter; `setup` pulls it from
        # the datamodule so it stays out of the checkpoint.
        self.source_info = None

        # The frozen virtual accelerator's state is stripped at save and re-supplied
        # by injection at load, so allow its keys to be missing from the state_dict.
        # `on_load_checkpoint` backfills them, so this tolerance is a safety net for
        # checkpoints written by other versions rather than the everyday path.
        self.strict_loading = False

    @classmethod
    def from_spec(cls, spec: dict, **kwargs) -> "LitGPSRLUME":
        """Build a ``LitGPSRLUME`` from a canonical model spec.

        The one convenience constructor: builds a fresh (untrained)
        ``GPSRLUMEModel`` from the spec via
        :func:`gpsr.lume.builders.build_gpsr_lume_model`, rather than injecting a
        pre-calibrated one. The spec is the single representation used everywhere --
        fresh build, checkpoint embed, and self-contained reload all speak it.

        Build a spec from lattice / name-map JSON files with
        :func:`gpsr.lume.builders.model_spec_from_files`::

            spec = model_spec_from_files(lattice_path, name_map_path, energy,
                                         accelerator_builder=BUILDER_IMPORT_PATH,
                                         generator={"cls": NNParticleBeamGenerator,
                                                    "config": {"n_particles": 10000}})
            lit = LitGPSRLUME.from_spec(spec, lr=1e-3)

        Parameters
        ----------
        spec : dict
            Canonical spec (see :mod:`gpsr.lume.builders`).
        **kwargs
            Remaining ``LitGPSRLUME`` arguments (``lr``, ``loss_func``).
        """
        return cls(build_gpsr_lume_model(spec), **kwargs)

    @classmethod
    def load_self_contained(cls, checkpoint_path, **kwargs) -> "LitGPSRLUME":
        """Load a self-contained checkpoint, rebuilding the model from its spec.

        Rebuilds the (calibrated) ``GPSRLUMEModel`` from the embedded spec via
        :func:`build_gpsr_lume_model` and restores the trained ``beam_generator``
        weights over it -- no externally supplied model needed.

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
        """Training step: shared loss, logged as ``loss``."""
        loss = self._shared_step(batch)
        self.log("loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step: same predict-and-score logic, logged as ``test_loss``.

        Run with ``L.Trainer(..., inference_mode=False)``. The default
        ``inference_mode=True`` produces inference tensors that do not track a
        version counter, which breaks Cheetah's transfer-map caching during the
        virtual accelerator's ``track``. ``inference_mode=False`` falls back to
        ``torch.no_grad()``, which still disables autograd without that issue.
        """
        loss = self._shared_step(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def setup(self, stage):
        """Pull per-source data metadata from the Trainer's datamodule.

        ``source_info`` is data metadata, not a hyperparameter, so it is sourced
        here (for every stage) rather than through ``__init__`` -- keeping it out
        of the checkpoint. The datamodule passed to ``Trainer.fit``/``Trainer.test``
        must expose ``source_info`` (``GPSRLUMEDataModule`` does). Bare
        ``load_from_checkpoint`` (no Trainer) leaves it ``None``.
        """
        self.source_info = self.trainer.datamodule.source_info

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Strip the frozen virtual accelerator's state; embed the construction spec.

        Drops the large ``lume_cheetah_model.*`` subtree (transient beam buffers),
        keeping only trained ``beam_generator.*`` weights, then embeds the model's
        construction spec under ``checkpoint[_SPEC_KEY]`` so it can be rebuilt
        standalone (see :meth:`load_self_contained`). A generator that cannot
        describe itself (``get_config`` raises ``NotImplementedError``) is skipped
        with a warning -- the checkpoint stays valid but must then be loaded by
        re-supplying a model.
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

        ``on_save_checkpoint`` dropped the ``lume_cheetah_model.*`` subtree, so a
        checkpoint's ``state_dict`` is missing those keys by construction. The
        accelerator on *this* instance was already rebuilt from the embedded spec
        (:meth:`load_self_contained`) or injected by the caller, so its tensors
        hold the values to keep -- copying them into the incoming ``state_dict``
        makes the load complete and silences Lightning's "keys that are in the
        model state dict but not in the checkpoint" warning, which would otherwise
        list every element parameter on every load.

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

        Delegates prediction to :meth:`GPSRLUMEModel.predict_multi_source`
        (one shared beam sample, joint reconstruction across sources), then
        reduces the per-source predictions vs. targets to a single scalar loss,
        averaged over all observable PVs and all sources. Shared by
        ``training_step`` and ``test_step`` so train and eval use identical
        predict-and-score logic; the callers only differ in what they log.

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
                # Both sides are normalized to unit intensity, so the loss scores
                # distribution shape rather than brightness. Position is *not*
                # normalized away: the images are compared where they landed, so a
                # predicted beam offset from the measured one is penalized. (An
                # earlier version centered both sides on their own centroid via
                # `gpsr.losses.center_images_on_centroid`, which made the loss blind
                # to that offset; the helper is still there if you want it back.)
                loss += self.loss_func(
                    normalize_images(targets[pv]), normalize_images(predictions[pv])
                )

            loss /= len(targets)
            total_loss += loss

        return total_loss / len(predictions_by_source)
