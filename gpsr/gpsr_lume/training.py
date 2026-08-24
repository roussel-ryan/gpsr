import warnings
from typing import Callable

import lightning as L
import torch

from gpsr.losses import normalize_images

from gpsr.gpsr_lume._imports import import_from_path, to_import_path
from gpsr.gpsr_lume.builders import (
    build_gpsr_lume_model,
    serialize_gpsr_lume_model,
)
from gpsr.gpsr_lume.model import GPSRLUMEModel

# Default reconstruction loss, stored as an import path so it stays JSON-pure in
# the checkpoint hyperparameters (a bare function object would be pickled).
# DEFAULT_LOSS = "gpsr.losses.mae_loss"
DEFAULT_LOSS = "gpsr.gpsr_lume.training.kl_div_loss"

# Sum of a screen image PV when the whole beam lands on the sensor. Cheetah's
# `Screen.reading` is normalised to unit total over the pixels, so it sums to
# exactly the fraction of beam charge that reached the sensor; the virtual
# accelerator's `ScreenImageVariable` then scales that reading by 65535 to give
# camera-like counts. An image PV therefore sums to
# ``IMAGE_FULL_SCALE * on-screen charge fraction``. Override
# ``LitGPSRLUME(image_full_scale=...)`` if that PV scaling ever changes.
IMAGE_FULL_SCALE = 65535.0

# On-screen charge fraction below which a predicted image is treated as "the beam
# missed the sensor" rather than as a dim image -- see
# :func:`normalize_images_floored`. Predictions are bimodal in practice (either
# ~1.0 of the charge lands or ~1e-17 does), so anything in (1e-6, 0.5) separates
# the two; 1% is a wide margin that leaves genuinely clipped beams untouched.
DEFAULT_ONSCREEN_FLOOR = 1e-2


class LitGPSRLUME(L.LightningModule):
    """Trains a ``GPSRLUMEModel``: a frozen LUME-Cheetah virtual accelerator plus
    a trainable beam generator (the only trainable piece).

    The model is *injected*, not built here -- ``__init__`` takes an assembled
    ``GPSRLUMEModel``, so the harness stays decoupled from how it was constructed
    and never names a concrete ``BeamGenerator`` subclass (it is
    generator-agnostic). Use :meth:`from_spec` for the common "build from a spec"
    path (pair it with :func:`gpsr.gpsr_lume.builders.model_spec_from_files` to build a
    spec from lattice / name-map JSON files).

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
        loss_func: Callable | str = DEFAULT_LOSS,
        image_full_scale: float = IMAGE_FULL_SCALE,
        onscreen_floor: float = DEFAULT_ONSCREEN_FLOOR,
    ):
        super().__init__()
        # Store loss_func as an import-path string *before* save_hyperparameters so
        # the checkpoint's hparams stay JSON-pure (a bare function would be pickled,
        # forcing weights_only=False at load). Accepts a callable or a path string.
        loss_func = to_import_path(loss_func)
        # gpsr_lume_model is an injected nn.Module, not a hyperparameter: it cannot
        # be captured by save_hyperparameters and is re-supplied at load.
        self.save_hyperparameters(ignore=["gpsr_lume_model"], logger=False)
        self.gpsr_lume_model = gpsr_lume_model
        self.lr = lr
        self.loss_func = import_from_path(loss_func)
        self.image_full_scale = image_full_scale
        self.onscreen_floor = onscreen_floor
        # source_info is data metadata, not a hyperparameter; `setup` pulls it from
        # the datamodule so it stays out of the checkpoint.
        self.source_info = None

        # The frozen virtual accelerator's state is stripped at save and re-supplied
        # by injection at load, so allow its keys to be missing from the state_dict.
        self.strict_loading = False

    @classmethod
    def from_spec(cls, spec: dict, **kwargs) -> "LitGPSRLUME":
        """Build a ``LitGPSRLUME`` from a canonical model spec.

        The one convenience constructor: builds a fresh (untrained)
        ``GPSRLUMEModel`` from the spec via
        :func:`gpsr.gpsr_lume.builders.build_gpsr_lume_model`, rather than injecting a
        pre-calibrated one. The spec is the single representation used everywhere --
        fresh build, checkpoint embed, and self-contained reload all speak it.

        Build a spec from lattice / name-map JSON files with
        :func:`gpsr.gpsr_lume.builders.model_spec_from_files`::

            spec = model_spec_from_files(lattice_path, name_map_path, energy,
                                         accelerator_builder=BUILDER_IMPORT_PATH,
                                         generator={"cls": NNParticleBeamGenerator,
                                                    "config": {"n_particles": 10000}})
            lit = LitGPSRLUME.from_spec(spec, lr=1e-3)

        Parameters
        ----------
        spec : dict
            Canonical spec (see :mod:`gpsr.gpsr_lume.builders`).
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
        """
        predictions_by_source = self.gpsr_lume_model.predict_multi_source(
            batch, self.source_info
        )

        total_loss = 0.0
        onscreen = []
        for source_name, predictions in predictions_by_source.items():
            targets = dict(batch[source_name]["observations"])

            loss = 0.0
            for pv in targets:
                # The target is a real camera frame: its own sum *is* the measurement,
                # so plain self-normalization is right. The prediction is not -- see
                # normalize_images_floored.
                normalized_target = normalize_images(targets[pv])
                centered_target = center_images(normalized_target)
                normalized_prediction = normalize_images_floored(
                    predictions[pv], self.image_full_scale, self.onscreen_floor
                )
                centered_prediction = center_images(normalized_prediction)
                # loss += self.loss_func(normalized_target, normalized_prediction)
                loss += self.loss_func(centered_target, centered_prediction)
                onscreen.append(
                    predictions[pv].detach().flatten(start_dim=-2).sum(-1).mean()
                    / self.image_full_scale
                )

            loss /= len(targets)
            total_loss += loss

        total_loss /= len(predictions_by_source)
        # Logged because a fit can look healthy while quietly parking scan steps off
        # the sensor; this is the metric that makes that visible.
        self.log("onscreen_fraction", torch.stack(onscreen).mean(), on_epoch=True)
        return total_loss


def normalize_images_floored(
    images: torch.Tensor,
    full_scale: float = IMAGE_FULL_SCALE,
    floor_fraction: float = DEFAULT_ONSCREEN_FLOOR,
) -> torch.Tensor:
    """Normalize *predicted* images without rescaling a beam that missed the sensor.

    ``gpsr.losses.normalize_images`` divides an image by its own sum. That is the
    right thing for a measured frame, but not for a prediction: when a predicted
    beam lands entirely off the sensor the image holds only ~1e-17 of the charge
    (numerically, the far tail of the imaging KDE), and dividing that by its own
    sum rescales the numerical dust into a smooth unit-mass blob. Under KL that
    blob scores *better* than an honest fit, so the objective ends up paying the
    generator to steer scan steps off the sensor -- and every time one drifts back
    on, the concealed error surfaces as a discrete jump in the loss.

    Flooring the denominator removes the reward without touching the healthy case.
    A screen image PV sums to ``full_scale`` times the fraction of beam charge that
    reached the sensor, so:

    - on-screen fraction above ``floor_fraction`` -- the divisor is the image's own
      sum, exactly as before;
    - below it -- the divisor stops shrinking, the image stays dark, and the loss
      charges for the missing intensity that the data says should be there.

    Parameters
    ----------
    images : torch.Tensor
        Predicted images shaped ``(..., W, H)``, in image-PV units.
    full_scale : float, default=:data:`IMAGE_FULL_SCALE`
        Sum of an image PV when the entire beam lands on the sensor.
    floor_fraction : float, default=:data:`DEFAULT_ONSCREEN_FLOOR`
        On-screen charge fraction below which the beam counts as having missed.

    Returns
    -------
    torch.Tensor
        Images of the same shape. Sums to 1 for an on-sensor beam, and to the
        on-screen fraction divided by ``floor_fraction`` for a beam that missed.
    """
    sums = images.sum(dim=(-1, -2), keepdim=True)
    return images / sums.clamp_min(floor_fraction * full_scale)


def kl_div_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Kullback-Leibler divergence ``KL(target || pred)`` as a scalar loss.

    True signed KL, ``sum_i t_i (log t_i - log p_i)``, summed over the image axes
    and averaged over the batch -- a drop-in ``loss_func`` for ``_shared_step``
    (returns a scalar, unlike ``gpsr.losses.kl_div`` which is per-pixel and takes
    an absolute value). Swap it in via the import path
    ``"gpsr.gpsr_lume.training.kl_div_loss"``.

    KL is defined between probability distributions, so ``target`` and ``pred``
    are expected pre-normalized to unit intensity (``normalize_images`` in
    ``_shared_step`` does this) -- no further rescaling is needed. The
    target-weighting (each term scales with ``t_i``) is what makes a small
    secondary peak carry weight proportional to its share of the beam's mass while
    the near-zero background contributes ~0 to both loss and gradient. KL is
    asymmetric: it penalizes *missing* target mass but is lenient about predicted
    mass where the target is zero.

    Parameters
    ----------
    target, pred : torch.Tensor
        Image tensors of matching shape ``(..., W, H)``, each normalized so the
        last two axes sum to 1.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence, summed over pixels and averaged over the batch.
    """
    eps = 1e-10
    # target * (log target - log pred), with target as the numerator so background
    # pixels (target ~ 0) vanish rather than blowing up the log ratio. Sum over the
    # image axes gives true KL; mean over the batch keeps the scale resolution- and
    # batch-independent (comparable to an MAE, so lr need not be retuned).
    per_pixel = target * ((target + eps).log() - (pred + eps).log())
    return per_pixel.sum(dim=(-1, -2)).mean()


def center_images(images: torch.Tensor) -> torch.Tensor:
    """Shift each image so its intensity centroid sits at the geometric center.

    Differentiable w.r.t. the pixel intensities: the shift is a per-image integer
    (a cyclic ``roll``, which is just a permutation of pixels, so gradients flow
    straight through). The shift *amount* is rounded from the centroid and carries
    no gradient itself; for a sub-pixel, shift-differentiable centering use a
    Fourier phase shift or ``grid_sample`` instead.

    Wrapping is cyclic (``roll``): intensity pushed off one edge reappears on the
    opposite edge. That is harmless when the beam is compact and away from the
    borders, which is the usual case after centering.

    Parameters
    ----------
    images : torch.Tensor
        Image tensor shaped ``(..., W, H)`` (last axis y/rows, second-to-last
        x/columns -- matching ``gpsr`` conventions). Any number of leading batch dims.

    Returns
    -------
    torch.Tensor
        Images of the same shape, each centered on its own centroid.
    """
    *batch_shape, width, height = images.shape
    flat = images.reshape(-1, width, height)
    n = flat.shape[0]

    # Centroid (in pixel coordinates) of each image, weighted by intensity.
    coord_x = torch.arange(width, device=images.device, dtype=images.dtype)
    coord_y = torch.arange(height, device=images.device, dtype=images.dtype)
    x_proj = flat.sum(dim=-1)  # (n, W)
    y_proj = flat.sum(dim=-2)  # (n, H)
    x_centroid = (x_proj * coord_x).sum(-1) / x_proj.sum(-1).clamp_min(1e-8)
    y_centroid = (y_proj * coord_y).sum(-1) / y_proj.sum(-1).clamp_min(1e-8)

    # Integer shift that moves each centroid to the geometric center. Detached
    # from the graph: the shift is a discrete index, not a differentiable value.
    shift_x = torch.round((width - 1) / 2 - x_centroid).long()  # (n,)
    shift_y = torch.round((height - 1) / 2 - y_centroid).long()  # (n,)

    # Per-image cyclic roll via gather (torch.roll applies one shift to the whole
    # tensor; gather lets each image roll by its own amount). rolled[i] = src[i - s].
    idx_x = (torch.arange(width, device=images.device) - shift_x[:, None]) % width
    flat = flat.gather(1, idx_x[:, :, None].expand(n, width, height))
    idx_y = (torch.arange(height, device=images.device) - shift_y[:, None]) % height
    flat = flat.gather(2, idx_y[:, None, :].expand(n, width, height))

    return flat.reshape(*batch_shape, width, height)
