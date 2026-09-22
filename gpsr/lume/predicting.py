"""
Prediction: track a beam through the model to get predicted observations.

The public entry points mirror the plotting API and form a single/ensemble x
single-source/multi-source grid:

- :func:`predict_images` — single beam (sampled from the model's generator),
  returns ``(n_samples, W, H)`` per PV. Feed to
  :func:`gpsr.lume.plotting.plot_images`.
- :func:`predict_ensemble_images` — vectorized ensemble beam, returns
  ``(n_draws, n_samples, W, H)`` per PV. Feed to
  :func:`gpsr.lume.plotting.plot_ensemble_images`.
- :func:`predict_multi_source_images` — single beam shared across sources,
  returns ``{source_name: {pv: (n_samples, W, H)}}``.
- :func:`predict_multi_source_ensemble_images` — ensemble beam shared across
  sources, returns ``{source_name: {pv: (n_draws, n_samples, W, H)}}``.

The multi-source functions take a ``sources`` spec (see :func:`_unpack_source`);
:meth:`GPSRLUMEDataModule.to_sources_spec` builds one from a datamodule without
dragging in the (heavy, and here unused) measured observations.

Only screen (image) observations are supported today; this module is the intended
home for future per-observation-type prediction logic as more observation types
(e.g. scalar diagnostics) are added.
"""

from __future__ import annotations

import torch

from cheetah.particles import ParticleBeam

from gpsr.losses import normalize_images


def _add_scan_broadcast_axis(beam: ParticleBeam) -> ParticleBeam:
    """Return a copy of ``beam`` with a size-1 scan-step axis inserted.

    An ensemble beam has particles of shape ``(n_draws, n_particles, 7)`` -- its
    leading (draw) dim would collide with the settings' leading (scan-step) dim
    during tracking. Cheetah broadcasts the beam's batch dims against each
    element's parameter batch dims (e.g. quad ``k1``) aligned from the *right*, so
    a bare ``(n_draws,)`` beam batch lands on the same axis as the settings'
    ``(n_samples,)`` -> collision unless one is 1. Inserting a size-1 axis just
    before the particle dim makes the beam batch ``(n_draws, 1)``, which broadcasts
    against ``(n_samples,)`` to the outer product ``(n_draws, n_samples)`` -- so a
    single ``track`` yields per-PV images of shape ``(n_draws, n_samples, W, H)``.

    Only the per-particle tensors that actually carry a draw dim get the axis: the
    ``list_to_beam`` build path stacks only ``particles`` (leaving
    ``particle_charges`` / ``survival_probabilities`` at ``(n_particles,)``, which
    already broadcast), while a beam whose buffers were also stacked to
    ``(n_draws, n_particles)`` needs the axis on them too. The original beam is left
    untouched; device/dtype/species are preserved by reusing the buffers.
    """

    def _maybe_unsqueeze(buffer):
        # Insert the scan-step axis only if the buffer carries the draw dim.
        return buffer.unsqueeze(-2) if buffer.dim() > 1 else buffer

    return ParticleBeam(
        particles=beam.particles.unsqueeze(-3),
        energy=beam.energy,
        particle_charges=_maybe_unsqueeze(beam.particle_charges),
        survival_probabilities=_maybe_unsqueeze(beam.survival_probabilities),
        s=beam.s,
        species=beam.species,
    )


def _slice_beam_draws(beam: ParticleBeam, start: int, stop: int) -> ParticleBeam:
    """Return a copy of an ensemble ``beam`` restricted to draws ``[start:stop)``.

    Slices along the leading draw dim of the per-particle tensors that carry it.
    Mirrors :func:`_add_scan_broadcast_axis`: ``particles`` always carries the draw
    dim; ``particle_charges`` / ``survival_probabilities`` may sit at
    ``(n_particles,)`` (broadcast across draws -> passed through) or
    ``(n_draws, n_particles)`` (sliced). ``energy`` / ``s`` / ``species`` do not
    carry a draw dim in this codebase's ensemble beams and are reused as-is. The
    original beam is left untouched; device/dtype/species are preserved.
    """

    def _maybe_slice(buffer):
        # Slice the draw dim only if the buffer carries it.
        return buffer[start:stop] if buffer.dim() > 1 else buffer

    return ParticleBeam(
        particles=beam.particles[start:stop],
        energy=beam.energy,
        particle_charges=_maybe_slice(beam.particle_charges),
        survival_probabilities=_maybe_slice(beam.survival_probabilities),
        s=beam.s,
        species=beam.species,
    )


@torch.no_grad()
def predict_images(
    model,
    beamline_settings: dict,
    observations_metadata: dict,
    beam: ParticleBeam | None = None,
    beamline_constants: dict | None = None,
    normalize: bool = True,
) -> dict:
    """Inference wrapper: call the model correctly for prediction.

    Packages the non-obvious inference gotchas so callers don't repeat them:
    ``@torch.no_grad`` (not ``inference_mode``, which breaks Cheetah's
    transfer-map cache during ``track``), and unit-sum normalization matching
    :meth:`LitGPSRLUME._shared_step` so predictions are directly comparable to
    training loss. The physics -- beam sampling, screen setup, tracking -- lives
    in :meth:`GPSRLUMEModel.forward`, and renders the same ``cloud-in-cell``
    image the fit took gradients through, so a prediction is comparable to the
    training objective rather than to a differently imaged version of it.

    The normalization is :func:`gpsr.losses.normalize_images` -- each image
    divided by its own sum. Note what that hides: a predicted beam that landed
    entirely off the sensor holds only numerical dust, and normalizing rescales it
    into a plausible-looking blob. Read ``normalize=False`` sums if you need to
    tell a dim image from a miss.

    Decoupled from any datamodule: the settings can be a measured scan's *or*
    arbitrary values the model never trained on (e.g. quad strengths between
    scan points).

    A single entry of a ``sources`` spec (see :func:`_unpack_source`) is exactly
    this function's per-source arguments, so one source can be predicted by
    splatting it: ``predict_images(model, **datamodule.to_sources_spec()[name])``.

    The output fills the ``observations`` slot of :func:`gpsr.lume.plotting.plot_images`::

        preds = predict_images(model, settings, metadata)
        plot_images(settings, preds, metadata)             # predicted images alone

        # or overlay predicted (bottom) against measured (top):
        plot_images(**dataset.to_dict(), overlay_images=preds)

    For ensemble predictions (vectorized beam, leading draw dim) use
    :func:`predict_ensemble_images` instead.

    The model and setting tensors must already share a device (this bypasses the
    Trainer, so nothing moves them) -- load eval on CPU; see the device note in
    AGENTS.md.

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    beamline_settings : dict[str, Tensor]
        Scan parameters keyed by PV name; each tensor's leading dim is n_samples.
    observations_metadata : dict[str, dict]
        Per-observation metadata (``type``, ``shape``, ``pixel_size``, ...) whose
        keys are the PVs to predict. Passed through to
        :meth:`GPSRLUMEModel.forward`, which applies the screen configuration.
    beam : ParticleBeam | None
        Passed through to :meth:`GPSRLUMEModel.forward`. If ``None``, forward
        samples one from ``model.gpsr_lume_model.beam_generator()``. Used by
        :func:`predict_ensemble_images` to inject a pre-built ensemble beam;
        end-users generally leave this ``None``.
    beamline_constants : dict[str, Tensor] | None
        Fixed parameters folded into the settings before tracking (mirrors
        :meth:`GPSRLUMEModel.predict_multi_source`'s per-source constants). Keys
        must not overlap ``beamline_settings``.
    normalize : bool, default=True
        If True, each predicted image is normalized so its pixel intensities sum
        to 1 (via :func:`gpsr.losses.normalize_images`) -- the same normalization
        :meth:`LitGPSRLUME._shared_step` applies before scoring the loss.

    Returns
    -------
    dict[str, Tensor]
        Predicted images keyed by observation PV. Shape ``(n_samples, W, H)``.
        Normalized to unit sum per image unless ``normalize=False``.
    """
    # `or {}` would call bool() on a TensorDict, which raises; test None explicitly.
    beamline_constants = beamline_constants if beamline_constants is not None else {}
    predictions = model.gpsr_lume_model(
        settings=dict(beamline_settings) | dict(beamline_constants),
        observations_metadata=observations_metadata,
        beam=beam,
    )
    if normalize:
        predictions = {pv: normalize_images(image) for pv, image in predictions.items()}
    return predictions


@torch.no_grad()
def predict_ensemble_images(
    model,
    beamline_settings: dict,
    observations_metadata: dict,
    beam: ParticleBeam,
    beamline_constants: dict | None = None,
    normalize: bool = True,
    chunk_size: int | None = None,
) -> dict:
    """Return predicted screen images for a vectorized ensemble of beams.

    Injects a pre-built vectorized ``ParticleBeam`` (``particles`` of shape
    ``(n_draws, n_particles, 7)``) and tracks it through the configured
    accelerator. A size-1 scan-step axis is inserted internally (see
    :func:`_add_scan_broadcast_axis`) so the draw dim and the settings' scan-step
    dim form an outer product in a single ``track`` call. The output carries a
    leading draw dim::

        preds = predict_ensemble_images(model, settings, metadata, beam=beam, chunk_size=10)
        plot_ensemble_images(settings, preds, metadata)

    For single-beam prediction (sampling from the model's generator) use
    :func:`predict_images` instead.

    ``torch.no_grad`` (not ``inference_mode``) is used because inference tensors
    break Cheetah's transfer-map caching during ``track``. The model, beam, and
    setting tensors must already share a device.

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    beamline_settings : dict[str, Tensor]
        Scan parameters keyed by PV name; each tensor's leading dim is n_samples.
    observations_metadata : dict[str, dict]
        Per-observation metadata (``type``, ``shape``, ``pixel_size``, ...) used to
        configure the observable elements. Its keys are the PVs predicted.
    beam : ParticleBeam
        Vectorized ensemble beam; ``particles`` must have shape
        ``(n_draws, n_particles, 7)``. The size-1 scan-step broadcast axis is
        inserted internally.
    beamline_constants : dict[str, Tensor] | None
        Fixed parameters folded into the settings before tracking. Keys must not
        overlap ``beamline_settings``.
    normalize : bool, default=True
        If True, each predicted image is normalized to unit pixel sum via
        :func:`gpsr.losses.normalize_images`. Slicing the draw dim
        for chunking is exact because each image is normalized independently.
    chunk_size : int | None, default=None
        If given, the draw dim is tracked in slices of at most ``chunk_size`` draws
        and the per-chunk images are concatenated to the full
        ``(n_draws, n_samples, W, H)`` result. This caps peak memory at the
        ``chunk_size x n_samples`` outer product without changing the output.

    Returns
    -------
    dict[str, Tensor]
        Predicted images keyed by observation PV. Shape
        ``(n_draws, n_samples, W, H)``. Normalized to unit sum per image unless
        ``normalize=False``.
    """
    beam = _add_scan_broadcast_axis(beam)

    if chunk_size is None:
        return predict_images(
            model,
            beamline_settings,
            observations_metadata,
            beam=beam,
            beamline_constants=beamline_constants,
            normalize=normalize,
        )

    n_draws = beam.particles.shape[0]
    observables_pvs = list(observations_metadata.keys())
    chunk_preds = [
        predict_images(
            model,
            beamline_settings,
            observations_metadata,
            beam=_slice_beam_draws(beam, start, start + chunk_size),
            beamline_constants=beamline_constants,
            normalize=normalize,
        )
        for start in range(0, n_draws, chunk_size)
    ]
    return {
        pv: torch.cat([chunk[pv] for chunk in chunk_preds], dim=0)
        for pv in observables_pvs
    }


def _unpack_source(source_name: str, source: dict) -> tuple[dict, dict, dict]:
    """Validate one entry of a ``sources`` spec and return its three pieces.

    A ``sources`` spec is a mapping ``{source_name: source}`` where each
    ``source`` is a dict with keys mirroring :func:`predict_images`'s per-source
    arguments::

        {
            "beamline_settings": {pv: Tensor},        # required
            "observations_metadata": {pv: dict},      # required
            "beamline_constants": {pv: Tensor},       # optional, defaults to {}
        }

    Returns ``(beamline_settings, observations_metadata, beamline_constants)``.
    Kept deliberately observation-free: unlike a :class:`GPSRLUMEDataset`, a
    source needs no measured images, so predictions can be made at arbitrary
    settings the model never trained on (mirrors :func:`predict_images`'s
    datamodule-decoupled design).
    """
    missing = {"beamline_settings", "observations_metadata"} - source.keys()
    if missing:
        raise ValueError(
            f"Source '{source_name}' is missing required key(s) {sorted(missing)}; "
            f"got keys {sorted(source.keys())}."
        )
    return (
        source["beamline_settings"],
        source["observations_metadata"],
        source.get("beamline_constants"),
    )


@torch.no_grad()
def predict_multi_source_images(
    model,
    sources_spec: dict[str, dict],
    beam: ParticleBeam | None = None,
    normalize: bool = True,
) -> dict:
    """Predict single-beam screen images for every source against one shared beam.

    The multi-source analogue of :func:`predict_images`. Samples a single beam
    once (or accepts a pre-built one) and tracks *that same beam* through every
    source, so the reconstruction is jointly consistent across sources -- the
    same shared-beam semantics as :meth:`GPSRLUMEModel.predict_multi_source`.
    This is why the loop cannot simply call :func:`predict_images` with
    ``beam=None`` per source: each such call would sample an *independent* beam
    and the sources would no longer share a distribution.

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    sources_spec : dict[str, dict]
        Per-source spec keyed by source name; each value is unpacked by
        :func:`_unpack_source` into ``beamline_settings``,
        ``observations_metadata`` and optional ``beamline_constants``. Build one
        from a datamodule with :meth:`GPSRLUMEDataModule.to_sources_spec`.
    beam : ParticleBeam | None
        Beam shared across all sources. If ``None``, one is sampled from
        ``model.gpsr_lume_model.beam_generator()`` and shared -- so every source
        is evaluated against the same draw.
    normalize : bool, default=True
        Per-image unit-sum normalization, applied per source (see
        :func:`predict_images`).

    Returns
    -------
    dict[str, dict[str, Tensor]]
        Per source: predicted images keyed by observation PV, shape
        ``(n_samples, W, H)``.
    """
    if beam is None:
        beam = model.gpsr_lume_model.beam_generator()  # one sample, shared below
    predictions = {}
    for source_name, source in sources_spec.items():
        settings, metadata, constants = _unpack_source(source_name, source)
        predictions[source_name] = predict_images(
            model,
            settings,
            metadata,
            beam=beam,
            beamline_constants=constants,
            normalize=normalize,
        )
    return predictions


@torch.no_grad()
def predict_multi_source_ensemble_images(
    model,
    sources_spec: dict[str, dict],
    beam: ParticleBeam,
    normalize: bool = True,
    chunk_size: int | None = None,
) -> dict:
    """Predict ensemble screen images for every source against one shared beam.

    The multi-source analogue of :func:`predict_ensemble_images`. The single
    pre-built ensemble ``beam`` is shared across all sources (joint
    reconstruction); each per-source call re-inserts the size-1 scan-step
    broadcast axis on its own copy, so sharing the raw beam is safe and
    ``chunk_size`` behaves exactly as in the single-source case.

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    sources_spec : dict[str, dict]
        Per-source spec keyed by source name (see :func:`_unpack_source` and
        :meth:`GPSRLUMEDataModule.to_sources_spec`).
    beam : ParticleBeam
        Vectorized ensemble beam shared across sources; ``particles`` must have
        shape ``(n_draws, n_particles, 7)``.
    normalize : bool, default=True
        Per-image unit-sum normalization (see :func:`predict_ensemble_images`).
    chunk_size : int | None, default=None
        Draw-dim chunking, threaded through to :func:`predict_ensemble_images`
        for each source to cap peak memory without changing the output.

    Returns
    -------
    dict[str, dict[str, Tensor]]
        Per source: predicted images keyed by observation PV, shape
        ``(n_draws, n_samples, W, H)``.
    """
    predictions = {}
    for source_name, source in sources_spec.items():
        settings, metadata, constants = _unpack_source(source_name, source)
        predictions[source_name] = predict_ensemble_images(
            model,
            settings,
            metadata,
            beam=beam,
            beamline_constants=constants,
            normalize=normalize,
            chunk_size=chunk_size,
        )
    return predictions
