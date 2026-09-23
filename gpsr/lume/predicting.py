"""Track a beam through the model to get predicted observations.

Four entry points, over single/ensemble beams x single/multi-source:

- ``predict_images`` -> ``(n_samples, W, H)`` per PV
- ``predict_ensemble_images`` -> ``(n_draws, n_samples, W, H)`` per PV
- ``predict_multi_source_images`` -> ``{source: {pv: (n_samples, W, H)}}``
- ``predict_multi_source_ensemble_images``
  -> ``{source: {pv: (n_draws, n_samples, W, H)}}``

The multi-source pair takes a ``sources`` spec from
``GPSRLUMEDataModule.to_sources_spec``. Only screen observations are supported.
"""

from __future__ import annotations

import torch

from cheetah.particles import ParticleBeam

from gpsr.losses import normalize_images


def _add_scan_broadcast_axis(beam: ParticleBeam) -> ParticleBeam:
    """Return a copy of ``beam`` with a size-1 scan-step axis inserted.

    Cheetah aligns the beam's batch dims against each element's parameter batch
    dims from the *right*, so a bare ``(n_draws,)`` beam batch would collide with
    the settings' ``(n_samples,)``. Making the beam batch ``(n_draws, 1)`` instead
    broadcasts to the outer product, so one ``track`` yields per-PV images of
    shape ``(n_draws, n_samples, W, H)``.

    Only buffers that carry a draw dim get the axis; ``list_to_beam`` leaves
    ``particle_charges`` / ``survival_probabilities`` at ``(n_particles,)``, which
    already broadcast. The original beam is left untouched.
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

    ``particles`` always carries the draw dim; ``particle_charges`` /
    ``survival_probabilities`` are sliced only if they do too. ``energy`` / ``s``
    / ``species`` never do and are reused as-is.
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
    """Predict screen images for a single beam sampled from the model.

    Uses ``torch.no_grad`` rather than ``inference_mode``, whose tensors break
    Cheetah's transfer-map cache during ``track``. The physics lives in
    ``GPSRLUMEModel.forward``.

    Normalization divides each image by its own sum, matching
    ``LitGPSRLUME._shared_step`` so predictions are comparable to the training
    loss. Beware what it hides: a beam that landed entirely off the sensor holds
    only numerical dust, which normalizing rescales into a plausible-looking blob.
    Read ``normalize=False`` sums to tell a dim image from a miss.

    The settings need not come from a datamodule -- arbitrary values the model
    never trained on work too (e.g. quad strengths between scan points).

    The model and setting tensors must already share a device; this bypasses the
    Trainer, so nothing moves them.

        preds = predict_images(model, settings, metadata)
        plot_images(settings, preds, metadata)

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    beamline_settings : dict[str, Tensor]
        Scan parameters keyed by PV name; each tensor's leading dim is n_samples.
    observations_metadata : dict[str, dict]
        Per-observation metadata (``type``, ``shape``, ``pixel_size``, ...); its
        keys are the PVs to predict.
    beam : ParticleBeam | None
        If ``None``, one is sampled from the model's generator. Used by
        ``predict_ensemble_images`` to inject a pre-built ensemble beam.
    beamline_constants : dict[str, Tensor] | None
        Fixed parameters folded into the settings before tracking. Keys must not
        overlap ``beamline_settings``.
    normalize : bool, default=True
        Normalize each image to unit pixel sum.

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

    Tracks a pre-built ensemble beam, inserting a size-1 scan-step axis so the
    draw and scan-step dims form an outer product in a single ``track`` call.

        preds = predict_ensemble_images(model, settings, metadata, beam=beam)
        plot_ensemble_images(settings, preds, metadata)

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    beamline_settings : dict[str, Tensor]
        Scan parameters keyed by PV name; each tensor's leading dim is n_samples.
    observations_metadata : dict[str, dict]
        Per-observation metadata; its keys are the PVs predicted.
    beam : ParticleBeam
        Ensemble beam; ``particles`` must have shape
        ``(n_draws, n_particles, 7)``.
    beamline_constants : dict[str, Tensor] | None
        Fixed parameters folded into the settings before tracking. Keys must not
        overlap ``beamline_settings``.
    normalize : bool, default=True
        Normalize each image to unit pixel sum.
    chunk_size : int | None, default=None
        Track the draw dim in slices of at most this many draws, capping peak
        memory at the ``chunk_size x n_samples`` outer product. Chunking does not
        change the output, since each image is normalized independently.

    Returns
    -------
    dict[str, Tensor]
        Predicted images keyed by observation PV, shape
        ``(n_draws, n_samples, W, H)``.
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

    A spec is ``{source_name: source}``, where each ``source`` holds
    ``predict_images``'s per-source arguments:

        {
            "beamline_settings": {pv: Tensor},        # required
            "observations_metadata": {pv: dict},      # required
            "beamline_constants": {pv: Tensor},       # optional, defaults to {}
        }

    A source carries no measured images, so predictions can be made at arbitrary
    settings the model never trained on.
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

    One beam is sampled once and tracked through every source, so the
    reconstruction is jointly consistent across them. Calling ``predict_images``
    per source with ``beam=None`` would instead sample an independent beam each
    time, and the sources would no longer share a distribution.

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    sources_spec : dict[str, dict]
        Per-source spec keyed by source name, as built by
        ``GPSRLUMEDataModule.to_sources_spec``.
    beam : ParticleBeam | None
        Beam shared across all sources. If ``None``, one is sampled and shared.
    normalize : bool, default=True
        Normalize each image to unit pixel sum.

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

    Each per-source call inserts the scan-step broadcast axis on its own copy, so
    the raw beam can be shared.

    Parameters
    ----------
    model : LitGPSRLUME
        A (typically trained / checkpoint-restored) model.
    sources_spec : dict[str, dict]
        Per-source spec keyed by source name, as built by
        ``GPSRLUMEDataModule.to_sources_spec``.
    beam : ParticleBeam
        Ensemble beam shared across sources; ``particles`` must have shape
        ``(n_draws, n_particles, 7)``.
    normalize : bool, default=True
        Normalize each image to unit pixel sum.
    chunk_size : int | None, default=None
        Draw-dim chunking, forwarded to ``predict_ensemble_images``.

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
