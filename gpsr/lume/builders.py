"""Build a ``GPSRLUMEModel`` from a JSON-pure **spec**: two constructors, each
named by a dotted import path plus its kwargs.

    {
      "accelerator": {
        "builder": "virtual_accelerator.cheetah.factory.build_cheetah_model",
        "config": {"lattice": "<lattice JSON>", "name_map": {...}},
      },
      "generator": {
        "cls": "gpsr.beams.NNParticleBeamGenerator",
        "config": {"energy": 1.0e8, "n_particles": 10000, ...},
      },
    }

The builder is resolved from its path at build time, so no facility package is
imported here. It must return a ``LUMECheetahModel``, take ``energy`` as a kwarg
rather than in ``config``, and treat ``LATTICE_CONFIG_KEY`` as reserved.

``build_gpsr_lume_model`` and ``serialize_gpsr_lume_model`` are inverses, and
idempotent from the first round-trip -- which is what lets a checkpoint's embedded
spec rebuild and re-serialize identically. See AGENTS.md.
"""

import json
import os
import tempfile
from typing import Callable

import torch

import cheetah
from lume_cheetah.model import LUMECheetahModel

from gpsr.beams import BeamGenerator, NNParticleBeamGenerator
from gpsr.lume._imports import import_from_path, to_import_path
from gpsr.lume.model import GPSRLUMEModel

# Reserved key inside an accelerator spec's `config`: the Cheetah lattice, as a JSON
# string. `serialize_gpsr_lume_model` rewrites it from the live segment, and reads the
# incoming value as the reference that distinguishes element parameters from the
# transient per-scan-step settings a forward pass leaves behind.
LATTICE_CONFIG_KEY = "lattice"

# Builder assumed by `_upgraded_spec` for *legacy* specs only -- flat,
# pre-`accelerator` checkpoints that could not name one. Not a default for new specs:
# `model_spec_from_files` requires the caller to name a builder.
_LEGACY_ACCELERATOR_BUILDER = "virtual_accelerator.cheetah.factory.build_cheetah_model"


def _segment_to_lattice_json(segment: cheetah.Segment) -> str:
    """Serialize a Cheetah ``Segment`` to a lattice-JSON string.

    Cheetah only exposes file-path (de)serialization for lattices, so this
    round-trips through a temp file and returns the JSON text. Captures the
    segment's *current* element parameters.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        os.close(fd)
        segment.to_lattice_json(tmp_path)
        with open(tmp_path) as f:
            return f.read()
    finally:
        os.remove(tmp_path)


def _lattice_json_to_segment(lattice_json: str) -> cheetah.Segment:
    """Deserialize a lattice-JSON string into a Cheetah ``Segment``.

    The inverse of ``_segment_to_lattice_json``, and likewise a temp-file
    round-trip because Cheetah's lattice (de)serialization is path-based.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        os.close(fd)
        with open(tmp_path, "w") as f:
            f.write(lattice_json)
        return cheetah.Segment.from_lattice_json(tmp_path)
    finally:
        os.remove(tmp_path)


def _unbatched_segment(
    segment: cheetah.Segment, reference_lattice: str | cheetah.Segment | None
) -> cheetah.Segment:
    """Return a copy of ``segment`` with per-scan-step element parameters collapsed.

    ``forward`` applies a batch of settings at once, leaving every parameter it
    wrote with a leading batch dim (a scanned quad's ``k1`` becomes 7 values). A
    rebuilt segment would keep those dims until overwritten, broadcasting a stale
    scan axis into the predictions of any PV the new data does not also scan.

    Batched parameters are collapsed: to their common value if every scan step
    agrees, else to the reference lattice's value. Unbatched parameters pass
    through, so calibration applied after the build survives.

    Parameters
    ----------
    segment : cheetah.Segment
        The live segment. Not modified.
    reference_lattice : str | cheetah.Segment | None
        The lattice recorded in the model's ``accelerator_spec``, supplying each
        parameter's intrinsic shape (Cheetah exposes no other way to tell an
        intrinsic dim from a batch one) and the fallback value for a scanned
        parameter. Lattice-JSON text, a path, or a ``Segment``. If ``None`` or
        unreadable, the copy is returned as-is.

    Returns
    -------
    cheetah.Segment
        A copy, with batch dimensions collapsed where possible.
    """
    unbatched = segment.clone()
    if reference_lattice is None:
        return unbatched

    try:
        if isinstance(reference_lattice, cheetah.Segment):
            reference = reference_lattice
        elif reference_lattice.lstrip().startswith("{"):
            reference = _lattice_json_to_segment(reference_lattice)
        else:
            # A path, which a builder may accept in place of inline JSON (a spec
            # recovered from a checkpoint always holds the text, but a hand-written
            # one often names the file).
            reference = cheetah.Segment.from_lattice_json(reference_lattice)
    except Exception:
        # Deliberately broad, and deliberately silent: this is a cleanup pass over
        # an already-valid lattice, while the caller runs inside
        # `LitGPSRLUME.on_save_checkpoint`, which catches only NotImplementedError.
        # Failing to tidy the snapshot must not abort a training run's checkpoint.
        return unbatched

    reference_elements = {element.name: element for element in reference.elements}
    for element in unbatched.elements:
        reference_element = reference_elements.get(element.name)
        if reference_element is None:
            continue
        for feature in element.defining_tensors:
            value = getattr(element, feature)
            reference_value = getattr(reference_element, feature, None)
            if not isinstance(reference_value, torch.Tensor):
                continue
            if value.ndim <= reference_value.ndim:
                continue
            if (
                value.shape[value.ndim - reference_value.ndim :]
                != reference_value.shape
            ):
                # Not the reference's shape with scan-step dims prepended, so the two
                # are not the same quantity batched -- leave it alone rather than
                # guess. Nothing in gpsr.lume produces this.
                continue
            # Leading dims beyond the reference's are scan-step dims; flatten them
            # so a uniform batch can be recognized whatever its nesting.
            per_step = value.reshape(-1, *reference_value.shape)
            collapsed = (
                per_step[0] if bool(per_step.eq(per_step[0]).all()) else reference_value
            )
            setattr(element, feature, collapsed.clone())

    return unbatched


def build_generator(generator: dict) -> BeamGenerator:
    """Construct a fresh (untrained) ``BeamGenerator`` from a spec's ``generator`` dict.

    ``generator["cls"]`` may be a live class or a dotted import path; either is
    rebuilt via ``cls.from_config(generator["config"])``. Trained weights are
    restored separately via ``state_dict``.

    Parameters
    ----------
    generator : dict
        ``{"cls": <class or import path>, "config": <get_config kwargs>}``.

    Returns
    -------
    BeamGenerator
        Untrained generator of the named class/config.
    """
    generator_cls: type[BeamGenerator] = import_from_path(
        to_import_path(generator["cls"])
    )
    return generator_cls.from_config(generator.get("config") or {})


def build_accelerator_from_spec(accelerator: dict, energy: float) -> LUMECheetahModel:
    """Resolve a spec's ``accelerator["builder"]`` and delegate to it.

    The real builder lives at the facility (e.g. the example's
    ``example_virtual_accelerator.factory.build_accelerator``); this only reads the
    spec and resolves the address.

    Parameters
    ----------
    accelerator : dict
        ``{"builder": <callable or import path>, "config": <builder kwargs>}``.
    energy : float
        Reference momentum p0c [eV/c], passed as the builder's ``energy`` kwarg.
        Not read from ``config`` -- it is stored once, on the generator.

    Returns
    -------
    LUMECheetahModel
        The frozen virtual accelerator.
    """
    builder: Callable[..., LUMECheetahModel] = import_from_path(
        to_import_path(accelerator["builder"])
    )
    return builder(**(accelerator.get("config") or {}), energy=energy)


def build_gpsr_lume_model(spec: dict) -> GPSRLUMEModel:
    """Construct a ``GPSRLUMEModel`` from a canonical spec dict.

    The generator is built first, since it owns the reference energy that is then
    threaded into the accelerator builder so both halves share one value.

    Produces an *untrained* model: the lattice is restored exactly, but the beam
    generator's trained weights must be restored separately via
    ``load_state_dict``.

    Parameters
    ----------
    spec : dict
        Canonical spec with keys ``"accelerator"`` and ``"generator"`` (see this
        module's docstring). Built from files via ``model_spec_from_files``, or
        recovered from a checkpoint. Pre-``accelerator`` specs (flat ``energy`` /
        ``lattice_json`` / ``name_map``) are upgraded on the fly.

    Returns
    -------
    GPSRLUMEModel
        Freshly constructed model (untrained generator weights).
    """
    spec = _upgraded_spec(spec)
    accelerator = spec["accelerator"]

    beam_generator = build_generator(spec["generator"])
    lume_cheetah_model = build_accelerator_from_spec(
        accelerator, energy=float(beam_generator.energy)
    )

    return GPSRLUMEModel(
        lume_cheetah_model=lume_cheetah_model,
        beam_generator=beam_generator,
        # Record how the accelerator was built, normalized to a JSON-pure form, so
        # `serialize_gpsr_lume_model` can reproduce this spec without inspecting the
        # (facility-specific) model that came out of it.
        accelerator_spec={
            "builder": to_import_path(accelerator["builder"]),
            "config": dict(accelerator.get("config") or {}),
        },
    )


def model_spec_from_files(
    cheetah_lattice_path: str,
    name_map_path: str,
    energy: float,
    accelerator_builder: str | Callable,
    generator: dict | None = None,
) -> dict:
    """Read the lattice / name-map JSON files into an inline canonical spec.

    Inlines both files, so the returned spec is self-contained and JSON-pure.

    Parameters
    ----------
    cheetah_lattice_path : str
        Path to the Cheetah lattice JSON.
    name_map_path : str
        Path to a JSON object mapping Cheetah element names (e.g. ``"QE10525"``) to
        EPICS control names (e.g. ``"QUAD:IN10:525"``).
    energy : float
        Reference momentum p0c [eV/c]. Stored *once*, as the beam generator's
        ``energy`` config (via ``setdefault``); ``build_gpsr_lume_model`` threads
        it from the built generator into the accelerator builder.
    accelerator_builder : str | Callable
        The accelerator assembly function, as a dotted import path or a live
        callable (stored as a path). Required: this is the one facility-specific
        decision in a spec, so the package supplies no default. At E341 it is
        ``"virtual_accelerator.cheetah.factory.build_cheetah_model"``.
    generator : dict, optional
        ``{"cls": <class or import path>, "config": <kwargs>}``. Defaults to
        ``NNParticleBeamGenerator`` with an empty config.

    Returns
    -------
    dict
        A canonical spec (see this module's docstring).
    """
    with open(cheetah_lattice_path) as f:
        lattice_json = f.read()
    with open(name_map_path) as f:
        name_map = json.load(f)

    generator = dict(generator or {"cls": NNParticleBeamGenerator})
    # Store the generator class as an import-path string so the spec is JSON-pure
    # (matches what serialize_gpsr_lume_model produces and what the checkpoint holds).
    generator["cls"] = to_import_path(generator["cls"])
    config = dict(generator.get("config") or {})
    config.setdefault("energy", energy)
    generator["config"] = config

    return {
        "accelerator": {
            "builder": to_import_path(accelerator_builder),
            "config": {LATTICE_CONFIG_KEY: lattice_json, "name_map": name_map},
        },
        "generator": generator,
    }


def serialize_gpsr_lume_model(model: GPSRLUMEModel) -> dict:
    """Capture a ``GPSRLUMEModel``'s construction artifacts as a JSON-able dict.

    The accelerator's recorded build recipe is re-emitted with
    ``LATTICE_CONFIG_KEY`` refreshed from the live segment, so element parameters
    adjusted after the build are captured. Trained generator weights are NOT
    included -- they live in the checkpoint ``state_dict``. The dict is JSON-pure,
    so the checkpoint loads with ``weights_only=True``.

    Notes
    -----
    The refreshed lattice snapshots the *live* segment, which is not a pristine copy
    of the file it was built from. Three kinds of difference land in it, and only
    the first is what "capture the calibration" means:

    - **Element parameters as currently set.** Includes float noise from a round
      trip through an action variable's control-system units (a constant
      quadrupole's ``k1`` can differ in its last digit). This is the state the fit
      actually saw.
    - **Screen configuration derived from the data**, applied on every forward pass
      by ``GPSRLUMEModel._setup_screen``: ``resolution``, ``pixel_size``,
      ``is_active``, ``method``. Recorded but not authoritative -- the next forward
      pass overwrites it from whatever metadata the datamodule carries.
    - **Per-scan-step settings from the last forward pass**, which are *not*
      recorded: ``_unbatched_segment`` collapses them first.

    Raises
    ------
    NotImplementedError
        If the beam generator does not support ``get_config``, or if the model
        carries no recorded ``accelerator_spec`` (it was assembled outside
        ``build_gpsr_lume_model``). Such a model cannot be made self-contained
        and must be rebuilt and re-supplied explicitly.
    """
    accelerator_spec = model.accelerator_spec
    if accelerator_spec is None:
        raise NotImplementedError(
            "The model carries no 'accelerator_spec', so the recipe that built its "
            "virtual accelerator is unknown and cannot be serialized. Build it with "
            "build_gpsr_lume_model, or pass accelerator_spec= to GPSRLUMEModel."
        )

    config = dict(accelerator_spec.get("config") or {})
    # Re-serialize from the live segment rather than reusing the recorded lattice, so
    # that element parameters adjusted after the build are captured. This is the one
    # config key gpsr.lume interprets; the rest is opaque builder kwargs. The recorded
    # lattice is still needed, as the reference that tells a parameter's intrinsic
    # shape from a batch of scan settings left behind by the last forward pass.
    recorded_lattice_json = config.get(LATTICE_CONFIG_KEY)
    config[LATTICE_CONFIG_KEY] = _segment_to_lattice_json(
        _unbatched_segment(
            model.lume_cheetah_model.simulator.segment, recorded_lattice_json
        )
    )

    beam_generator = model.beam_generator
    return {
        "accelerator": {
            "builder": to_import_path(accelerator_spec["builder"]),
            "config": config,
        },
        "generator": {
            "cls": to_import_path(type(beam_generator)),
            "config": beam_generator.get_config(),
        },
    }


def _upgraded_spec(spec: dict) -> dict:
    """Return ``spec`` in canonical form, upgrading a pre-``accelerator`` one.

    Specs predating the addressed accelerator were flat -- ``{"energy",
    "lattice_json", "name_map", "generator"}``. Their keys map onto
    ``_LEGACY_ACCELERATOR_BUILDER`` (the only builder they could have used), and the
    top-level ``energy`` folds into the generator config, now its only home.
    """
    if "accelerator" in spec:
        return spec

    if "lattice_json" not in spec:
        raise KeyError(
            "Spec has neither an 'accelerator' entry nor a legacy 'lattice_json'; got "
            f"keys {sorted(spec)}. See gpsr.lume.builders for the spec shape."
        )

    generator = dict(spec["generator"])
    config = dict(generator.get("config") or {})
    if "energy" not in config:
        if "energy" not in spec:
            raise KeyError(
                "Legacy spec has no top-level 'energy' and generator.config has no "
                "'energy' either; one of the two must supply it."
            )
        config["energy"] = spec["energy"]
    generator["config"] = config

    return {
        "accelerator": {
            "builder": _LEGACY_ACCELERATOR_BUILDER,
            "config": {
                LATTICE_CONFIG_KEY: spec["lattice_json"],
                "name_map": spec["name_map"],
            },
        },
        "generator": generator,
    }
