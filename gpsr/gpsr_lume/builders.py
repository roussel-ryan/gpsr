"""Factory functions that build a ``GPSRLUMEModel`` from a single JSON-pure **spec**
rather than from a pickled ``.pt``.

The spec is one dict that fully describes a model as *two addressed constructors*:
a virtual accelerator and a beam generator, each named by a dotted import-path
string plus the kwargs it needs. The accelerator's assembly function is reached
only through ``accelerator["builder"]``, so no facility package is imported here
and none is named by default -- :func:`model_spec_from_files` *requires* the
caller to say which builder to use. The facility choice is therefore made at the
edge (a training script's ``--accelerator-builder`` flag, or a notebook constant),
not inside this package.

What this module is *not* agnostic about is the simulator: the builder has to
return a ``LUMECheetahModel`` wrapping a Cheetah ``Segment``, because
:func:`serialize_gpsr_lume_model` reads the live lattice back out through
``model.lume_cheetah_model.simulator.segment``. Any facility, but Cheetah-backed.

The *same* spec drives every construction path: a fresh build, a checkpoint embed,
and a self-contained reload. ``build_gpsr_lume_model`` and
:func:`serialize_gpsr_lume_model` are inverses, and their composition is
**idempotent** -- the first pass normalizes, every pass after it is a fixed point::

    s1 = serialize_gpsr_lume_model(build_gpsr_lume_model(spec))
    s2 = serialize_gpsr_lume_model(build_gpsr_lume_model(s1))
    assert s1 == s2        # holds
    assert s1 == spec      # does NOT hold in general -- see below

That first pass rewrites two things, both library-side normalization rather than
information loss (the rebuilt segment's element parameters are bit-identical):

- Cheetah's ``to_lattice_json`` emits the *current* schema, so a lattice file written
  by an older version gains keys (``"metadata": {}``), has omitted defaults filled in
  (``"tracking_method"``), and has float64 literals rounded to float32.
- ``BeamGenerator.get_config()`` returns the generator's *complete* config, including
  defaults a partial input spec left out (e.g. ``"output_scale"``).

Idempotence from the first pass is what checkpointing needs: a checkpoint's embedded
spec rebuilds to a model that re-serializes to that same spec.

Spec shape::

    {
      "accelerator": {
        "builder": "virtual_accelerator.cheetah.factory.build_cheetah_model",
        "config": {
          "lattice": "<cheetah lattice JSON string>",
          "name_map": {"QE10525": "QUAD:IN10:525", ...},
        },
      },
      "generator": {
        "cls": "gpsr.beams.NNParticleBeamGenerator",   # import path (or a live class)
        "config": {"energy": 1.0e8, "n_particles": 10000, ...},  # get_config() kwargs
      },
    }

The accelerator-builder contract
--------------------------------
``builder(**config, energy=<float>) -> LUMECheetahModel``. Two rules:

- ``energy`` is *not* in ``config``. The reference momentum p0c [eV/c] is stored
  once, as the generator's ``energy``, and threaded from the built generator into
  the builder (see :func:`build_gpsr_lume_model`). The generator and the
  accelerator must agree on it -- the accelerator converts EPICS magnet settings to
  Cheetah geometric strengths via magnetic rigidity, which depends on energy -- so a
  single source makes that structural rather than a guarded invariant.
- :data:`LATTICE_CONFIG_KEY` is reserved. :func:`serialize_gpsr_lume_model`
  overwrites it with the *live* segment's JSON so element parameters adjusted after
  the build are captured in the checkpoint, and reads the incoming value as the
  reference that tells those apart from the transient per-scan-step settings a
  forward pass leaves behind (see that function's Notes).

Kept separate from ``training.py`` so that the latter stays generator-agnostic (it
never names a concrete ``BeamGenerator`` subclass) and model construction lives in
one place. :func:`model_spec_from_files` is the file-path convenience that reads the
lattice / name-map JSON files into an inline spec; it is the one place a default
concrete generator class is named. It defaults *no* accelerator builder.

Action variables
----------------
A ``LUMECheetahModel`` is built from *action variables*: one object per EPICS PV,
each knowing the Cheetah element (``element_name``) and attribute it reads/writes
plus any unit conversion. Deriving them from a segment + name-map is facility
knowledge, and it lives in the facility package. At E341 the builder is
``virtual_accelerator.cheetah.factory.build_cheetah_model``, whose spec fields
``lattice`` and ``name_map`` *are* this module's accelerator ``config``, so the
config is splatted in unchanged. No facility code is imported here at all -- the
builder is resolved from its dotted path at build time.
"""

import json
import os
import tempfile
from typing import Callable

import torch

import cheetah
from lume_cheetah.model import LUMECheetahModel

from gpsr.beams import BeamGenerator, NNParticleBeamGenerator
from gpsr.gpsr_lume._imports import import_from_path, to_import_path
from gpsr.gpsr_lume.model import GPSRLUMEModel

# Reserved key inside an accelerator spec's `config`: the Cheetah lattice, as a JSON
# string. `serialize_gpsr_lume_model` rewrites it from the live segment so that element
# parameters adjusted after the model was built survive into the checkpoint, and reads
# the incoming value as the reference that distinguishes those from the transient
# per-scan-step settings a forward pass leaves behind.
LATTICE_CONFIG_KEY = "lattice"

# Builder assumed by `_upgraded_spec` for *legacy* specs only -- flat, pre-`accelerator`
# checkpoints that predate the builder being addressable and so cannot name one. This is
# a historical fact about those files, NOT a default for new specs: nothing else in the
# package reads it, and `model_spec_from_files` requires the caller to name a builder.
# It is the package's only facility-specific reference, and it is a *string* -- resolved
# at build time, never imported, so `import gpsr.gpsr_lume` needs no facility package.
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

    The inverse of :func:`_segment_to_lattice_json`, and likewise a temp-file
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

    A forward pass applies a *batch* of beamline settings at once, so
    ``GPSRLUMEModel.forward`` leaves every element parameter it wrote carrying a
    leading batch dimension -- one entry per scan step of the last batch (e.g. a
    scanned quadrupole's ``k1`` becomes 7 values). That is transient per-batch
    state, not a property of the beamline, and serializing it verbatim would put a
    mid-scan snapshot into the checkpoint's lattice. Worse, the rebuilt segment
    would then *keep* those batch dimensions until something overwrote them: fine
    for a PV the new data also scans, silently wrong for one it does not, which
    would broadcast a stale scan axis into the predictions.

    Each such parameter is collapsed back to its intrinsic shape:

    - If every scan step holds the same value (a setting held constant across the
      batch, e.g. the TDC voltage within one source), that value is kept -- exact,
      and genuinely the accelerator's current setting.
    - If the values differ, the parameter was being *scanned* and no single value
      describes it. The reference lattice's value is restored, i.e. what a fresh
      build from the recorded spec would produce.

    Parameters unaffected by batching pass through untouched, so calibration
    applied after the build still survives (see :func:`serialize_gpsr_lume_model`).

    Parameters
    ----------
    segment : cheetah.Segment
        The live segment. Not modified.
    reference_lattice : str | cheetah.Segment | None
        The lattice recorded in the model's ``accelerator_spec``, supplying both
        each parameter's intrinsic shape (Cheetah exposes no other way to tell an
        intrinsic dimension from a batch one) and the fallback value for a scanned
        parameter. Lattice-JSON text, a path to a lattice JSON, or a ``Segment`` --
        the same three forms a builder accepts for :data:`LATTICE_CONFIG_KEY`, since
        this is that key's recorded value. If ``None`` or unreadable, the copy is
        returned as-is.

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
                # guess. Nothing in gpsr.gpsr_lume produces this.
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

    Resolves ``generator["cls"]`` -- which may be a live class *or* a dotted
    import-path string -- to a ``BeamGenerator`` subclass, then rebuilds it via the
    ``from_config`` contract (``cls.from_config(generator["config"])``). Routing both
    the fresh-build and checkpoint-reload paths through ``from_config`` keeps a single
    instantiation mechanism; trained weights are restored separately via ``state_dict``.

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


def build_accelerator(accelerator: dict, energy: float) -> LUMECheetahModel:
    """Construct a ``LUMECheetahModel`` from a spec's ``accelerator`` dict.

    Resolves ``accelerator["builder"]`` -- a dotted import-path string, or a live
    callable -- and calls it with the spec's ``config`` plus ``energy``. This
    indirection is what keeps the package facility-agnostic: the builder that knows
    how to turn a lattice into per-PV action variables is *named*, never imported
    here (see this module's docstring for the contract).

    Parameters
    ----------
    accelerator : dict
        ``{"builder": <callable or import path>, "config": <builder kwargs>}``.
    energy : float
        Reference momentum p0c [eV/c], passed as the builder's ``energy`` kwarg.
        Deliberately not read from ``config`` -- it is stored once, on the generator.

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

    The single model builder: resolves the spec's two addressed constructors. The
    beam generator is built *first* (see :func:`build_generator`) because it owns the
    reference energy, which is then threaded into the accelerator builder (see
    :func:`build_accelerator`) so both halves share one value by construction.
    Inverse of :func:`serialize_gpsr_lume_model` (see this module's docstring on the
    normalization their first round-trip applies).

    Produces an *untrained* model: the lattice (with its serialized element
    parameters) is restored exactly, but the beam generator's trained weights must be
    restored separately via ``load_state_dict``.

    Parameters
    ----------
    spec : dict
        Canonical spec with keys ``"accelerator"`` and ``"generator"`` (see this
        module's docstring). Built from files via :func:`model_spec_from_files`, or
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
    lume_cheetah_model = build_accelerator(
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
    *,
    accelerator_builder: str | Callable,
    generator: dict | None = None,
) -> dict:
    """Read the lattice / name-map JSON files into an inline canonical spec.

    File-path convenience for the common fresh-build case: inlines the lattice and
    name-map files so the returned spec is self-contained and JSON-pure, ready for
    :func:`build_gpsr_lume_model` (or ``LitGPSRLUME.from_spec``).

    Parameters
    ----------
    cheetah_lattice_path : str
        Path to the Cheetah lattice JSON.
    name_map_path : str
        Path to a JSON object mapping Cheetah element names (e.g. ``"QE10525"``) to
        EPICS control names (e.g. ``"QUAD:IN10:525"``).
    energy : float
        Reference momentum p0c [eV/c]. Stored *once*, as the beam generator's
        ``energy`` config (via ``setdefault``); :func:`build_gpsr_lume_model` threads
        it from the built generator into the accelerator builder.
    accelerator_builder : str | Callable
        The accelerator assembly function, as a dotted import path or a live callable
        (stored as a path). **Required, by design** -- this is the one facility-specific
        decision in a spec, so the caller makes it explicitly rather than inheriting a
        default from this package. At E341 that is
        ``"virtual_accelerator.cheetah.factory.build_cheetah_model"``; see the
        ``--accelerator-builder`` flag on ``scripts/{4d,6d}_train.py``.
    generator : dict, optional
        ``{"cls": <class or import path>, "config": <kwargs>}``. Defaults to
        ``NNParticleBeamGenerator`` with an empty config.

    Returns
    -------
    dict
        A canonical spec (see this module's docstring).

    Notes
    -----
    The ``generator`` default makes this the one place a concrete generator class is
    named, which is what keeps ``training.py`` generator-agnostic. There is
    deliberately no matching accelerator default: the package names no facility.
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

    Everything needed to rebuild an architecturally-equivalent model via
    :func:`build_gpsr_lume_model`: the accelerator's recorded build recipe is
    re-emitted with :data:`LATTICE_CONFIG_KEY` refreshed from the live segment, so
    element parameters adjusted after the build are captured. Trained generator
    weights are NOT included -- they live in the checkpoint ``state_dict`` and are
    restored separately. The dict is JSON-pure (both constructors are stored as
    import-path strings, not pickled) so the checkpoint loads with
    ``weights_only=True``.

    Returns a canonical spec (see this module's docstring) -- the inverse of
    :func:`build_gpsr_lume_model`, idempotent from the first round-trip on.

    Notes
    -----
    The refreshed lattice is a snapshot of the *live* segment, which a model that
    has been run is not a pristine copy of the file it was built from. Three kinds
    of difference land in the emitted lattice, and only the first is what "capture
    the calibration" means:

    - **Element parameters as currently set.** Includes float noise from a
      round trip through the control-system units of an action variable (a constant
      quadrupole's ``k1`` can differ in its last digit), which is real and worth
      recording -- it is the state the fit actually saw.
    - **Screen configuration derived from the data**, applied on every forward pass
      by ``GPSRLUMEModel._setup_screen`` from the datamodule's
      ``observations_metadata``: ``resolution``, ``pixel_size``, ``kde_bandwidth``,
      ``is_active``. Recorded, but not authoritative -- the next forward pass
      overwrites it from whatever metadata that datamodule carries. Likewise
      ``method``, which follows train/eval state (``kde`` vs ``histogram``) and so
      reflects nothing but the mode the model happened to be in when saved.
    - **Per-scan-step settings from the last forward pass**, which are *not*
      recorded: they are collapsed first by :func:`_unbatched_segment`, since a
      batch of scan settings is transient state rather than a property of the
      beamline.

    Raises
    ------
    NotImplementedError
        If the beam generator does not support ``get_config``, or if the model
        carries no recorded ``accelerator_spec`` (it was assembled outside
        :func:`build_gpsr_lume_model`). Such a model cannot be made self-contained
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
    # config key gpsr.gpsr_lume interprets; the rest is opaque builder kwargs. The recorded
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

    Specs written before the accelerator became an addressed constructor were flat --
    ``{"energy", "lattice_json", "name_map", "generator"}`` -- with ``energy``
    duplicated at the top level and inside ``generator["config"]``. Checkpoints
    holding one still load: the flat keys map onto
    :data:`_LEGACY_ACCELERATOR_BUILDER` -- the builder such a spec was necessarily
    built with, since it had no way to name one -- and the top-level ``energy`` folds
    into the generator config, which is now its only home.
    """
    if "accelerator" in spec:
        return spec

    if "lattice_json" not in spec:
        raise KeyError(
            "Spec has neither an 'accelerator' entry nor a legacy 'lattice_json'; got "
            f"keys {sorted(spec)}. See gpsr.gpsr_lume.builders for the spec shape."
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
