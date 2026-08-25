"""Assembly of the example virtual accelerator: lattice + variables -> model.

This is the **facility seam**. ``gpsr.lume`` never imports a facility package; it
reaches the assembly function through a dotted import-path string in a model spec::

    spec["accelerator"] = {
        "builder": "example_virtual_accelerator.factory.build_accelerator",
        "config": {"lattice": <lattice JSON string or path>},
    }

and calls it as ``builder(**config, energy=<float>)``. That string is what a
checkpoint records, so this module -- not just its contents -- is part of the
reproducibility surface: a checkpoint can only be rebuilt where this path is
importable.

Two rules the contract imposes, both easy to get wrong:

- ``energy`` is **not** in ``config``. The reference momentum lives once, on the
  beam generator, and ``gpsr.lume.builders.build_gpsr_lume_model`` threads it into
  this function. The accelerator needs it because converting a magnet's control
  value to a Cheetah geometric strength goes through magnetic rigidity.
- ``config["lattice"]`` is **reserved**. ``serialize_gpsr_lume_model`` overwrites
  it with the *live* segment's JSON so that calibration applied after the build is
  captured in the checkpoint. So on reload, ``lattice`` arrives as JSON *text*,
  not the path it started as -- which is why :func:`build_accelerator` accepts
  both.

The four steps below are exactly what ``virtual_accelerator.ipynb`` does cell by
cell; this function is only those steps behind one name, so that they have an
address a spec can point at.
"""

import os
import tempfile

import torch
from cheetah import ParticleBeam, Segment
from lume_cheetah import CheetahSimulator, LUMECheetahModel

from example_virtual_accelerator.variables import build_action_variables

#: Dotted import path of :func:`build_accelerator` -- what a spec's
#: ``accelerator["builder"]`` holds and what a checkpoint records. Derived from
#: ``__name__`` so it cannot drift from the module's actual location.
BUILDER_IMPORT_PATH = f"{__name__}.build_accelerator"


def _segment_from_lattice_json_text(lattice_json: str, dtype) -> Segment:
    """Build a ``Segment`` from lattice-JSON *text*.

    Cheetah's public lattice loader takes a path, so the text round-trips through
    a temporary file rather than reaching into Cheetah's internal dict parser.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(lattice_json)
        return Segment.from_lattice_json(tmp_path, dtype=dtype)
    finally:
        os.remove(tmp_path)


def _resolve_segment(lattice, dtype) -> Segment:
    """Load ``lattice`` into a Cheetah ``Segment``.

    Parameters
    ----------
    lattice : str | Segment
        Lattice-JSON text, a path to a lattice JSON, or an already-built
        ``Segment`` (used as-is, not copied). A string is read as JSON text when
        it starts with ``{`` and as a path otherwise -- the same test the
        SLAC factory uses, and what lets one config key serve both the
        file-path build and the checkpoint reload.
    dtype : torch.dtype | None
        Element data type. ``None`` leaves Cheetah's default.

    Returns
    -------
    Segment
    """
    if isinstance(lattice, Segment):
        return lattice
    if not isinstance(lattice, str):
        raise TypeError(
            f"'lattice' must be a lattice-JSON string, a path, or a cheetah "
            f"Segment; got {type(lattice).__name__}."
        )
    if lattice.lstrip().startswith("{"):
        return _segment_from_lattice_json_text(lattice, dtype)
    return Segment.from_lattice_json(lattice, dtype=dtype)


def build_accelerator(
    *,
    lattice,
    energy: float,
    dtype: torch.dtype | None = None,
) -> LUMECheetahModel:
    """Build the example ``LUMECheetahModel``.

    Parameters
    ----------
    lattice : str | Segment
        Lattice-JSON text, a path to a lattice JSON, or a ``Segment``. See
        :func:`_resolve_segment`.
    energy : float
        Reference momentum p0c [eV/c]. Sets the placeholder beam's energy, which
        the simulator freezes into ``simulator.energies`` at construction and
        every magnet variable then uses for its rigidity conversion.
    dtype : torch.dtype, optional
        Element data type. Defaults to Cheetah's own default.

    Returns
    -------
    LUMECheetahModel
        Model wired to the action variables in
        :mod:`example_virtual_accelerator.variables`.

    Notes
    -----
    The incoming beam is a single-particle placeholder rather than a real
    distribution because a reconstruction fits a *trainable* beam through this
    model and replaces the distribution on every forward pass. Only the energy
    survives, and only because the rigidity conversion needs it.
    """
    segment = _resolve_segment(lattice, dtype)

    placeholder_beam = ParticleBeam(
        torch.zeros(1, 7),
        energy=torch.tensor(energy, dtype=dtype or torch.float32),
    )

    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=placeholder_beam,
    )

    return LUMECheetahModel(
        simulator=simulator,
        action_variables=build_action_variables(segment),
    )
