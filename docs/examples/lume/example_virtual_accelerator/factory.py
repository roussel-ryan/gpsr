import os
import tempfile

import torch
from cheetah import ParticleBeam, Segment
from lume_cheetah import CheetahSimulator, LUMECheetahModel

from example_virtual_accelerator.variables import build_action_variables

BUILDER_IMPORT_PATH = f"{__name__}.build_accelerator"


def _segment_from_lattice_json_text(lattice_json: str, dtype) -> Segment:
    """Build a ``Segment`` from lattice-JSON."""
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
        it starts with ``{`` and as a path otherwise.
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
    lattice,
    energy: float,
    dtype: torch.dtype | None = None,
) -> LUMECheetahModel:
    """Build the example ``LUMECheetahModel``.

    Parameters
    ----------
    lattice : str | Segment
        Lattice-JSON text, a path to a lattice JSON, or an already-built
        ``Segment`` (used as-is, not copied). A string is read as JSON text when
        it starts with ``{`` and as a path otherwise.
    energy : float
        Reference energy in eV. Sets the reference beam's energy, which
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
