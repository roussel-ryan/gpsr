"""Action variables for the example virtual accelerator.

A ``LUMECheetahModel`` is built from *action variables*: one object per control
-system PV, each knowing which Cheetah element and attribute it reads/writes and
the unit conversion between the two. That knowledge is **facility-dependent** --
it encodes the local PV naming convention and the engineering units the control
system speaks -- so it lives here rather than in any library. Everything these
classes inherit from (``lume_cheetah.actions``) is generic; only the names and
the conversion factors below are specific to this example.

Why a module and not a notebook cell
------------------------------------
These classes are defined in a file, and imported by the notebooks, so that they
have an **importable dotted path**. A serialized model records each variable as
``{"variable_class": "<module>.<ClassName>", ...fields}`` -- ``Variable`` is a
pydantic model, so its fields dump to JSON for free -- and rebuilding resolves
that path with ``importlib``. A class defined in a notebook cell dumps as
``__main__.QuadrupoleBCTRLVariable``, which resolves inside the same kernel and
fails everywhere else, so a model built that way cannot be reloaded. The same
reasoning applies to the accelerator *builder* path in a ``gpsr.lume`` spec.

The conversions
---------------
Each class is one unit conversion, and each is a place a real facility can differ:

===========================  ==============  ===================================
Class                        PV unit         Cheetah attribute
===========================  ==============  ===================================
``QuadrupoleBCTRLVariable``  kG              ``k1`` [1/m^2], via rigidity
``CavityAREQVariable``       MV              ``voltage`` [V]
``CavityPREQVariable``       degrees         ``phase`` [rad/2pi, i.e. turns]
``ScreenImageVariable``      counts          ``reading`` [unit total over pixels]
===========================  ==============  ===================================

The quadrupole conversion is why a model needs a reference energy at build time:
``BCTRL`` is an integrated field, and turning it into a geometric strength needs
the magnetic rigidity, which depends on the beam momentum.
"""

from typing import Union

import torch
from cheetah import Segment
from lume_cheetah.actions import (
    CheetahReadOnlyNDVariable,
    CheetahWritableScalarVariable,
)

#: Cheetah element name -> control-name prefix, grouped by element type because
#: the type selects which variable class (and so which conversion) applies. At a
#: real facility this mapping is derived from a device database or read from a
#: checked-in name map; this lattice is small enough to write it out.
QUADRUPOLE_NAMES = {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"}
CAVITY_NAMES = {"tdc": "TDC"}
SCREEN_NAMES = {"s1": "S1", "s2": "S2"}

#: Full-scale value of a screen image PV, i.e. the sum over pixels for a beam
#: entirely on the sensor. Cheetah's ``Screen.reading`` is normalized to unit
#: total, so it sums to the on-screen charge *fraction* regardless of particle
#: count or pixel size; this factor turns that into camera-like counts.
IMAGE_FULL_SCALE = 65535


def get_magnetic_rigidity(energy):
    """Magnetic rigidity ($B\\rho$) in kG-m.

    Parameters
    ----------
    energy : float | torch.Tensor
        Reference momentum p0c [eV/c].

    Returns
    -------
    float | torch.Tensor
        Rigidity [kG-m].
    """
    return 33.356 * energy / 1e9


class QuadrupoleBCTRLVariable(CheetahWritableScalarVariable):
    """Quadrupole control/desired integrated field strength (BCTRL/BDES) in kG.

    The control system sets an integrated field; Cheetah holds a geometric
    focusing strength ``k1`` [1/m^2]. Converting between them needs both the
    element length and the local beam energy, which the simulator supplies
    (``simulator.energies``, frozen at construction from the initial beam).
    """

    unit: str = "kG"
    element_attribute: str = "k1"

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        return (
            getattr(element, self.element_attribute)
            * element.length
            * get_magnetic_rigidity(energy)
        )

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        new_k1 = value / get_magnetic_rigidity(energy) / element.length
        setattr(element, self.element_attribute, new_k1)


class CavityAREQVariable(CheetahWritableScalarVariable):
    """Writable cavity amplitude request in MV."""

    unit: str = "MV"
    element_attribute: str = "voltage"

    def _get(self, simulator):
        # Cheetah stores the physical voltage in V; this PV is in MV.
        return self._get_direct_attribute(simulator, self.element_attribute) / 1e6

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, self.element_attribute, value * 1e6)


class CavityPREQVariable(CheetahWritableScalarVariable):
    """Writable cavity phase request in degrees."""

    unit: str = "degrees"
    element_attribute: str = "phase"

    def _get(self, simulator):
        # Cheetah stores phase in rad/2pi (turns); this PV is in degrees.
        return self._get_direct_attribute(simulator, self.element_attribute) * 360.0

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, self.element_attribute, value / 360.0)


class ScreenImageVariable(CheetahReadOnlyNDVariable):
    """Read-only screen image array, in camera-like counts."""

    unit: str = "counts"
    element_attribute: str = "reading"

    # Re-declared only to work around an upstream bug: NDVariable's dtype
    # coercion is gated on `"dtype" in cls.__annotations__`, i.e. the class's own
    # annotations, so a subclass that does not repeat it keeps the serialized
    # string `"torch.float32"` and then every get() fails with the self-
    # contradictory `Expected dtype torch.float32, got torch.float32`. Must be
    # spelled `Union[...]`; the equivalent-looking `torch.dtype | str` is a
    # `types.UnionType`, which the same gate does not unwrap. For the same
    # reason, do not add `from __future__ import annotations` to this module.
    dtype: Union[torch.dtype, str] = torch.float32

    def _get(self, simulator):
        # `.mT` transposes only the last two axes: `reading` is (..., y, x) and
        # the PV convention is (..., x, y). Never use `.T`, which reverses *all*
        # dims -- GPSR tracks batched settings, so a (n_steps, 1, W, H) reading
        # would come back (H, W, 1, n_steps) and fail shape validation. Unbatched
        # calls pass either way, so the mistake hides until a scan runs.
        return super()._get(simulator).mT * IMAGE_FULL_SCALE


def build_action_variables(segment: Segment) -> list:
    """Build the full PV interface for the example lattice.

    Parameters
    ----------
    segment : Segment
        The Cheetah segment the variables address. Read for the screen
        resolutions, which size each image variable's declared ``shape``.

    Returns
    -------
    list
        Action variables, ready to pass to ``LUMECheetahModel``.

    Notes
    -----
    The screen ``shape`` here is the lattice's *nominal* camera geometry. When a
    model built from these variables is used for a reconstruction, the measured
    geometry from the dataset's ``observations_metadata`` replaces it per
    observed screen on every forward pass.
    """
    quadrupoles = [
        QuadrupoleBCTRLVariable(name=f"{control_name}:BCTRL", element_name=element_name)
        for element_name, control_name in QUADRUPOLE_NAMES.items()
    ]

    # One cavity contributes two PVs: amplitude and phase are set independently.
    cavities = [
        variable_class(name=f"{control_name}:{suffix}", element_name=element_name)
        for element_name, control_name in CAVITY_NAMES.items()
        for suffix, variable_class in (
            ("AREQ", CavityAREQVariable),
            ("PREQ", CavityPREQVariable),
        )
    ]

    screens = [
        ScreenImageVariable(
            name=f"{control_name}:Image:ArrayData",
            element_name=element_name,
            shape=tuple(getattr(segment, element_name).resolution),
        )
        for element_name, control_name in SCREEN_NAMES.items()
    ]

    return quadrupoles + cavities + screens
