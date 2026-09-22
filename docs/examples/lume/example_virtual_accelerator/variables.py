from typing import Union

import torch
from cheetah import Segment
from lume_cheetah.actions import (
    CheetahReadOnlyNDVariable,
    CheetahWritableScalarVariable,
)

# Cheetah element name to PV-name prefix conversion
QUADRUPOLE_NAMES = {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"}
CAVITY_NAMES = {"tdc": "TDC"}
SCREEN_NAMES = {"s1": "S1", "s2": "S2"}

# Full-scale value of a screen image PV
IMAGE_FULL_SCALE = 65535


def get_magnetic_rigidity(energy):
    """Magnetic rigidity ($B\\rho$) in kG-m.

    Parameters
    ----------
    energy : float | torch.Tensor
        Reference energy in eV.

    Returns
    -------
    float | torch.Tensor
        Rigidity in kG-m.
    """
    return 33.356 * energy / 1e9


class QuadrupoleBCTRLVariable(CheetahWritableScalarVariable):
    """Quadrupole control/desired integrated field strength (BCTRL/BDES) in kG."""

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
    """Writable cavity amplitude request variable in MV."""

    unit: str = "MV"
    element_attribute: str = "voltage"

    def _get(self, simulator):
        # Cheetah stores the physical voltage in V; this PV is in MV.
        return self._get_direct_attribute(simulator, self.element_attribute) / 1e6

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, self.element_attribute, value * 1e6)


class CavityPREQVariable(CheetahWritableScalarVariable):
    """Writable cavity phase request variable in degrees."""

    unit: str = "degrees"
    element_attribute: str = "phase"

    def _get(self, simulator):
        # Cheetah stores phase in rad/2pi (turns); this PV is in degrees.
        return self._get_direct_attribute(simulator, self.element_attribute) * 360.0

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, self.element_attribute, value / 360.0)


class ScreenImageVariable(CheetahReadOnlyNDVariable):
    """Read-only screen image array."""

    unit: str = "counts"
    element_attribute: str = "reading"

    dtype: Union[torch.dtype, str] = torch.float32

    def _get(self, simulator):
        # `.mT` transposes only the last two axes: (...,y, x) - > (..., x, y)
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
    The screen ``shape`` here is the lattice's nominal camera geometry. When a
    model built from these variables is used for a reconstruction, the measured
    geometry from the dataset's ``observations_metadata`` replaces it per
    observed screen on every forward pass.
    """
    quadrupoles = [
        QuadrupoleBCTRLVariable(name=f"{control_name}:BCTRL", element_name=element_name)
        for element_name, control_name in QUADRUPOLE_NAMES.items()
    ]

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
