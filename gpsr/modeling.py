from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor

from cheetah.accelerator import (
    Quadrupole,
    Drift,
    TransverseDeflectingCavity,
    Dipole,
    Segment,
    Screen,
)
from cheetah.particles import Beam
from cheetah.accelerator import Element
from gpsr.beams import BeamGenerator


class GPSRLattice(torch.nn.Module, ABC):
    @abstractmethod
    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        pass

    @abstractmethod
    def track_and_observe(self, beam: Beam) -> Tuple[Tensor, ...]:
        """
        tracks beam through the lattice and returns observations

        Returns
        -------
        results: Tuple[Tensor]
            Tuple of results from each measurement path
        """
        pass


class GPSR(torch.nn.Module):
    def __init__(self, beam_generator: BeamGenerator, lattice: GPSRLattice):
        super(GPSR, self).__init__()
        self.beam_generator = deepcopy(beam_generator)
        self.lattice = deepcopy(lattice)

    def forward(self, x: Tensor):
        # generate beam
        initial_beam = self.beam_generator()

        # set lattice parameters
        self.lattice.set_lattice_parameters(x)

        return self.lattice.track_and_observe(initial_beam)


class GPSRQuadScanLattice(GPSRLattice):
    def __init__(self, l_quad: float, l_drift: float, screen: Screen):
        super().__init__()
        q1 = Quadrupole(
            torch.tensor(l_quad), torch.tensor(0.0), tracking_method="bmadx"
        )
        d1 = Drift(torch.tensor(l_drift))
        self.segment = Segment([q1, d1, screen])
        self.screen = screen

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        # track the beam through the accelerator in a batched way
        self.segment(beam)
        return tuple(self.segment.elements[-1].reading.transpose(-1, -2).unsqueeze(0))

    def set_lattice_parameters(self, x: torch.Tensor):
        self.segment.elements[0].k1.data = x[:, 0]


class GPSR6DLattice(GPSRLattice):
    def __init__(
        self,
        l_quad: float,
        l_tdc: float,
        f_tdc: float,
        phi_tdc: float,
        l_bend: float,
        theta_on: float,
        l1: float,
        l2: float,
        l3: float,
        screen_1: Screen,
        screen_2: Screen,
        upstream_elements=None,
    ):
        super().__init__()

        upstream_elements = upstream_elements or []

        # Drift from Quad to TDC (0.5975)
        l_d1 = l1 - l_quad / 2 - l_tdc / 2

        # Drift from TDC to Bend (0.3392)
        l_d2 = l2 - l_tdc / 2 - l_bend / 2

        # Drift from Bend to YAG 2 (corrected for dipole on/off)
        l_d3 = l3 - l_bend / 2 / np.cos(theta_on)

        q = Quadrupole(
            torch.tensor(l_quad),
            torch.tensor(0.0),
            name="SCAN_QUAD",
            num_steps=5,
            tracking_method="bmadx",
        )
        d1 = Drift(torch.tensor(l_d1))

        tdc = TransverseDeflectingCavity(
            length=torch.tensor(l_tdc),
            voltage=torch.tensor(0.0),
            frequency=torch.tensor(f_tdc),
            phase=torch.tensor(phi_tdc),
            tilt=torch.tensor(3.14 / 2),
            name="SCAN_TDC",
        )

        d2 = Drift(length=torch.tensor(l_d2))

        # initialize with dipole on
        l_arc = l_bend * theta_on / np.sin(theta_on)

        bend = Dipole(
            name="SCAN_DIPOLE",
            length=torch.tensor(l_arc).float(),
            angle=torch.tensor(0.0).float(),
            dipole_e1=torch.tensor(0.0).float(),
            dipole_e2=torch.tensor(theta_on).float(),
            tracking_method="bmadx",
        )

        d3 = Drift(name="DIPOLE_TO_SCREEN", length=torch.tensor(l_d3).float())

        segment = Segment([*upstream_elements, q, d1, tdc, d2, bend, d3])

        self.l_bend = l_bend
        self.l3 = l3
        self.screen_1 = screen_1
        self.screen_2 = screen_2
        self.segment = segment

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        # track the beam through the accelerator in a batched way
        final_beam = self.segment(beam)

        # note that the axis that speicfies the screen number is the axis after the batch axis
        # e.g. (N x 2 x 3) where N is the batch size, 2 is the number of screens,
        # note this is not the last axis for indexing the sub-beams

        # we require the beam to be at least 2D, so we can use the last axis to index the screens
        if len(final_beam.sigma_x.shape) < 2:
            raise ValueError(
                "Beam must have at least 2 dimensions corresponding to the dipole strengths for each screen"
            )

        n_batch_dims = len(final_beam.sigma_x.shape) - 1
        batch_size = (slice(None),) * n_batch_dims  # Use a tuple instead of a list

        obs = []
        for i, screen in enumerate((self.screen_1, self.screen_2)):
            screen.track(final_beam[batch_size + (i,)])  # Concatenate explicitly
            obs.append(screen.reading)

        return tuple(obs)

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        """
        sets the quadrupole / TDC / dipole parameters

        Parameters:
        -----------
        x : Tensor
            Specifies the scan parameters in a batched manner with the
            following shape: (K x N x 2 x 3) where 2 is the number of dipole states,
            N is the number of TDC states and K is the number of quadrupole
            strengths. The elements of the final dimension correspond to the
            quadrupole strengths, TDC voltages, and dipole angles respectively

        """
        # set quad/TDC parameters
        self.segment.SCAN_QUAD.k1.data = x[..., 0]
        self.segment.SCAN_TDC.voltage.data = x[..., 1]

        # set dipole parameters
        G = x[..., 2]
        bend_angle = torch.arcsin(self.l_bend * G)
        arc_length = bend_angle / G
        self.segment.SCAN_DIPOLE.angle.data = bend_angle
        self.segment.SCAN_DIPOLE.length.data = arc_length
        self.segment.SCAN_DIPOLE.dipole_e2.data = bend_angle

        # set parameters of drift between dipole and screen
        self.segment.DIPOLE_TO_SCREEN.length.data = (
            self.l3 - self.l_bend / 2 / torch.cos(bend_angle)
        )


class GenericGPSRLattice(GPSRLattice):
    """
    Attributes:
        segment: The base cheetah Segment structure used for beam tracking.
        variable_elements: A list of tuples, where each tuple contains an element object and
                          the name of the parameter to be varied as a string.
        observable_elements: A list of elements that have the 'reading' property.
    """

    def __init__(
        self,
        segment,
        variable_elements: List[Tuple[Element, str]],
        observable_elements: List[Element],
    ):
        """
        Initializes the GPSRLattice instance.

        Args:
            segment: The cheetah Segment used for beam tracking.
            variable_elements: A list of tuples, where each tuple contains a cheetah element object and
                              the name of the parameter to be varied as a string.
            observable_elements: A list of elements that have the 'reading' property.
        """
        super().__init__()

        for element in variable_elements:
            if not hasattr(element[0], element[1]):
                raise AttributeError(
                    f"Variable element {element[0].name} does not have parameter '{element[1]}'."
                )

        for element in observable_elements:
            if not hasattr(element, "reading"):
                raise AttributeError(
                    f"Observable element {element.name} does not have a 'reading' property."
                )

        self.segment = segment
        self.variable_elements = variable_elements
        self.observable_elements = observable_elements

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        """
        Tracks a beam through the segment and collects observations from designated elements.

        Args:
            beam: The beam object to be tracked.

        Returns:
            A tuple of tensors representing the observations from the observable elements.
        """

        # Compute the merged transfer maps for the segment
        merged_segment = self.segment.transfer_maps_merged(beam)

        # Apply the merged segment transformations to the beam
        merged_segment(beam)

        # Collect observations from the observable elements
        observations = tuple([element.reading for element in self.observable_elements])

        return observations

    def set_lattice_parameters(self, settings: torch.Tensor):
        """
        Sets the parameters of variable elements in the segment.

        Args:
            settings: A tensor containing the new parameter values for the variable elements.
        """
        for i, element in enumerate(self.variable_elements, 0):
            setattr(element[0], element[1], settings[..., i])


class GPSR5DLattice(GPSRLattice):
    def __init__(
        self,
        l_quad: Tensor,
        l_dipole: Tensor,
        theta_dipole: Tensor,
        l1: Tensor,
        l2: Tensor,
        l3: Tensor,
        l4: Tensor,
        screen_1: Screen,
        screen_2: Screen,
    ):
        super().__init__()

        tensor_kwargs = {"dtype": l_quad.dtype}

        # DQ6
        dq6 = Quadrupole(
            l_quad,
            torch.tensor(0.0, **tensor_kwargs),
            name="DQ6",
            num_steps=5,
            tracking_method="bmadx",
        )

        # Drift from DQ6 to DQ7
        d1 = Drift(l1 - l_quad)

        # DQ7
        dq7 = Quadrupole(
            l_quad,
            torch.tensor(0.0, **tensor_kwargs),
            name="DQ7",
            num_steps=5,
            tracking_method="bmadx",
        )

        # Drift from DQ7 to SCREEN_A
        d2 = Drift(l2 - l_quad / 2)

        # Drift from SCREEN_A to Dipole
        d3 = Drift(l3 - l_dipole / 2)

        # Dipole
        l_arc = l_dipole * theta_dipole / torch.sin(theta_dipole)
        dipole = Dipole(
            name="DIPOLE",
            length=l_arc,
            angle=torch.tensor(0.0, **tensor_kwargs),
            dipole_e1=torch.tensor(0.0, **tensor_kwargs),
            dipole_e2=theta_dipole,
            tracking_method="bmadx",
        )

        # Drift from Dipole to SCREEN_B
        d4 = Drift(l4)

        segment = Segment([dq6, d1, dq7, d2, screen_1, d3, dipole, d4, screen_2])

        self.screen_1 = screen_1
        self.screen_2 = screen_2
        self.segment = segment

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        # track the beam through the accelerator in a batched way
        self.segment(beam)

        # get the readings from the screens
        obs = (self.screen_1.reading[:, 0, :, :], self.screen_2.reading[:, 1, :, :])
        return obs

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        """
        sets the quadrupole parameters

        Parameters:
        -----------
        x : Tensor

        """
        # set quad parameters
        self.segment.DQ6.k1.data = x[..., 0]
        self.segment.DQ7.k1.data = x[..., 1]
