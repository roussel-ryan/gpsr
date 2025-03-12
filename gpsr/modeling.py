from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from cheetah.accelerator import (
    Quadrupole,
    Drift,
    TransverseDeflectingCavity,
    Dipole,
    Cavity,
    Segment,
    Screen,
)
from cheetah.particles import Beam
from gpsr.beams import BeamGenerator
from gpsr.diagnostics import ImageDiagnostic


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
    def __init__(self, l_quad: float, l_drift: float, diagnostic: Screen):
        super().__init__()
        q1 = Quadrupole(torch.tensor(l_quad), torch.tensor(0.0))
        d1 = Drift(torch.tensor(l_drift))
        self.lattice = Segment([q1, d1])
        self.diagnostic = diagnostic

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        final_beam = self.lattice(beam)
        observations = self.diagnostic(final_beam)

        return tuple(observations.unsqueeze(0))

    def set_lattice_parameters(self, x: torch.Tensor):
        self.lattice.elements[0].k1.data = x


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
        )

        d3 = Drift(name="DIPOLE_TO_SCREEN", length=torch.tensor(l_d3).float())

        lattice = Segment([*upstream_elements, q, d1, tdc, d2, bend, d3])

        self.l_bend = l_bend
        self.l3 = l3
        self.screen_1_diagonstic = screen_1
        self.screen_2_diagonstic = screen_2
        self.lattice = lattice

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        # track the beam through the accelerator in a batched way
        final_beam = self.lattice(beam)

        # check to make sure the beam has the correct batch dimension
        # if not its likely because set_lattice_parameters has not been called yet
        particle_shape = final_beam.particles.shape
        if not particle_shape[0] == 2:
            raise RuntimeError(
                "particle tracking did not return the correct "
                "particle batch shape, did you call "
                "set_lattice_parameters yet. Found particle shape "
                f"{particle_shape}"
            )

        # observe the beam at the different diagnostics based on the first batch
        # dimension
        screen_1_observation = self.screen_1_diagonstic(final_beam[0])
        screen_2_observation = self.screen_2_diagonstic(final_beam[1])

        return screen_1_observation, screen_2_observation

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        """
        sets the quadrupole / TDC / dipole parameters

        Parameters:
        -----------
        x : Tensor
            Specifies the scan parameters in a batched manner with the
            following shape: (2 x N x K x 3) where 2 is the number of dipole states,
            N is the number of TDC states and K is the number of quadrupole
            strengths. The elements of the final dimension correspond to the
            dipole angles, TDC voltages, and quadrupole strengths respectively

        """
        if not (x.shape[0] == 2 and x.shape[-1] == 3):
            raise ValueError(f"incorrect input shape, got {x.shape}")

        # set quad/TDC parameters
        self.lattice.SCAN_QUAD.k1.data = x[..., 2]
        self.lattice.SCAN_TDC.voltage.data = x[..., 1]

        # set dipole parameters
        G = x[..., 0]
        bend_angle = torch.arcsin(self.l_bend * G)
        arc_length = bend_angle / G
        self.lattice.SCAN_DIPOLE.angle.data = bend_angle
        self.lattice.SCAN_DIPOLE.length.data = arc_length
        self.lattice.SCAN_DIPOLE.dipole_e2.data = bend_angle

        # set parameters of drift between dipole and screen
        self.lattice.DIPOLE_TO_SCREEN.length.data = (
            self.l3 - self.l_bend / 2 / torch.cos(bend_angle)
        )

class LinacPhaseQuadDipoleLattice(GPSRLattice):
    def __init__(
        self,
        l_linac: float,
        v_linac: float,
        f_linac: float,
        l_l_q: float,
        l_quad: float,
        k1_q1: float,
        l_q1_q2: float,
        k1_q2: float,
        l_q2_q3: float,
        k1_q3: float,
        l_q3_q4: float,
        l_q4_s1: float,
        l_s1_b: float,
        l_bend: float,
        theta_bend: float,
        l_b_s2: float,
        resolution_s1: tuple[int, int],
        px_size_s1: float,
        resolution_s2: tuple[int, int],
        px_size_s2: float
    ):
        super().__init__()

        linac = Cavity(
            length=torch.tensor(l_linac),
            voltage=torch.tensor(v_linac),
            frequency=torch.tensor(f_linac),
            phase=torch.tensor(90.0),
            name="SCAN_LINAC",
        )

        d_l_q = Drift(torch.tensor(l_l_q - l_linac / 2 - l_quad / 2))

        q1 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q1),
            num_steps=5,
            tracking_method="bmadx",
            name="QUAD1",
        )

        d_q1_q2 = Drift(torch.tensor(l_q1_q2 - l_quad))

        q2 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q2),
            num_steps=5,
            tracking_method="bmadx",
            name="QUAD2",
        )

        d_q2_q3 = Drift(torch.tensor(l_q2_q3 - l_quad))

        q3 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q3),
            num_steps=5,
            tracking_method="bmadx",
            name="QUAD3",
        )

        d_q3_q4 = Drift(torch.tensor(l_q3_q4 - l_quad))

        q4 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q2),
            num_steps=5,
            tracking_method="bmadx",
            name="SCAN_QUAD",
        )

        d_q4_s1 = Drift(torch.tensor(l_q4_s1 - l_quad / 2))

        s1 = Screen(
            resolution=resolution_s1,
            pixel_size=torch.tensor([px_size_s1, px_size_s1]),
            method="kde",
            kde_bandwidth=torch.tensor(px_size_s1/2),
            is_active=True,
            name="SCREEN1",
        )

        d_s1_b = Drift(torch.tensor(l_s1_b - l_bend / 2))

        l_arc = l_bend * theta_bend / np.sin(theta_bend)

        bend = Dipole(
            name="SCAN_DIPOLE",
            length=torch.tensor(l_arc).float(),
            angle=torch.tensor(theta_bend).float(),
            dipole_e1=torch.tensor(0.0).float(),
            dipole_e2=torch.tensor(theta_bend).float(),
        )

        d_b_s2 = Drift(torch.tensor(l_b_s2))

        s2 = Screen(
            resolution=resolution_s2,
            pixel_size=torch.tensor([px_size_s2, px_size_s2]),
            method="kde",
            kde_bandwidth=torch.tensor(px_size_s2/2),
            is_active=True,
            name="SCREEN2",
        )

        segment = Segment(
            [
            linac,
            d_l_q,
            q1,
            d_q1_q2,
            q2,
            d_q2_q3,
            q3,
            d_q3_q4,
            q4,
            d_q4_s1,
            s1,
            d_s1_b,
            bend,
            d_b_s2,
            s2
            ]
        )

        self.screen_1 = s1
        self.screen_2 = s2
        self.segment = segment

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        # track the beam through the accelerator in a batched way
        final_beam = self.segment.track(beam)
        #print(final_beam.energy.shape)
        #print(final_beam.energy)
        # # check to make sure the beam has the correct batch dimension
        # # if not its likely because set_lattice_parameters has not been called yet
        # particle_shape = final_beam.particles.shape
        # if not particle_shape[0] == 2:
        #     raise RuntimeError(
        #         "particle tracking did not return the correct "
        #         "particle batch shape, did you call "
        #         "set_lattice_parameters yet. Found particle shape "
        #         f"{particle_shape}"
        #     )

        # # observe the beam at the different diagnostics based on the first batch
        # # dimension
        screen_1_observation = self.segment.SCREEN1.reading
        screen_2_observation = self.segment.SCREEN2.reading

        return screen_1_observation, screen_2_observation

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        """
        sets the quadrupole / linac parameters

        Parameters:
        -----------
        x : Tensor
            Specifies the scan parameters in a batched manner with the
            following shape: (N x M x 2) where N is the number of linac phases
            and M is the number of quadrupole strengths. The elements of the 
            final dimension correspond to the linac phases and quadrupole 
            strengths respectively.
        """
        # set quad/linac parameters
        self.segment.SCAN_LINAC.phase.data = x[..., 0]
        self.segment.SCAN_QUAD.k1.data = x[..., 1]

class LinacVoltageQuadDipoleLattice(GPSRLattice):
    def __init__(
        self,
        l_linac: float,
        p_linac: float,
        f_linac: float,
        l_l_q: float,
        l_quad: float,
        k1_q1: float,
        l_q1_q2: float,
        k1_q2: float,
        l_q2_q3: float,
        k1_q3: float,
        l_q3_q4: float,
        l_q4_s1: float,
        l_s1_b: float,
        l_bend: float,
        theta_bend: float,
        l_b_s2: float,
        resolution_s1: tuple[int, int],
        px_size_s1: float,
        resolution_s2: tuple[int, int],
        px_size_s2: float
    ):
        super().__init__()

        linac = Cavity(
            length=torch.tensor(l_linac),
            voltage=torch.tensor(0.0),
            frequency=torch.tensor(f_linac),
            phase=torch.tensor(p_linac),
            name="SCAN_LINAC",
        )

        d_l_q = Drift(torch.tensor(l_l_q - l_linac / 2 - l_quad / 2))

        q1 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q1),
            num_steps=5,
            tracking_method="bmadx",
            name="QUAD1",
        )

        d_q1_q2 = Drift(torch.tensor(l_q1_q2 - l_quad))

        q2 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q2),
            num_steps=5,
            tracking_method="bmadx",
            name="QUAD2",
        )

        d_q2_q3 = Drift(torch.tensor(l_q2_q3 - l_quad))

        q3 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q3),
            num_steps=5,
            tracking_method="bmadx",
            name="QUAD3",
        )

        d_q3_q4 = Drift(torch.tensor(l_q3_q4 - l_quad))

        q4 = Quadrupole(
            length=torch.tensor(l_quad),
            k1=torch.tensor(k1_q2),
            num_steps=5,
            tracking_method="bmadx",
            name="SCAN_QUAD",
        )

        d_q4_s1 = Drift(torch.tensor(l_q4_s1 - l_quad / 2))

        s1 = Screen(
            resolution=resolution_s1,
            pixel_size=torch.tensor(px_size_s1),
            method="kde",
            kde_bandwidth=torch.tensor(px_size_s1/2),
            is_active=True,
            name="SCREEN1",
        )

        d_s1_b = Drift(torch.tensor(l_s1_b - l_bend / 2))

        l_arc = l_bend * theta_bend / np.sin(theta_bend)

        bend = Dipole(
            name="SCAN_DIPOLE",
            length=torch.tensor(l_arc).float(),
            angle=torch.tensor(theta_bend).float(),
            dipole_e1=torch.tensor(0.0).float(),
            dipole_e2=torch.tensor(theta_bend).float(),
        )

        d_b_s2 = Drift(torch.tensor(l_b_s2))

        s2 = Screen(
            resolution=resolution_s2,
            pixel_size=torch.tensor(px_size_s2),
            method="kde",
            kde_bandwidth=torch.tensor(px_size_s2/2),
            is_active=True,
            name="SCREEN2",
        )

        segment = Segment(
            [
            linac,
            d_l_q,
            q1,
            d_q1_q2,
            q2,
            d_q2_q3,
            q3,
            d_q3_q4,
            q4,
            d_q4_s1,
            s1,
            d_s1_b,
            bend,
            d_b_s2,
            s2
            ]
        )

        self.screen_1 = s1
        self.screen_2 = s2
        self.segment = segment

    def track_and_observe(self, beam) -> Tuple[Tensor, ...]:
        # track the beam through the accelerator in a batched way
        final_beam = self.segment.track(beam)

        # # check to make sure the beam has the correct batch dimension
        # # if not its likely because set_lattice_parameters has not been called yet
        # particle_shape = final_beam.particles.shape
        # if not particle_shape[0] == 2:
        #     raise RuntimeError(
        #         "particle tracking did not return the correct "
        #         "particle batch shape, did you call "
        #         "set_lattice_parameters yet. Found particle shape "
        #         f"{particle_shape}"
        #     )

        # # observe the beam at the different diagnostics based on the first batch
        # # dimension
        screen_1_observation = self.segment.SCREEN1.reading
        screen_2_observation = self.segment.SCREEN2.reading

        return screen_1_observation, screen_2_observation

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        """
        sets the quadrupole / linac parameters

        Parameters:
        -----------
        x : Tensor
            Specifies the scan parameters in a batched manner with the
            following shape: (N x M x 2) where N is the number of linac phases
            and M is the number of quadrupole strengths. The elements of the 
            final dimension correspond to the linac phases and quadrupole 
            strengths respectively.
        """
        # set quad/linac parameters
        self.segment.SCAN_LINAC.voltage.data = x[..., 0]
        self.segment.SCAN_QUAD.k1.data = x[..., 1]