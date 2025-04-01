from unittest.mock import MagicMock

import pytest
import torch
from cheetah import Segment, ParticleBeam
from cheetah.accelerator import Quadrupole, Drift, Screen, TransverseDeflectingCavity
from cheetah.particles import Beam

from gpsr.modeling import (
    GPSR,
    GPSRLattice,
    GPSRQuadScanLattice,
    GPSR6DLattice,
    GenericGPSRLattice,
)


class TestModeling:
    def test_gpsr_lattice_abstract_methods(self):
        # Test if GPSRLattice is abstract
        with pytest.raises(TypeError):
            GPSRLattice()

    def test_gpsr_quad_scan_lattice_initialization(self):
        l_quad = 0.5
        l_drift = 1.0
        diagnostic = MagicMock(spec=Screen)
        diagnostic.name = "test"

        lattice = GPSRQuadScanLattice(l_quad, l_drift, diagnostic)

        assert isinstance(lattice.lattice, Segment)
        assert lattice.diagnostic is diagnostic
        assert isinstance(lattice.lattice.elements[0], Quadrupole)
        assert isinstance(lattice.lattice.elements[1], Drift)

    def test_gpsr_quad_scan_lattice_set_lattice_parameters(self):
        l_quad = 0.5
        l_drift = 1.0
        diagnostic = MagicMock(spec=Screen)
        diagnostic.name = "test"

        lattice = GPSRQuadScanLattice(l_quad, l_drift, diagnostic)

        x = torch.tensor([0.1]).unsqueeze(0)
        lattice.set_lattice_parameters(x)

        assert torch.isclose(lattice.lattice.elements[0].k1, torch.tensor(0.1))

    def test_gpsr_quad_scan_lattice_track_and_observe(self):
        l_quad = 0.5
        l_drift = 1.0
        diagnostic = MagicMock(spec=Screen)
        diagnostic.transfer_map = lambda x, y: torch.eye(7)
        diagnostic.reading = torch.eye(3)
        diagnostic.name = "test"

        lattice = GPSRQuadScanLattice(l_quad, l_drift, diagnostic)

        beam = ParticleBeam(energy=torch.tensor(1e6), particles=torch.rand((10, 7)))
        observations = lattice.track_and_observe(beam)

        assert isinstance(observations, tuple)
        assert len(observations) == 1
        assert torch.equal(observations[0], torch.eye(3))

    def test_gpsr_6d_lattice_initialization(self):
        l_quad = 0.5
        l_tdc = 0.6
        f_tdc = 1e9
        phi_tdc = 0.0
        l_bend = 0.8
        theta_on = 0.1
        l1, l2, l3 = 1.0, 1.5, 2.0
        screen_1 = MagicMock(spec=Screen)
        screen_2 = MagicMock(spec=Screen)
        screen_1.name = "screen_1"
        screen_2.name = "screen_2"

        lattice = GPSR6DLattice(
            l_quad,
            l_tdc,
            f_tdc,
            phi_tdc,
            l_bend,
            theta_on,
            l1,
            l2,
            l3,
            screen_1,
            screen_2,
        )

        assert isinstance(lattice.lattice, Segment)
        assert lattice.screens[0] is screen_1
        assert lattice.screens[1] is screen_2

    def test_gpsr_6d_lattice_set_lattice_parameters(self):
        l_quad = 0.5
        l_tdc = 0.6
        f_tdc = 1e9
        phi_tdc = 0.0
        l_bend = 0.8
        theta_on = 0.1
        l1, l2, l3 = 1.0, 1.5, 2.0
        screen_1 = MagicMock(spec=Screen)
        screen_2 = MagicMock(spec=Screen)

        lattice = GPSR6DLattice(
            l_quad,
            l_tdc,
            f_tdc,
            phi_tdc,
            l_bend,
            theta_on,
            l1,
            l2,
            l3,
            screen_1,
            screen_2,
        )

        x = torch.tensor([[[[0.2, 0.5, 0.1], [0.2, 0.5, 0.1]]]]).reshape(2, 1, 1, 3)
        lattice.set_lattice_parameters(x)

        assert torch.equal(
            lattice.lattice.SCAN_QUAD.k1, torch.tensor([0.2, 0.2]).reshape(2, 1, 1)
        )
        assert torch.equal(
            lattice.lattice.SCAN_TDC.voltage, torch.tensor([0.5, 0.5]).reshape(2, 1, 1)
        )
        assert torch.allclose(
            lattice.lattice.SCAN_DIPOLE.angle,
            torch.tensor([0.0801, 0.0801]).reshape(2, 1, 1),
            atol=1e-2,
        )
        assert torch.allclose(
            lattice.lattice.SCAN_DIPOLE.length,
            torch.tensor([0.8035, 0.8035]).reshape(2, 1, 1),
            atol=1e-2,
        )
        assert torch.allclose(
            lattice.lattice.SCAN_DIPOLE.dipole_e2,
            torch.tensor([0.0801, 0.0801]).reshape(2, 1, 1),
            atol=1e-2,
        )

    def test_gpsr_6d_lattice_track_and_observe(self):
        l_quad = 0.5
        l_tdc = 0.6
        f_tdc = 1e9
        phi_tdc = 0.0
        l_bend = 0.8
        theta_on = 0.1
        l1, l2, l3 = 1.0, 1.5, 2.0
        screen_1 = MagicMock(spec=Screen)
        screen_2 = MagicMock(spec=Screen)
        screen_1.reading = torch.eye(2)
        screen_2.reading = torch.eye(2)
        screen_1.name = "screen_1"
        screen_2.name = "screen_2"
        screen_1.transfer_map = lambda x, y: torch.eye(7)
        screen_2.transfer_map = lambda x, y: torch.eye(7)

        lattice = GPSR6DLattice(
            l_quad,
            l_tdc,
            f_tdc,
            phi_tdc,
            l_bend,
            theta_on,
            l1,
            l2,
            l3,
            screen_1,
            screen_2,
        )

        beam = ParticleBeam(energy=torch.tensor(1e6), particles=torch.rand((10, 7)))
        with pytest.raises(ValueError):
            lattice.track_and_observe(beam)

        lattice.set_lattice_parameters(torch.rand(10, 2, 3))
        observations = lattice.track_and_observe(beam)

        assert isinstance(observations, tuple)
        assert len(observations) == 2
        assert torch.equal(observations[0], torch.eye(2))
        assert torch.equal(observations[1], torch.eye(2))

    def test_gpsr_forward(self):
        beam_generator = MagicMock()
        beam_generator.return_value = MagicMock(spec=Beam)

        lattice = MagicMock(spec=GPSRLattice)
        lattice.track_and_observe.return_value = (torch.tensor([1.0]),)

        gpsr = GPSR(beam_generator, lattice)

        x = torch.tensor([0.5])
        results = gpsr.forward(x)

        assert isinstance(results, tuple)
        assert len(results) == 1
        assert torch.equal(results[0], torch.tensor([1.0]))

    def test_generic_gpsr_lattice_initialization(self):
        TDC = TransverseDeflectingCavity(
            length=torch.tensor(1.0),
            voltage=torch.tensor(100.0),
            phase=torch.tensor(0.0),
            name="TDC",
        )
        d1 = Drift(length=torch.tensor(1.0))
        d2 = Drift(length=torch.tensor(1.0))
        d3 = Drift(length=torch.tensor(1.0))

        q1 = Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(0.5), name="q1")

        screen_1 = Screen(name="screen_1")
        screen_2 = Screen(name="screen_2")

        seg = Segment([TDC, d1, screen_1, d2, q1, d3, screen_2])

        var_elements = [(seg.q1, "k1"), (seg.TDC, "voltage")]
        obs_elements = [seg.screen_1, seg.screen_2]

        lattice = GenericGPSRLattice(
            seg, variable_elements=var_elements, observable_elements=obs_elements
        )

        assert isinstance(lattice.lattice, Segment)
        assert lattice.variable_elements == var_elements
        assert lattice.observable_elements == obs_elements

    def test_generic_gpsr_lattice_set_lattice_parameters(self):
        TDC = TransverseDeflectingCavity(
            length=torch.tensor(1.0),
            voltage=torch.tensor(100.0),
            phase=torch.tensor(0.0),
            name="TDC",
        )
        d1 = Drift(length=torch.tensor(1.0))
        d2 = Drift(length=torch.tensor(1.0))
        d3 = Drift(length=torch.tensor(1.0))

        q1 = Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(0.5), name="q1")

        screen_1 = Screen(name="screen_1")
        screen_2 = Screen(name="screen_2")

        seg = Segment([TDC, d1, screen_1, d2, q1, d3, screen_2])

        var_elements = [(seg.q1, "k1"), (seg.TDC, "voltage")]
        obs_elements = [seg.screen_1, seg.screen_2]

        lattice = GenericGPSRLattice(
            seg, variable_elements=var_elements, observable_elements=obs_elements
        )
        x = torch.tensor([[0.1, 0.5], [0.1, 0.5]])
        lattice.set_lattice_parameters(x)

        assert torch.equal(lattice.lattice.q1.k1, torch.tensor([0.1, 0.1]))
        assert torch.equal(lattice.lattice.TDC.voltage, torch.tensor([0.5, 0.5]))

    def test_generic_gpsr_lattice_track_and_observe(self):
        TDC = TransverseDeflectingCavity(
            length=torch.tensor(1.0),
            voltage=torch.tensor(100.0),
            phase=torch.tensor(0.0),
            name="TDC",
        )
        d1 = Drift(length=torch.tensor(1.0))
        d2 = Drift(length=torch.tensor(1.0))
        d3 = Drift(length=torch.tensor(1.0))

        q1 = Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(0.5), name="q1")

        screen_1 = Screen(name="screen_1")
        screen_2 = Screen(name="screen_2")

        seg = Segment([TDC, d1, screen_1, d2, q1, d3, screen_2])

        var_elements = [(seg.q1, "k1"), (seg.TDC, "voltage")]
        obs_elements = [seg.screen_1, seg.screen_2]

        lattice = GenericGPSRLattice(
            seg, variable_elements=var_elements, observable_elements=obs_elements
        )

        beam = ParticleBeam(energy=torch.tensor(1e6), particles=torch.rand((10, 7)))

        x = torch.tensor([[0.1, 0.5], [0.1, 0.5]])
        lattice.set_lattice_parameters(x)
        observations = lattice.track_and_observe(beam)

        assert isinstance(observations, tuple)
        assert len(observations) == 2
        assert observations[0].shape == seg.screen_1.reading.shape
        assert observations[1].shape == seg.screen_2.reading.shape
