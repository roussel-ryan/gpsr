import torch

from phase_space_reconstruction.beam_models.nn import NNParticleBeamGenerator
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.modeling import GPSR6DLattice


class Test6DLattice:
    def test_6d_lattice(self):
        n = 50
        diagnostic = ImageDiagnostic(
            bins_x=torch.stack((
                torch.linspace(-20, 20, n) * 1e-3,
                torch.linspace(-30, 30, n) * 1e-3,
            )),
            bins_y=torch.stack((
                torch.linspace(-20, 20, n) * 1e-3,
                torch.linspace(-30, 30, n) * 1e-3,
            )),
            bandwidth=torch.tensor((0.1, 0.15)) * 1e-3
        )

        gpsr_lattice = GPSR6DLattice(
            l_quad=0.1,
            l_tdc=0.1,
            f_tdc=1.3e9,
            phi_tdc=0.0,
            l_bend=0.3018,
            theta_on=- 20.0 * 3.14 / 180.0,
            l1=0.790702,
            l2=0.631698,
            l3=0.889,
            p0c=10.0e6,
            diagnostic=diagnostic
        )

        beam = NNParticleBeamGenerator(100, torch.tensor(10.0e6))

        gpsr_lattice.set_lattice_parameters(
            torch.stack((
                torch.tensor((0.0, 0.0, 0.0)),
                torch.tensor((0.0, 0.0, 1.0))
            ))
        )

        obs, final_beam = gpsr_lattice.track_and_observe(beam())

        assert obs.shape == torch.Size([2, 50, 50])
        assert gpsr_lattice.lattice.elements[-1].L.shape == torch.Size([2, 1])
        assert gpsr_lattice.lattice.elements[-1].L[0] != \
               gpsr_lattice.lattice.elements[-1].L[1]

        assert final_beam.x.shape == torch.Size([2, 100])
