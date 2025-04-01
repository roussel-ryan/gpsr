from cheetah.particles import ParticleBeam
from gpsr.visualization import compare_beams
import torch


class TestVisualizationUtils:
    def test_compare_beams(self):
        beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            alpha_x=torch.tensor(1.0),
            alpha_y=torch.tensor(1.0),
            num_particles=100,
        )

        compare_beams(
            beam_1=beam,
            beam_2=beam,
            dimensions=("x", "px", "y", "py"),
            bins=50,
        )
