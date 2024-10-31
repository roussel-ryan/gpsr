import torch
from torch import Size
from torch.distributions import MultivariateNormal

from cheetah.particles import ParticleBeam
from cheetah.utils.bmadx import bmad_to_cheetah_coords


class NNTransform(torch.nn.Module):
    def __init__(
            self,
            n_hidden,
            width,
            dropout=0.0,
            activation=torch.nn.Tanh(),
            output_scale=1e-2,
            phase_space_dim=6,
    ):
        """
        Nonparametric transformation - NN
        """
        super(NNTransform, self).__init__()

        layer_sequence = [torch.nn.Linear(phase_space_dim, width), activation]

        for i in range(n_hidden):
            layer_sequence.append(torch.nn.Linear(width, width))
            layer_sequence.append(torch.nn.Dropout(dropout))
            layer_sequence.append(activation)

        layer_sequence.append(torch.nn.Linear(width, phase_space_dim))

        self.stack = torch.nn.Sequential(*layer_sequence)
        self.register_buffer("output_scale", torch.tensor(output_scale))

    def forward(self, X):
        return self.stack(X) * self.output_scale


class NNParticleBeamGenerator(torch.nn.Module):
    def __init__(
            self,
            n_particles,
            energy,
            base_dist=MultivariateNormal(torch.zeros(6), torch.eye(6)),
            transformer=NNTransform(2, 20, output_scale=1e-2),
            **kwargs
    ):
        super(NNParticleBeamGenerator, self).__init__()
        self.transformer = transformer
        self.base_dist = base_dist
        self.register_buffer("beam_energy", energy)

        self.set_base_particles(n_particles, **kwargs)

    def set_base_particles(self, n_particles, **kwargs):
        self.register_buffer(
            "base_particles", self.base_dist.sample(Size([n_particles]))
        )

    def forward(self):
        transformed_beam = self.transformer(self.base_particles)
        transformed_beam = bmad_to_cheetah_coords(
            transformed_beam,
            self.beam_energy,
            torch.tensor(0.511e6)
        )
        return ParticleBeam(*transformed_beam)
