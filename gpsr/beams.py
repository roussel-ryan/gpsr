from abc import abstractmethod, ABC

import torch
from torch import Size, Tensor
from torch.nn import Module
from torch.distributions import MultivariateNormal, Distribution

from cheetah.particles import ParticleBeam
from cheetah.utils.bmadx import bmad_to_cheetah_coords


class BeamGenerator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self) -> ParticleBeam:
        pass


class NNTransform(torch.nn.Module):
    def __init__(
        self,
        n_hidden: int,
        width: int,
        dropout: float = 0.0,
        activation: Module = torch.nn.Tanh(),
        output_scale: float = 1e-2,
        phase_space_dim: int = 6,
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

    def forward(self, X: Tensor) -> Tensor:
        return self.stack(X) * self.output_scale


class NNParticleBeamGenerator(BeamGenerator):
    def __init__(
        self,
        n_particles: int,
        energy: float,
        base_dist: Distribution = MultivariateNormal(torch.zeros(6), torch.eye(6)),
        transformer: NNTransform = NNTransform(2, 20, output_scale=1e-2),
    ):
        super(NNParticleBeamGenerator, self).__init__()
        self.transformer = transformer
        self.base_dist = base_dist
        self.register_buffer("beam_energy", torch.tensor(energy))
        self.register_buffer("particle_charges", torch.tensor(1.0))

        self.set_base_particles(n_particles)

    def set_base_particles(self, n_particles: int):
        self.register_buffer(
            "base_particles", self.base_dist.sample(Size([n_particles]))
        )

    def forward(self) -> ParticleBeam:
        transformed_beam = self.transformer(self.base_particles)
        transformed_beam = bmad_to_cheetah_coords(
            transformed_beam, self.beam_energy, torch.tensor(0.511e6)
        )
        return ParticleBeam(
            *transformed_beam,
            particle_charges=self.particle_charges,
        )
