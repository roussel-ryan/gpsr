from abc import abstractmethod, ABC

import torch
from torch import Size, Tensor
from torch.nn import Module
from torch.distributions import MultivariateNormal, Distribution
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist

from cheetah.particles import ParticleBeam, ParameterBeam
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
    
class BayesianGaussianBeamGenerator(PyroModule):
    """ defines a probablistic model for a Gaussian beam distribution using Pyro"""
    def __init__(self, energy: float, particle_charges: float = 1.0, prior_beamsize_scale: float = 1.0e-7):
        super().__init__()  
        self.register_buffer("energy", torch.tensor(energy))
        self.register_buffer("particle_charges", torch.tensor(particle_charges))

        # define optimization parameters --> used to parameterize the beam distribution
        self.L_omega = PyroSample(
            dist.LKJCholesky(6, torch.tensor(1.0))
        )

        #self.mean = PyroSample(
        #    dist.Normal(
        #        torch.ones(6),
        #        torch.ones(6)*0.1
        #    )
        #)
        #self.scale = PyroSample(
        #    dist.InverseGamma(
        #        torch.ones(6)*1.0,
        #        torch.ones(6)*1.0
        #    )
        #)

        self.theta = PyroSample(
            dist.InverseGamma(
                torch.ones(6)*0.5,
                torch.ones(6)*5.0
            ).to_event(1)
        )

        self.scale = PyroSample(
            dist.HalfNormal(
                torch.ones(1)*0.1
            )
        )

    def calculate_cov(self, L_omega: Tensor, theta: Tensor, scale: Tensor) -> Tensor:
        """ calculate the covariance matrix from the Cholesky factor and the theta parameters in unit (unscaled) space"""
        L_Omega = torch.matmul(
            torch.diag_embed(theta), L_omega
        )
        cov = L_Omega @ L_Omega.transpose(-1, -2)
        return cov * scale * 1e-7
    
    def get_cov_samples(self, n_samples: int) -> Tensor:
        cov_samples = []
        for i in range(n_samples):
            cov_samples += [self.calculate_cov(self.L_omega, self.theta, self.scale)]
        return torch.stack(cov_samples)
    
    def forward(self) -> ParticleBeam:
        cov = self.calculate_cov(self.L_omega, self.theta, self.scale)
        cheetah_cov = torch.eye(7).unsqueeze(0).tile(*cov.shape[:-2], 1, 1)
        cheetah_cov[..., :6, :6] = cov

        return ParameterBeam(
            mu = torch.zeros(7), 
            cov=cheetah_cov, 
            energy=self.energy,
        )



