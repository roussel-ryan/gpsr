import math
from abc import abstractmethod, ABC
from typing import Any

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


class GenModel(torch.nn.Module, ABC):
    """Base class for generative model.
    
    The generative model is defined for coordinates z. The phase space coordinates x 
    are obtained by a linear transformation x = Lz, where L is obtained by a Cholesky 
    decomposition of the covariance matrix: S = <xx^T> = LL^T. The probability densities
    are related by p(x) = p(z) / |det(L)|.

    This allows the base distribution of the generative model to always have identify
    covariance matrix.
    """
    def __init__(self, ndim: int, cov_matrix: torch.Tensor = None) -> None:
        super().__init__()

        self.ndim = ndim

        # self.cov_matrix = cov_matrix
        # if self.cov_matrix is None:
        #     self.cov_matrix = torch.eye(self.ndim)

        # self.unnorm_matrix = torch.linalg.cholesky(cov_matrix)
        # self.unnorm_matrix_log_det = torch.log(torch.linalg.det(self.unnorm_matrix))
        # self.norm_matrix = torch.linalg.inv(self.unnorm_matrix)

        cov_matrix = torch.clone(cov_matrix)
        if cov_matrix is None:
            cov_matrix = torch.eye(self.ndim)
        self.register_buffer("cov_matrix", cov_matrix)

        unnorm_matrix = torch.linalg.cholesky(cov_matrix)
        unnorm_matrix_log_det = torch.log(torch.linalg.det(unnorm_matrix))
        norm_matrix = torch.linalg.inv(unnorm_matrix)

        self.register_buffer("unnorm_matrix", unnorm_matrix)
        self.register_buffer("unnorm_matrix_log_det", unnorm_matrix_log_det)
        self.register_buffer("norm_matrix", norm_matrix)

    @abstractmethod
    def _sample(self, n: int) -> torch.Tensor:
        """Generate samples {z_i}."""
        pass

    @abstractmethod
    def _log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities {log(p(z_i))}."""
        pass

    @abstractmethod
    def _sample_and_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples {z_i} and log probabilities {log(p(z_i))}."""
        pass

    def sample(self, n: int) -> torch.Tensor:
        """Generate samples {x_i}."""
        z = self._sample(n)
        x = torch.matmul(z, self.unnorm_matrix.T)
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities {log(p(x_i))}."""
        z = torch.matmul(x, self.norm_matrix.T)
        log_prob = self._log_prob(z)
        log_prob = log_prob - self.unnorm_matrix_log_det
        return log_prob
    
    def sample_and_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples {x_i} and log probabilities {log(p(x_i))}."""
        z, log_prob = self._sample_and_log_prob(n)
        x = torch.matmul(z, self.unnorm_matrix.T)
        log_prob = log_prob - self.unnorm_matrix_log_det
        return (x, log_prob)
    

class NSF(GenModel):
    """Implements flow-based generative model using rational-quadratic splines.
    
    This class uses the Zuko library.
    """
    def __init__(
        self,
        transforms: int = 3, 
        hidden_layers: int = 3, 
        hidden_units: int = 64, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        import zuko

        self._flow = zuko.flows.NSF(
            features=self.ndim, 
            transforms=transforms, 
            hidden_features=(hidden_layers * [hidden_units])
        )
        self._flow = zuko.flows.Flow(self._flow.transform.inv, self._flow.base)
    
    def _sample(self, n: int) -> torch.Tensor:
        return self._flow().rsample((n,))

    def _log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self._flow().log_prob(z)
    
    def _sample_and_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._flow().rsample_and_log_prob((n,))   


class EntropyBeamGenerator(BeamGenerator):
    """Generates beam and entropy in forward pass."""
    def __init__(
        self,
        gen_model: GenModel,
        prior: Any,
        n_particles: int,
        energy: float,
        mass: float = 0.511e+06,
        particle_charges: float = 1.0,
        device: torch.device = None
    ) -> None:
        """Constructor.
        
        gen_model: Generative model
        prior: Prior distribution over the phase space coordiantes. Must implement
               `prior.log_prob(x: torch.Tensor) -> torch.Tensor`, where `x` is 
               a set of particle coordinates.
        n_particles: Number of macro-particles in the beam
        energy: Reference particle energy [eV].
        mass: Reference particle mass [eV/c^2]. Defaults to electron mass.
        particle_charges: Macro-particle charges [C].
        """
        super(EntropyBeamGenerator, self).__init__()

        self.n_dim = 6
        self.n_particles = n_particles

        self.gen_model = gen_model
        self.prior = prior

        self.device = device

        self.register_buffer("energy", torch.tensor(energy))
        self.register_buffer("mass", torch.tensor(mass))
        self.register_buffer("particle_charges", torch.tensor(particle_charges))

    def set_base_particles(self, n_particles: int) -> None:
        self.n_particles = n_particles
    
    def forward(self) -> tuple[ParticleBeam, torch.Tensor]:
        """Return beam and estimated entropy."""
        x, log_p = self.gen_model.sample_and_log_prob(self.n_particles)

        log_q = 0.0
        if self.prior is not None:
            log_q = self.prior.log_prob(x)

        entropy = -torch.mean(log_p - log_q)

        x = bmad_to_cheetah_coords(x, self.energy, self.mass)
        beam = ParticleBeam(*x, particle_charges=self.particle_charges)
        return (beam, entropy)
    

# Classes in torch.distribution classes cannot be sent to GPU.
class Prior(torch.nn.Module, ABC):
    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def sample(self, n: int) -> torch.Tensor:
        pass


class GaussianPrior(Prior):
    def __init__(self, loc: torch.Tensor, cov: torch.Tensor) -> None:
        super().__init__()
        
        self.ndim = len(loc)

        self.register_buffer("loc", loc)
        self.register_buffer("cov", cov)
        self.register_buffer("cov_inv", torch.linalg.inv(cov))
        self.register_buffer("log_cov_det", torch.log(torch.linalg.det(cov)))
        self.register_buffer("L", torch.linalg.cholesky(cov))
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        _x = x - self.loc
        _log_prob = 0.0
        _log_prob -= 0.5 * torch.sum(_x * torch.matmul(_x, self.cov_inv.T), axis=1)
        _log_prob -= 0.5 * (self.ndim * math.log(2.0 * math.pi) + self.log_cov_det)
        return _log_prob
    
    def sample(self, n: int) -> torch.Tensor:
        x = torch.randn((n, self.ndim))
        x = torch.matmul(x, self.L.T)
        return x