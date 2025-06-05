"""Train a flow-based generative model in the GPSR framework.

We begin by training a regular NN generative mode on a set of measured images.
The purpose of this step is two-fold. First, we want to check if a solution exists. 
Entropy-based training may require multiple epochs with a more expensive generative 
model, so this initial training should not add much to the overall training time. 
Failure to fit the data with the NN likely points to an incorrecet forward model.
(It is also possible (but unlikely) that the NN is not flexible enough to represent
the phase space distribution.) 

Second, we want to estimate the covariance matrix of the distribution. The covariance
matrix is typically overdetermined by the measurements, meaning all solutions should 
have the same second-order moments (<xx>, <xy>, etc.) but possibly different higher-
order moments. Flow-based generative models perform best when the scale or covariance
of the distribution is known ahead of time. For example, in density estimation, samples
are typically scaled to identity covariance before passing through the flow. The 
covariance matrix is also useful for defining a prior distribution over the phase 
space coordinates for relative entropy estimation.

We then train a flow-based generative model to minimize the KL divergence between
the reconstructed distribution p_{\theta}(x) and the prior distribution q(x), where
\theta represents the generative model parameters. The KL divergence is the negative
of the relative entropy, which we write as 

S(\theta) = S[p_{\theta}(x), q(x)] = \int p_{\theta}(x) ( \log p_{\theta}(x) - \log q(x) ) dx.

The entropy can take values over the range [-\infty, 0] with a maximum at S[p, q] = 0 
when p(x) = q(x).

In this script, we train the model soft constraints and a penalty method. At each epoch,
we minimize the loss

L(theta) = -S(theta) + \mu * D(theta)

with respect to \theta, where D(theta) is the mean absolute error between predicted 
and measured images and \mu is a constant. The initial epoch will pull the distribution to
the prior, while subsequent epochs will encourage consistency with the data.
"""

import abc
import argparse
from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import lightning
import numpy as np
import torch
import zuko

from cheetah.particles import ParticleBeam
from cheetah.utils.bmadx import bmad_to_cheetah_coords

from gpsr.beams import NNParticleBeamGenerator 
from gpsr.modeling import GPSR
from gpsr.modeling import GPSRLattice
from gpsr.modeling import GPSRQuadScanLattice
from gpsr.modeling import BeamGenerator
from gpsr.datasets import QuadScanDataset
from gpsr.datasets import split_dataset
from gpsr.losses import mae_loss
from gpsr.losses import normalize_images
from gpsr.train import LitGPSR


# Command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nsamp", type=int, default=10_000)
parser.add_argument("--iters", type=int, default=250)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--prior-scale", type=float, default=1.5)
parser.add_argument("--penalty-min", type=float, default=500.0)
parser.add_argument("--penalty-max", type=float, default=None)
parser.add_argument("--penalty-step", type=float, default=0.0)
parser.add_argument("--penalty-scale", type=float, default=2.0)
parser.add_argument("--eval-nsamp", type=int, default=100_000)
args = parser.parse_args()


# Data
# --------------------------------------------------------------------------------------

filename =  "example_data/example_datasets/reconstruction_4D.dset"
dset = torch.load(filename, weights_only=False)

train_k_ids = np.arange(0, len(dset.parameters), 2)
train_dset, test_dset = split_dataset(dset, train_k_ids)


# Lattice
# --------------------------------------------------------------------------------------

p0c = 43.36e+06  # reference particle momentum [eV/c]
gpsr_lattice = GPSRQuadScanLattice(l_quad=0.1, l_drift=1.0, screen=train_dset.screen)


# Model
# --------------------------------------------------------------------------------------

class Flow(torch.nn.Module, abc.ABC):
    """Base class for flow-based generative models.
    
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

        self.cov_matrix = cov_matrix
        if self.cov_matrix is None:
            self.cov_matrix = torch.eye(self.ndim)

        self.unnorm_matrix = torch.linalg.cholesky(cov_matrix)
        self.unnorm_matrix_log_det = torch.log(torch.linalg.det(self.unnorm_matrix))
        self.norm_matrix = torch.linalg.inv(self.unnorm_matrix)

    @abc.abstractmethod
    def _sample(self, n: int) -> torch.Tensor:
        """Generate samples {z_i}."""
        pass

    @abc.abstractmethod
    def _log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities {log(p(z_i))}."""
        pass

    @abc.abstractmethod
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


class ZukoFlow(Flow):
    """Implements flow-based generative model using the Zuko library."""
    def __init__(self, flow: zuko.flows.Flow, ndim: int, cov_matrix: torch.Tensor = None) -> None:
        super().__init__(ndim=ndim, cov_matrix=cov_matrix)
        self._flow = flow

    def _sample(self, n: int) -> torch.Tensor:
        return self._flow().rsample((n,))

    def _log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self._flow().log_prob(z)
    
    def _sample_and_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._flow().rsample_and_log_prob((n,))   
        

class FlowBeamGenerator(BeamGenerator):
    def __init__(
        self,
        flow: Flow,
        prior: Any,
        n_particles: int,
        energy: float,
        mass: float = 0.511e+06,
        particle_charges: float = 1.0,
    ) -> None:
        """Constructor.
        
        flow: Flow-based generative model.
        prior: Prior distribution over the phase space coordiantes. Must implement
               `prior.log_prob(x: torch.Tensor) -> torch.Tensor`, where `x` is 
               a set of particle coordinates.
        n_particles: Number of macro-particles in the beam
        energy: Reference particle energy [eV].
        mass: Reference particle mass [eV/c^2]. Defaults to electron mass.
        particle_charges: Macro-particle charges [C]. Defaults to 1.
        """
        super(FlowBeamGenerator, self).__init__()

        self.n_dim = 6
        self.n_particles = n_particles

        self.flow = flow
        self.prior = prior

        self.register_buffer("energy", torch.tensor(energy))
        self.register_buffer("mass", torch.tensor(mass))
        self.register_buffer("particle_charges", torch.tensor(particle_charges))
    
    def forward(self) -> tuple[ParticleBeam, torch.Tensor]:
        """Return beam and estimated entropy."""
        x, log_p = self.flow.sample_and_log_prob(self.n_particles)

        log_q = 0.0
        if self.prior is not None:
            log_q = self.prior.log_prob(x)

        entropy = -torch.mean(log_p - log_q)

        x = bmad_to_cheetah_coords(x, self.energy, self.mass)
        beam = ParticleBeam(*x, particle_charges=self.particle_charges)
        return (beam, entropy)
    

class FlowGPSR(GPSR):
    def __init__(self, beam_generator: FlowBeamGenerator, lattice: GPSRLattice) -> None:
        super().__init__(beam_generator=beam_generator, lattice=lattice)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return beam, entropy estimate, and predicted images."""
        beam, entropy = self.beam_generator()
        self.lattice.set_lattice_parameters(x)
        predictions = self.lattice.track_and_observe(beam)
        return (beam, entropy, predictions)
    

# Training
# --------------------------------------------------------------------------------------

## Start with NN generator to estimate covariance matrix.
gpsr_model = GPSR(NNParticleBeamGenerator(10_000, p0c), gpsr_lattice)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10)

litgpsr = LitGPSR(gpsr_model)
trainer = lightning.Trainer(limit_train_batches=100, max_epochs=250)
trainer.fit(model=litgpsr, train_dataloaders=train_loader)
        
beam = litgpsr.gpsr_model.beam_generator()

# Analyze beam
x = beam.particles[:, :6].detach().clone()
cov_matrix = torch.cov(x.T)
xmax = 4.0 * torch.std(x, axis=0)
limits = [(-float(_xmax), float(_xmax)) for _xmax in xmax]

beam.plot_distribution(
    bins=50,
    bin_ranges=limits,
    plot_2d_kws=dict(
        pcolormesh_kws=dict(cmap="Blues"),
    ),
)
plt.show()


## Create flow-based generative model
ndim = 6
flow = zuko.flows.NSF(features=ndim, transforms=3, hidden_features=(3 * [64]))
flow = zuko.flows.Flow(flow.transform.inv, flow.base)
flow = ZukoFlow(flow=flow, ndim=ndim, cov_matrix=cov_matrix)

prior_loc = torch.zeros(ndim)
prior_cov = cov_matrix * 1.0
prior = torch.distributions.MultivariateNormal(prior_loc, prior_cov)

beam_generator = FlowBeamGenerator(
    flow=flow, 
    prior=prior, 
    n_particles=10_000,
    energy=p0c,
)

gpsr_model = FlowGPSR(beam_generator=beam_generator, lattice=gpsr_lattice)


## Train flow-based generative model
def train_epoch(gpsr_model, dset, lr: float, penalty: float, iterations: int) -> dict:    
    optimizer = torch.optim.Adam(gpsr_model.parameters(), lr=lr)

    history = {}
    history["loss"] = []
    history["loss_reg"] = []
    history["loss_pred"] = []
    
    for iteration in range(iterations):
        optimizer.zero_grad()

        beam, entropy, predictions = gpsr_model(dset.parameters)

        loss_reg = -entropy

        y_meas = [normalize_images(y) for y in dset.observations]
        y_pred = [normalize_images(y) for y in predictions]
        diff = [mae_loss(_y_meas, _y_pred) for _y_meas, _y_pred in zip(y_meas, y_pred)]

        loss_pred = 0.0
        if len(diff) > 1:
            loss_pred += torch.add(*diff) / len(diff)
        else:
            loss_pred += diff[0]

        loss = loss_reg + loss_pred * penalty

        loss.backward()
        optimizer.step()

        print(f"iter={iteration} loss_reg={loss_reg.item():0.3e} loss_pred={loss_pred.item():0.3e}")

        history["loss"].append(loss.item())
        history["loss_reg"].append(loss_reg.item())
        history["loss_pred"].append(loss_pred.item())

    return history


penalty = args.penalty_min
for epoch in range(args.epochs):
    print("EPOCH={}".format(epoch))
    print("penalty={}".format(penalty))

    # Train generative model
    history = train_epoch(
        gpsr_model=gpsr_model,
        dset=train_dset,
        lr=args.lr,
        penalty=penalty,
        iterations=args.iters,
    )

    # Update penalty parameter
    penalty *= args.penalty_scale
    penalty += args.penalty_step
    if args.penalty_max is not None:
        penalty = min(penalty, args.penalty_max)

    with torch.no_grad():
        gpsr_model.beam_generator.n_particles = args.eval_nsamp

        # Plot loss
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        color = "red"
        ax1.plot(history["loss_pred"])
        ax2.plot(history["loss_reg"], color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax1.set_ylabel("MAE")
        ax2.set_ylabel("Negative entropy", color=color)
        plt.show()

        # Plot data
        beam, entropy, predictions = gpsr_model(train_dset.parameters)
        pred_dset = QuadScanDataset(train_dset.parameters, predictions[0].detach(), train_dset.screen)

        fig, ax = train_dset.plot_data(overlay_data=pred_dset)
        fig.set_size_inches(20, 3)
        plt.show()

        # Plot beam
        beam.plot_distribution(
            bins=50,
            bin_ranges=limits,
            plot_2d_kws=dict(
                pcolormesh_kws=dict(cmap="Blues"),
            ),
        )
        plt.show()

        gpsr_model.beam_generator.n_particles = args.nsamp
