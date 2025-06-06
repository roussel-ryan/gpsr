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
the reconstructed distribution :math:`p_{\theta}(x)` and the prior distribution 
:math:`q(x)`, where :math:`\theta` represents the generative model parameters. The 
KL divergence is the negative of the relative entropy, which we write as 

.. math:: 
    S(\theta) = S[p_{\theta}(x), q(x)] = \int p_{\theta}(x) ( \log p_{\theta}(x) - \log q(x) ) dx.

The entropy can take values over the range :math:`[-\infty, 0]` with a maximum at
:math:`S[p(x), q(x)] = 0` when p(x) = q(x).

In this script, we train the model soft constraints and a penalty method. At each epoch,
we minimize the loss

.. math:: L(theta) = -S(\theta) + \mu * D(\theta)

with respect to :math:`\theta`, where :math:`D(\theta)` is the mean absolute error 
between predicted and measured images and :math:`\mu` is a constant. The initial epoch 
pulss the distribution to the prior, while subsequent epochs encourage consistency with
the data.
"""

# To do:
# - Add callback to print/plot loss for data fit and entropy.
# - Fix GPU issue with Pytorch Lighting.

import abc
import argparse
import math
import sys
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
parser.add_argument("--iters-pre", type=int, default=250)
parser.add_argument("--iters", type=int, default=250)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--prior-scale", type=float, default=1.25)
parser.add_argument("--penalty-min", type=float, default=1000.0)
parser.add_argument("--penalty-max", type=float, default=None)
parser.add_argument("--penalty-step", type=float, default=200.0)
parser.add_argument("--penalty-scale", type=float, default=2.0)
parser.add_argument("--eval-nsamp", type=int, default=100_000)
parser.add_argument("--device", type=str, default="cpu")  # "auto", "cpu", "mps", ...
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
    def __init__(self, ndim: int, cov_matrix: torch.Tensor = None, device: torch.device = None) -> None:
        super().__init__()

        self.ndim = ndim
        self.device = device

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

    def to(self, device):
        self.device = device
        self.unnorm_matrix = self.unnorm_matrix.to(device)
        self.norm_matrix = self.norm_matrix.to(device)
        return self

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
        device: torch.device = None
    ) -> None:
        """Constructor.
        
        flow: Flow-based generative model.
        prior: Prior distribution over the phase space coordiantes. Must implement
               `prior.log_prob(x: torch.Tensor) -> torch.Tensor`, where `x` is 
               a set of particle coordinates.
        n_particles: Number of macro-particles in the beam
        energy: Reference particle energy [eV].
        mass: Reference particle mass [eV/c^2]. Defaults to electron mass.
        particle_charges: Macro-particle charges [C].
        """
        super(FlowBeamGenerator, self).__init__()

        self.n_dim = 6
        self.n_particles = n_particles

        self.flow = flow
        self.prior = prior

        self.device = device

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

class FlowLitGPSR(lightning.LightningModule, abc.ABC):
    def __init__(self, gpsr_model: FlowGPSR, lr: float = 0.001, penalty: float = 0.0) -> None:
        super().__init__()
        self.gpsr_model = gpsr_model
        self.lr = lr
        self.penalty = penalty

    def training_step(self, batch, batch_idx):
        # get the training data batch
        x, y = batch

        # make predictions using the GPSR model
        beam, entropy, pred = gpsr_model(x)

        # check to make sure the prediction shape matches the target shape EXACTLY
        # removing this check will allow the model to run, but it will not be correct
        for i in range(len(pred)):
            if not pred[i].shape == y[i].shape:
                raise RuntimeError(
                    f"prediction {i} shape {pred[i].shape} does not match target shape {y[i].shape}"
                )

        # normalize images
        y_normalized = [normalize_images(y_ele) for y_ele in y]
        pred_normalized = [normalize_images(pred_ele) for pred_ele in pred]

        # add up the loss functions from each prediction (in a tuple)
        diff = [
            mae_loss(y_ele, pred_ele)
            for y_ele, pred_ele in zip(y_normalized, pred_normalized)
        ]

        loss_pred = 0.0
        if len(diff) > 1:
            loss_pred += torch.add(*diff) / len(diff)
        else:
            loss_pred += diff[0]

        loss_reg = -entropy

        loss = loss_reg + self.penalty * loss_pred

        self.log("loss_pred", loss_pred, on_epoch=True)
        self.log("loss_reg", loss_reg, on_epoch=True)
        self.log("loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gpsr_model.parameters(), lr=self.lr)
        return optimizer


## Start with NN generator to estimate covariance matrix.
gpsr_model = GPSR(NNParticleBeamGenerator(args.nsamp, p0c), gpsr_lattice)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10)

litgpsr = LitGPSR(gpsr_model)
trainer = lightning.Trainer(accelerator=args.device, limit_train_batches=100, max_epochs=args.iters_pre)
trainer.fit(model=litgpsr, train_dataloaders=train_loader)

beam = litgpsr.gpsr_model.beam_generator()

## Analyze beam
x = beam.particles[:, :6].detach().clone()
cov_matrix = torch.cov(x.T)
xmax = 4.0 * torch.std(x, axis=0)
limits = [(-float(_xmax), float(_xmax)) for _xmax in xmax]

fig, axs = beam.plot_distribution(
    bins=50,
    bin_ranges=limits,
    plot_2d_kws=dict(
        pcolormesh_kws=dict(cmap="Blues"),
    ),
)
fig.set_size_inches(7.0, 7.0)
plt.show()


## Create flow-based model
ndim = 6
flow = zuko.flows.NSF(features=ndim, transforms=3, hidden_features=(3 * [64]))
flow = zuko.flows.Flow(flow.transform.inv, flow.base)
flow = ZukoFlow(flow=flow, ndim=ndim, cov_matrix=cov_matrix)

prior_loc = torch.zeros(ndim)
prior_cov = cov_matrix * args.prior_scale**2
prior = torch.distributions.MultivariateNormal(prior_loc, prior_cov)

beam_generator = FlowBeamGenerator(
    flow=flow, 
    prior=prior, 
    n_particles=args.nsamp,
    energy=p0c,
)

gpsr_model = FlowGPSR(beam_generator=beam_generator, lattice=gpsr_lattice)


## Train flow-based model
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10)

litgpsr = FlowLitGPSR(gpsr_model, lr=args.lr, penalty=args.penalty_min)
trainer = lightning.Trainer(
    accelerator=args.device,  # GPU not working
    limit_train_batches=100, 
    max_epochs=args.iters,
)

for epoch in range(args.epochs):
    print("EPOCH={}".format(epoch))
    print("PENALTY={}".format(litgpsr.penalty))

    trainer.fit(model=litgpsr, train_dataloaders=train_loader)
    trainer.fit_loop.max_epochs += args.iters

    # Update penalty parameter
    litgpsr.penalty *= args.penalty_scale
    litgpsr.penalty += args.penalty_step
    if args.penalty_max is not None:
        litgpsr.penalty = min(litgpsr.penalty, args.penalty_max)

    with torch.no_grad():
        # Increase particle count
        gpsr_model.eval()
        gpsr_model.beam_generator.n_particles = args.eval_nsamp

        # Generate beam
        beam, _ = gpsr_model.beam_generator()
        print(beam.particles.shape)

        # Make predictions
        beam_out, entropy, predictions = gpsr_model(train_dset.parameters)
        pred_dset = QuadScanDataset(train_dset.parameters, predictions[0].detach(), train_dset.screen)

        # Plot predictions
        fig, ax = train_dset.plot_data(overlay_data=pred_dset)
        fig.set_size_inches(20, 3)
        plt.show()

        # Plot beam
        fig, axs = beam.plot_distribution(
            bins=50,
            bin_ranges=limits,
            plot_2d_kws=dict(
                pcolormesh_kws=dict(cmap="Blues"),
            ),
        )
        fig.set_size_inches(7.0, 7.0)
        plt.show()

        # Reset particle count
        gpsr_model.beam_generator.n_particles = args.nsamp
        gpsr_model.train()
