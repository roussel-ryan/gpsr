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
import os
import pathlib
import sys
import time
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
from gpsr.beams import NSF
from gpsr.beams import EntropyBeamGenerator
from gpsr.modeling import GPSR
from gpsr.modeling import GPSRLattice
from gpsr.modeling import GPSRQuadScanLattice
from gpsr.modeling import EntropyGPSR
from gpsr.datasets import QuadScanDataset
from gpsr.datasets import split_dataset
from gpsr.losses import mae_loss
from gpsr.losses import normalize_images
from gpsr.train import LitGPSR
from gpsr.train import EntropyLitGPSR


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
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

timestamp = time.strftime("%y%m%d%H%M%S")

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ndim = 6


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
    

# Covariance matrix estimation
# --------------------------------------------------------------------------------------

# Start with NN generator to estimate covariance matrix.
gpsr_model = GPSR(NNParticleBeamGenerator(args.nsamp, p0c), gpsr_lattice)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10)

litgpsr = LitGPSR(gpsr_model)
trainer = lightning.Trainer(accelerator=args.device, limit_train_batches=100, max_epochs=args.iters_pre)
trainer.fit(model=litgpsr, train_dataloaders=train_loader)

beam = litgpsr.gpsr_model.beam_generator()

# Analyze beam
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
if args.show:
    plt.show()
plt.savefig(os.path.join(output_dir, "fig_pre_corner.png"), dpi=300)


# We do not expect to reconstruct the 6D distribution from these measurements,
# so we will get rid of the linear longitudinal-transverse correlations in
# the covariance matrix.
cov_matrix_old = torch.clone(cov_matrix)
cov_matrix = torch.eye(ndim)
cov_matrix[4, 4] = 1.0
cov_matrix[5, 5] = 0.005
cov_matrix[0:4, 0:4] = cov_matrix_old[0:4, 0:4]


# Create flow-based model
# --------------------------------------------------------------------------------------

gen_model = NSF(
    ndim=ndim, cov_matrix=cov_matrix, transforms=3, hidden_units=64, hidden_layers=3
)

prior_loc = torch.zeros(ndim)
prior_cov = cov_matrix * args.prior_scale**2
prior = torch.distributions.MultivariateNormal(prior_loc, prior_cov)

beam_generator = EntropyBeamGenerator(
    gen_model=gen_model, 
    prior=prior, 
    n_particles=args.nsamp,
    energy=p0c,
)

gpsr_model = EntropyGPSR(beam_generator=beam_generator, lattice=gpsr_lattice)


# Train flow-based model
# --------------------------------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(train_dset, batch_size=len(train_k_ids))

litgpsr = EntropyLitGPSR(gpsr_model, lr=args.lr, penalty=args.penalty_min)
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
        if args.show:
            plt.show()
        plt.savefig(os.path.join(output_dir, f"fig_images_{epoch:02.0f}.png"), dpi=300)

        # Plot beam (corner plot)
        fig, axs = beam.plot_distribution(
            bins=50,
            # bin_ranges=limits,
            plot_2d_kws=dict(
                pcolormesh_kws=dict(cmap="Blues"),
            ),
        )
        fig.set_size_inches(7.0, 7.0)
        if args.show:
            plt.show()
        plt.savefig(os.path.join(output_dir, f"fig_corner_{epoch:02.0f}.png"), dpi=300)

        # Reset particle count
        gpsr_model.beam_generator.n_particles = args.nsamp
        gpsr_model.train()
