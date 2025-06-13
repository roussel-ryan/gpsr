"""Train a flow-based generative model.

We begin by training a regular NN generative mode on a set of measured images.
The purpose of this step is three-fold. First, we want to check if a solution 
exists. An entropy-regularized solution may require more iterations with a 
more expensive generative model, so this initial training should not add much
to the overall training time. Second, we want to estimate the covariance 
matrix of the distribution. Flow-based generative models perform best when
the scale or covariance of the distribution is known. For example, in 
density estimation, samples are typically scaled to identity covariance 
before passing through the flow. The covariance matrix is also useful for 
defining a prior distribution for relative entropy estimation. Third, we
may want to compare the NN solution to the entropy-regularized solution to 
see if there is any difference in features.

We then train a normalizing flow to minimize the KL divergence between the
reconstructed distribution :math:`p_{\theta}(x)` and the prior distribution
:math:`q(x)`, where :math:`\theta` represents the generative model parameters.
The KL divergence is the negative of the relative entropy, which we write as 

.. math:: 
    S(\theta) = S[p_{\theta}(x), q(x)] = \int p_{\theta}(x) ( \log p_{\theta}(x) - \log q(x) ) dx.

The entropy can take values over the range :math:`[-\infty, 0]` with a 
maximum at :math:`S[p(x), q(x)] = 0` when p(x) = q(x).

In this script, we train the model soft constraints and a penalty method. At
each epoch, we minimize the loss

.. math:: L(theta) = -S(\theta) + \mu * D(\theta)

with respect to :math:`\theta`, where :math:`D(\theta)` is the mean absolute
error between predicted and measured images and :math:`\mu` is a constant. The
initial epoch pulls the distribution toward the prior, while subsequent epochs
encourage consistency with the data.
"""
import argparse
import os
import pathlib
import time

import torch
import lightning
import numpy as np
import matplotlib.pyplot as plt

from gpsr.beams import NNParticleBeamGenerator 
from gpsr.beams import NSF
from gpsr.beams import EntropyBeamGenerator
from gpsr.beams import GaussianPrior
from gpsr.modeling import GPSR
from gpsr.modeling import GPSRLattice
from gpsr.modeling import GPSR6DLattice
from gpsr.modeling import EntropyGPSR
from gpsr.datasets import SixDReconstructionDataset
from gpsr.datasets import split_dataset
from gpsr.train import LitGPSR
from gpsr.train import EntropyLitGPSR


# Command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--flow-transforms", type=int, default=3)
parser.add_argument("--flow-width", type=int, default=64)
parser.add_argument("--flow-depth", type=int, default=3)
parser.add_argument("--prior-scale", type=float, default=1.1)

parser.add_argument("--nsamp", type=int, default=20_000)
parser.add_argument("--iters", type=int, default=250)
parser.add_argument("--iters-pre", type=int, default=250)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--penalty-min", type=float, default=1000.0)
parser.add_argument("--penalty-max", type=float, default=None)
parser.add_argument("--penalty-step", type=float, default=1000.0)
parser.add_argument("--penalty-scale", type=float, default=2.0)

parser.add_argument("--eval-nsamp", type=int, default=256_000)

parser.add_argument("--plot-nsamp", type=int, default=256_000)
parser.add_argument("--plot-bins", type=int, default=50)
parser.add_argument("--plot-smooth", type=float, default=0.0)

parser.add_argument("--device", type=str, default="cpu")
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

filename =  "example_data/example_datasets/reconstruction_6D.dset"
dset = torch.load(filename, weights_only=False)

train_k_ids = np.arange(0, len(dset.six_d_parameters), 2)
train_dset, test_dset = split_dataset(dset, train_k_ids)


# Lattice
# --------------------------------------------------------------------------------------

p0c = 43.36e+06  # reference particle momentum [eV/c]

screens = train_dset.screens

l_quad = 0.11
l_tdc = 0.01
f_tdc = 1.3e9
phi_tdc = 0.0
l_bend = 0.3018
theta_on = -20.0 * 3.14 / 180.0
l1 = 0.790702
l2 = 0.631698
l3 = 0.889

gpsr_lattice = GPSR6DLattice(
    l_quad, l_tdc, f_tdc, phi_tdc, l_bend, theta_on, l1, l2, l3, *screens
)


# Train NN
# --------------------------------------------------------------------------------------

beam_generator = NNParticleBeamGenerator(args.nsamp, p0c)
gpsr_model = GPSR(beam_generator, gpsr_lattice)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10)

litgpsr = LitGPSR(gpsr_model)
trainer = lightning.Trainer(accelerator=args.device, limit_train_batches=100, max_epochs=args.iters_pre)
trainer.fit(model=litgpsr, train_dataloaders=train_loader)

litgpsr.gpsr_model.eval()
with torch.no_grad():
    litgpsr.gpsr_model.beam_generator.set_base_particles(args.plot_nsamp)
    beam = litgpsr.gpsr_model.beam_generator()

    # Compute covariance matrix
    cov_matrix = torch.cov(beam.particles[:, 0:6].T)

    # Compute plotting limits
    xmax = 4.0 * torch.std(beam.particles[:, 0:6], axis=0)
    limits = [(-float(_xmax), float(_xmax)) for _xmax in xmax]

    # Plot beam
    fig, axs = beam.plot_distribution(
        bins=args.plot_bins,
        bin_ranges=limits,
        plot_2d_kws=dict(
            histogram_smoothing=args.plot_smooth,
            pcolormesh_kws=dict(cmap="Blues"),
        ),
    )
    fig.set_size_inches(9.0, 8.0)
    fig.tight_layout()
    if args.show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "fig_pre_corner.png"), dpi=300)

    # Plot predictions
    params = train_dset.six_d_parameters
    predictions = tuple([ele.detach() for ele in litgpsr.gpsr_model(params)])
    pred_dset = SixDReconstructionDataset(params, predictions, train_dset.screens)
    train_dset.plot_data(
        publication_size=True,
        overlay_data=pred_dset,
        overlay_kwargs=dict(levels=[0.1, 0.5, 0.9]),
    )
    if args.show:
        plt.show()
    plt.savefig(os.path.join(output_dir, f"fig_pre_images.png"), dpi=300)


# Create normalizing flow
# --------------------------------------------------------------------------------------

gen_model = NSF(
    ndim=ndim, 
    cov_matrix=cov_matrix, 
    transforms=args.flow_transforms, 
    hidden_units=args.flow_width, 
    hidden_layers=args.flow_depth,
)

prior_loc = torch.zeros(ndim)
prior_cov = cov_matrix * args.prior_scale**2
prior = GaussianPrior(prior_loc, prior_cov)

beam_generator = EntropyBeamGenerator(
    gen_model=gen_model, 
    prior=prior, 
    n_particles=args.nsamp,
    energy=p0c,
)

gpsr_model = EntropyGPSR(beam_generator=beam_generator, lattice=gpsr_lattice)


# Train normalizing flow
# --------------------------------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(train_dset, batch_size=len(train_k_ids))

litgpsr = EntropyLitGPSR(gpsr_model, lr=args.lr, penalty=args.penalty_min)
trainer = lightning.Trainer(
    accelerator=args.device,
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
        litgpsr.gpsr_model.eval()
        litgpsr.gpsr_model.beam_generator.set_base_particles(args.plot_nsamp)

        # Generate beam and predictions
        params = train_dset.six_d_parameters
        beam, entropy, predictions = litgpsr.gpsr_model(params)
        predictions = tuple([ele.detach() for ele in predictions])
        pred_dset = SixDReconstructionDataset(params, predictions, train_dset.screens)

        # Plot beam (corner plot)
        fig, axs = beam.plot_distribution(
            bins=args.plot_bins,
            bin_ranges=limits,
            plot_2d_kws=dict(
                histogram_smoothing=args.plot_smooth,
                pcolormesh_kws=dict(cmap="Blues"),
            ),
        )
        fig.set_size_inches(9.0, 8.0)
        fig.tight_layout()
        if args.show:
            plt.show()
        plt.savefig(os.path.join(output_dir, f"fig_corner_{epoch:02.0f}.png"), dpi=300)

        # Plot predictions
        train_dset.plot_data(
            publication_size=True,
            overlay_data=pred_dset,
            overlay_kwargs=dict(levels=[0.1, 0.5, 0.9]),
        )
        if args.show:
            plt.show()
        plt.savefig(os.path.join(output_dir, f"fig_images_{epoch:02.0f}.png"), dpi=300)

        # Reset particle count
        litgpsr.gpsr_model.beam_generator.set_base_particles(args.nsamp)
        litgpsr.gpsr_model.train()