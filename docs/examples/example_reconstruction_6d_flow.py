"""Train a flow-based generative model.

We start by training a NN generative model. The purpose of this step is to check 
that a solution exists and to compare to the regularized solutions later on. 
This step also estimates the 6 x 6 covariance matrix, which allows us to define
the generative model in a normalized space without linear correlations between 
variables. The covariance matrix is also useful for defining a prior 
distribution over the phase space coordiantes.

We then train a normalizing flow to maximize the entropy of the model 
distribution :math:`p_{\theta}(x)` relative to a Gaussian prior :math:`q(x)`, 
where :math:`\theta` represents the model parameters. We encourage consistency 
with the data using a soft penalty method. For an increasing series of penalty
parameters :math:`\lambda`, we minimize the following regularized loss function:

.. math:: L(theta) = -S(\theta) + \lambda * D(\theta),

where :math:`S(\theta)` is the entropy and :math:`D(\theta)` is the mean 
absolute prediction error.
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
from gpsr.losses import mae_loss
from gpsr.losses import normalize_images
from gpsr.modeling import GPSR
from gpsr.modeling import GPSR6DLattice
from gpsr.modeling import EntropyGPSR
from gpsr.datasets import SixDReconstructionDataset
from gpsr.datasets import split_dataset
from gpsr.train import LitGPSR
from gpsr.train import EntropyLitGPSR


def main(args: argparse.Namespace) -> None:

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

    train_k_ids = np.arange(0, len(dset.six_d_parameters), args.data_split)
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
    trainer = lightning.Trainer(limit_train_batches=100, max_epochs=args.iters_pre)
    trainer.fit(model=litgpsr, train_dataloaders=train_loader)

    with torch.no_grad():
        litgpsr.gpsr_model.eval()
        litgpsr.gpsr_model.beam_generator.set_base_particles(args.plot_nsamp)
        beam = litgpsr.gpsr_model.beam_generator()

        # Compute covariance matrix
        cov_matrix = torch.cov(beam.particles[:, 0:6].T)
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

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10)

    litgpsr = EntropyLitGPSR(gpsr_model, lr=args.lr, penalty=args.penalty_min)
    trainer = lightning.Trainer(limit_train_batches=100, max_epochs=args.iters)

    loss_pred_old = 1.00e+14
    loss_reg_old = 1.00e+14

    for epoch in range(args.epochs):
        # Train
        print("EPOCH={}".format(epoch))
        print("PENALTY={}".format(litgpsr.penalty))

        trainer.fit(model=litgpsr, train_dataloaders=train_loader)
        trainer.fit_loop.max_epochs += args.iters

        # Plot
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

        # Evaluate
        with torch.no_grad():
            litgpsr.gpsr_model.eval()
            litgpsr.gpsr_model.beam_generator.set_base_particles(args.plot_nsamp)

            # Compute losses
            parameters = train_dset.parameters
            y_meas = train_dset.observations[0]
            beam, entropy, y_pred = litgpsr.gpsr_model(parameters)
            y_meas = [normalize_images(y) for y in y_meas]
            y_pred = [normalize_images(y) for y in y_pred]
            loss_pred = [mae_loss(ym, yp) for ym, yp in zip(y_meas, y_pred)]
            loss_pred = sum(loss_pred) / len(loss_pred)

            # Early stopping
            loss_pred_rel = ((loss_pred - loss_pred_old) / loss_pred_old)
            loss_pred_old = loss_pred
            
            print("LOSS (PRED) = {:0.3e}".format(loss_pred))
            print("LOSS (PRED) REL CHANGE = {:0.3e}".format(loss_pred_rel))

            if (epoch > 2) and (loss_pred_rel > -args.loss_pred_rtol):
                print("Stopping early")
                break

            litgpsr.gpsr_model.beam_generator.set_base_particles(args.nsamp)
            litgpsr.gpsr_model.train()

        # Update penalty parameter
        litgpsr.penalty *= args.penalty_scale
        litgpsr.penalty += args.penalty_step
        if args.penalty_max is not None:
            litgpsr.penalty = min(litgpsr.penalty, args.penalty_max)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-split", type=int, default=2)

    parser.add_argument("--flow-transforms", type=int, default=3)
    parser.add_argument("--flow-width", type=int, default=64)
    parser.add_argument("--flow-depth", type=int, default=3)
    parser.add_argument("--prior-scale", type=float, default=1.1)

    parser.add_argument("--nsamp", type=int, default=20_000)
    parser.add_argument("--iters", type=int, default=250)
    parser.add_argument("--iters-pre", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--penalty-min", type=float, default=2000)
    parser.add_argument("--penalty-max", type=float, default=None)
    parser.add_argument("--penalty-step", type=float, default=10_000)
    parser.add_argument("--penalty-scale", type=float, default=3.0)
    parser.add_argument("--loss-pred-rtol", type=float, default=0.01)

    parser.add_argument("--plot-nsamp", type=int, default=256_000)
    parser.add_argument("--plot-bins", type=int, default=50)
    parser.add_argument("--plot-smooth", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    main(args)