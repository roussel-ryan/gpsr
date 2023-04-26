import logging
import os
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchensemble import SnapshotEnsembleRegressor

from phase_space_reconstruction import modeling
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.losses import MENTLoss
from phase_space_reconstruction.modeling import ImageDataset

from beamline import create_6d_diagnostic_beamline


def load_data():
    folder = ""

    all_k = torch.load(folder + "kappa.pt").float().unsqueeze(-1)
    all_images = torch.load(folder + "train_images.pt").float()
    xx = torch.stack(torch.load(folder + "xx.pt"))

    bins = xx[0].T[0]
    gt_beam = torch.load(folder + "ground_truth_dist.pt")

    return all_k, all_images, bins, xx, gt_beam


def train_single_model():
    # import and organize data for training
    all_k, all_images, bins, xx, gt_beam = load_data()

    if torch.cuda.is_available():
        all_k = all_k.cuda()
        all_images = all_images.cuda()
        bins = bins.cuda()
        gt_beam = gt_beam.cuda()


    train_dset = ImageDataset(all_k[::2], all_images[::2])
    test_dset = ImageDataset(all_k[1::2], all_images[1::2])

    train_dataloader = DataLoader(train_dset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2

    # create phase space reconstruction model
    diagnostic = ImageDiagnostic(bins, bandwidth=bandwidth)
    lattice = create_6d_diagnostic_beamline()

    n_particles = 10000
    nn_transformer = modeling.NNTransform(2, 20, output_scale=1e-2)
    nn_beam = modeling.InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(10.0e6),
    )

    model = modeling.PhaseSpaceReconstructionModel(
        lattice,
        diagnostic,
        nn_beam
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # train model
    n_epochs = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MENTLoss(torch.tensor(1e11))

    for i in range(n_epochs):
        for batch_idx, elem in enumerate(train_dataloader):
            k, target_images = elem[0], elem[1]

            optimizer.zero_grad()
            output = model(k)
            loss = loss_fn(output, target_images)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            print(i, loss)

    # visualize predictions
    image_pred, entropy_pred, cov_pred = model(all_k)
    n_im = 5
    fig,ax = plt.subplots(2, n_im, sharex="all", sharey="all", figsize=(40,10))
    for i in range(n_im):
        ax[0, i].pcolor(*xx.cpu(), all_images[i*4].squeeze().detach().cpu())
        ax[1, i].pcolor(*xx.cpu(), image_pred[i*4].squeeze().detach().cpu())
    ax[0, 2].set_title("ground truth")
    ax[1, 2].set_title("prediction")


if __name__ == "__main__":
    train_single_model()
    plt.show()

