import logging
import os
import sys

import torch
from losses import WeightedConstrainedLoss
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, random_split, Subset
from torchensemble import SnapshotEnsembleRegressor, VotingRegressor

from utils import kl_div

sys.path.append("../")

from modeling import (
    ImageDataset,
    Imager,
    InitialBeam,
    MaxEntropyQuadScan,
    NonparametricTransform,
    QuadScanTransport,
)

logging.basicConfig(level=logging.INFO)


def create_ensemble(bins, bandwidth):
    defaults = {
        "s": torch.tensor(0.0).float(),
        "p0c": torch.tensor(65.0e6).float(),
        "mc2": torch.tensor(0.511e6).float(),
    }

    transformer = NonparametricTransform(4, 50, 0.0, torch.nn.Tanh())
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))

    module_kwargs = {
        "initial_beam": InitialBeam(100000, transformer, base_dist, **defaults),
        "transport": QuadScanTransport(
            torch.tensor(0.12), torch.tensor(2.84 + 0.54), 1
        ),
        "imager": Imager(bins, bandwidth),
        "condition": False,
    }

    ensemble = SnapshotEnsembleRegressor(
        estimator=MaxEntropyQuadScan, estimator_args=module_kwargs, n_estimators=5
    )

    # ensemble = VotingRegressor(
    #    estimator=MaxEntropyQuadScan,
    #    estimator_args=module_kwargs,
    #    n_estimators=2
    # )
    return ensemble


def get_data(folder):
    all_k = torch.load(folder + "kappa.pt").float()
    all_images = torch.load(folder + "train_images.pt").float()
    xx = torch.load(folder + "xx.pt")
    bins = xx[0].T[0]

    n_samples = 1
    all_k = all_k[:-1, :n_samples]
    all_images = all_images[:-1, :n_samples]
    if torch.cuda.is_available():
        all_k = all_k.cuda()
        all_images = all_images.cuda()

    print(all_images.shape)
    print(all_k.shape)
    print(bins.shape)

    return all_k, all_images, bins, xx


def get_datasets(all_k, all_images, save_dir):
    train_dset = ImageDataset(all_k[::2], all_images[::2])
    test_dset = ImageDataset(all_k[1::2], all_images[1::2])
    torch.save(train_dset, save_dir + "/train.dset")
    torch.save(test_dset, save_dir + "/test.dset")

    return train_dset, test_dset


if __name__ == "__main__":
    folder = ""
    save_dir = "alpha_1000_snapshot"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    all_k, all_images, bins, xx = get_data(folder)
    train_dset, test_dset = get_datasets(all_k, all_images, save_dir)
    print(len(train_dset))

    train_dataloader = DataLoader(train_dset, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2
    ensemble = create_ensemble(bins, bandwidth)

    alpha = torch.tensor(1000.0).to(all_k)
    criterion = WeightedConstrainedLoss(alpha)
    ensemble.set_criterion(criterion)

    n_epochs = 1500
    # ensemble.set_scheduler("StepLR", gamma=0.1, step_size=200, verbose=False)
    ensemble.set_optimizer("Adam", lr=0.001)

    ensemble.fit(
        train_dataloader, epochs=n_epochs, save_dir=save_dir, lr_clip=[0.0001, 10]
    )
    torch.save(criterion.loss_record, save_dir + "/loss_log.pt")
