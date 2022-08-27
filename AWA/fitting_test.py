import logging
import os

import sys

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, random_split, Subset
from torchensemble import SnapshotEnsembleRegressor, VotingRegressor

from utils import kl_div

sys.path.append("../../")

from modeling import (
    ImageDataset,
    Imager,
    InitialBeam,
    NonparametricTransform,
    QuadScanModel,
    QuadScanTransport,
)

logging.basicConfig(level=logging.INFO)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)


class MaxEntropyQuadScan(QuadScanModel):
    def forward(self, K):
        initial_beam = self.beam_generator()
        output_beam = self.lattice(initial_beam, K)
        output_coords = torch.cat(
            [output_beam.x.unsqueeze(0), output_beam.y.unsqueeze(0)]
        )
        output_images = self.imager(output_coords)

        cov = (
                torch.cov(initial_beam.data.T)
                + torch.eye(6, device=initial_beam.data.device) * 1e-8
        )
        exp_factor = torch.det(2 * 3.14 * 2.71 * cov)

        return output_images, -0.5 * torch.log(exp_factor), cov


class CustomLoss(torch.nn.MSELoss):
    def __init__(self, alpha):
        super().__init__()
        self.loss_record = []
        self.alpha = alpha

    def forward(self, input_data, target):
        image_loss = kl_div(target, input_data[0]).sum()
        entropy_loss = self.alpha * input_data[1]
        self.loss_record.append([image_loss, entropy_loss, input_data[2]])
        return image_loss + entropy_loss


def create_ensemble(bins, bandwidth):
    defaults = {
        "s": torch.tensor(0.0).float(),
        "p0c": torch.tensor(10.0e6).float(),
        "mc2": torch.tensor(0.511e6).float(),
    }

    transformer = NonparametricTransform(4, 50, 0.0, torch.nn.Tanh())
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))

    module_kwargs = {
        "initial_beam": InitialBeam(100000, transformer, base_dist, **defaults),
        "transport": QuadScanTransport(torch.tensor(0.1), torch.tensor(1.0), 5),
        "imager": Imager(bins, bandwidth),
        "condition": False,
        "init_weights": init_weights,
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
    all_k = torch.load(folder + "kappa.pt")[:-1, :1].float()
    all_images = torch.load(folder + "train_images.pt")[:-1, :1].float()
    xx = torch.load(folder + "xx.pt")
    bins = xx[0].T[0].float()

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

    save_dir = "alpha_1e-3_snapshot"
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

    alpha = 1e-3
    criterion = CustomLoss(alpha)
    ensemble.set_criterion(criterion)

    n_epochs = 2500
    # ensemble.set_scheduler("StepLR", gamma=0.1, step_size=200, verbose=False)
    ensemble.set_optimizer("Adam", lr=0.001)

    ensemble.fit(train_dataloader, epochs=n_epochs, save_dir=save_dir)
    torch.save(criterion.loss_record, save_dir + "/loss_log.pt")
