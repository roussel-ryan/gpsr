import logging

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, random_split, Subset
from torchensemble import VotingRegressor, SnapshotEnsembleRegressor

import sys

sys.path.append("../")

from modeling import Imager, QuadScanTransport, ImageDataset, \
    QuadScanModel, InitialBeam, \
    NonparametricTransform

logging.basicConfig(level=logging.INFO)


class MaxEntropyQuadScan(QuadScanModel):
    def forward(self, K):
        initial_beam = self.beam_generator()
        output_beam = self.lattice(initial_beam, K)
        output_coords = torch.cat(
            [output_beam.x.unsqueeze(0), output_beam.y.unsqueeze(0)]
        )
        output_images = self.imager(output_coords)

        cov = torch.cov(initial_beam.data.T) + torch.eye(6,
                                                         device=initial_beam.data.device) * 1e-8
        exp_factor = torch.det(2 * 3.14 * 2.71 * cov)

        return output_images, -0.5 * torch.log(exp_factor), cov


def kl_div(target, pred):
    eps = 1e-8
    return target * torch.abs((target + eps).log() - (pred + eps).log())


def log_squared_error(target, pred):
    eps = 1e-8
    return target * ((target + eps).log() - (pred + eps).log()) ** 2


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
        "p0c": torch.tensor(65.0e6).float(),
        "mc2": torch.tensor(0.511e6).float(),
    }

    transformer = NonparametricTransform(4, 50, 0.0, torch.nn.Tanh())
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))

    module_kwargs = {
        "initial_beam": InitialBeam(100000, transformer, base_dist, **defaults),
        "transport": QuadScanTransport(torch.tensor(0.12), torch.tensor(2.84 + 0.54),
                                       5),
        "imager": Imager(bins, bandwidth),
        "condition": False
    }

    ensemble = VotingRegressor(
        estimator=MaxEntropyQuadScan,
        estimator_args=module_kwargs,
        n_estimators=2
    )
    return ensemble


def get_data(folder):
    all_k = torch.load(folder + "kappa.pt")
    all_images = torch.load(folder + "train_images.pt")
    xx = torch.load(folder + "xx.pt")
    bins = xx[0].T[0]

    n_samples = 2
    all_k = all_k.cuda()[:, :n_samples]
    all_images = all_images.cuda()[:, :n_samples]

    return all_k, all_images, bins, xx


def get_datasets(all_k, all_images):
    train_dset = ImageDataset(all_k[::2], all_images[::2])
    test_dset = ImageDataset(all_k[1::2], all_images[1::2])
    torch.save(train_dset, "train.dset")
    torch.save(test_dset, "test.dset")

    return train_dset, test_dset


if __name__ == "__main__":
    folder = ""
    all_k, all_images, bins, xx = get_data(folder)
    train_dset, test_dset = get_datasets(all_k, all_images)

    train_dataloader = DataLoader(train_dset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2
    ensemble = create_ensemble(bins, bandwidth)

    alpha = 1e-3
    criterion = CustomLoss(alpha)
    ensemble.set_criterion(criterion)

    n_epochs = 10
    ensemble.set_scheduler("StepLR", gamma=0.1, step_size=200, verbose=False)
    ensemble.set_optimizer("Adam", lr=0.001)

    save_dir = "alpha_1e-3"
    ensemble.fit(train_dataloader, epochs=n_epochs, save_dir=save_dir)
    torch.save(criterion.loss_record, save_dir + "/loss_log.pt")
