import logging
import os

import sys

import torch
from torch.autograd import grad
from torch.nn import Parameter
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, random_split, Subset

from utils import kl_div

sys.path.append("../../")

from modeling import (
    ImageDataset,
    Imager,
    InitialBeam,
    MaxEntropyQuadScan,
    NonparametricTransform,
    QuadScanTransport,
)

logging.basicConfig(level=logging.INFO)


class GradientSquaredLoss(torch.nn.MSELoss):
    def __init__(self, l0, model):
        super().__init__()
        self.loss_record = []
        self.register_parameter("lambda_", Parameter(l0))
        self.model = model

    def forward(self, input_data, target):
        image_loss = kl_div(target, input_data[0]).sum()
        entropy_loss = input_data[1]

        # attempt to maximize the entropy loss while constraining on the image loss
        # using lagrange multipliers see:
        # https://en.wikipedia.org/wiki/Lagrange_multiplier

        unconstrained_loss = entropy_loss + self.lambda_ * image_loss
        z = grad(unconstrained_loss, self.model.parameters(), create_graph=True)
        grad_loss = torch.norm(torch.cat([ele.unsqueeze(0) for ele in z]))

        return grad_loss


def construct_optimizer(bins, bandwidth):
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
    }
    model = MaxEntropyQuadScan(**module_kwargs)
    loss_module = GradientSquaredLoss(torch.tensor(100.0), model)
    optim = torch.optim.Adam(list(model.parameters()) + [loss_module.lambda_], lr=0.001)

    return model, loss_module, optim


def get_data(folder):
    all_k = torch.load(folder + "kappa.pt")
    all_images = torch.load(folder + "train_images.pt")
    xx = torch.load(folder + "xx.pt")
    bins = xx[0].T[0]

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
    folder = "../../test_case_4/"

    save_dir = "alpha_1e-3_snapshot_lr_01"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    all_k, all_images, bins, xx = get_data(folder)
    train_dset, test_dset = get_datasets(all_k, all_images, save_dir)
    print(len(train_dset))

    train_dataloader = DataLoader(train_dset, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2
    model, loss_function, optim = construct_optimizer(bins, bandwidth)

    for ii in range(1000):
        # get predictions from the model
        model_outputs = model()
        grad_loss = loss_function()


