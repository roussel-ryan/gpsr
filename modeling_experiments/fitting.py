from copy import deepcopy

import torch

from modeling import Imager, InitialBeam, InitialBeam2, QuadScanTransport
from torch.nn.functional import kl_div, mse_loss
from torch.utils.data import DataLoader, Dataset, random_split
from torchensemble import VotingRegressor
from experimental import ExperimentalInitialBeam, NonparametricTransformLReLU, \
    condition_initial_beam


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=5.0)


# create data loader
class ImageDataset(Dataset):
    def __init__(self, k, images):
        self.images = images
        self.k = k

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.k[idx], self.images[idx]


class QuadScanModel(torch.nn.Module):
    def __init__(self, initial_beam, transport, imager, condition=True):
        super(QuadScanModel, self).__init__()
        self.beam_generator = deepcopy(initial_beam)

        self.lattice = transport
        self.imager = imager

        self.beam_generator.apply(init_weights)

        # condition initial beam
        if condition:
            self.beam_generator = condition_initial_beam(self.beam_generator)

    def forward(self, K):
        initial_beam = self.beam_generator()
        output_beam = self.lattice(initial_beam, K)
        output_coords = torch.cat(
            [output_beam.x.unsqueeze(0), output_beam.y.unsqueeze(0)]
        )
        output_images = self.imager(output_coords)
        # return output_images

        scalar_metric = 0
        # calculate 6D emittance of input beam
        # emit =
        cov = torch.cov(initial_beam.data.T)
        scalar_metric = torch.norm(initial_beam.data, dim=1).pow(2).mean()
        #scalar_metric = cov.trace()
        return output_images, scalar_metric


def det(A):
    return A[0, 0] * A[1, 1] - A[1, 0] ** 2


class CustomLoss(torch.nn.MSELoss):
    def forward(self, input, target):
        # image_loss = mse_loss(input[0], target, reduction="sum")
        # return image_loss + 1.0 * input[1]
        eps = 1e-8
        image_loss = torch.sum(target * ((target + eps).log() - (input[0] + eps).log()))
        return image_loss# + 1.0e2 * input[1]


if __name__ == "__main__":
    all_k = torch.load("../test_case2/kappa.pt")
    all_images = torch.load("../test_case2/images.pt").unsqueeze(1)
    bins = torch.load("../test_case2/bins.pt")

    #bins = (bins[:-1] + bins[1:]) / 2

    all_k = all_k.cuda()
    all_images = all_images.cuda()

    print(all_k.shape)
    print(all_images.shape)

    train_dset, test_dset = random_split(ImageDataset(all_k, all_images), [16, 4])
    train_dataloader = DataLoader(train_dset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dset)

    torch.save(train_dset, "train.dset")
    torch.save(test_dset, "test.dset")

    bin_width = bins[1] - bins[0]
    print(bin_width)
    bandwidth = bin_width

    defaults = {
        "s": torch.tensor(0.0).float(),
        "p0c": torch.tensor(10.0e6).float(),
        "mc2": torch.tensor(0.511e6).float(),
    }

    transformer = NonparametricTransformLReLU(2, 20)

    module_kwargs = {
        "initial_beam": ExperimentalInitialBeam(100000, transformer, **defaults),
        "transport": QuadScanTransport(torch.tensor(0.1), torch.tensor(1.0)),
        "imager": Imager(bins, bandwidth),
    }

    ensemble = VotingRegressor(
        estimator=QuadScanModel, estimator_args=module_kwargs, n_estimators=1, n_jobs=1
    )

    # criterion = torch.nn.MSELoss(reduction="sum")
    criterion = CustomLoss()
    ensemble.set_criterion(criterion)

    n_epochs = 200
    #ensemble.set_scheduler("CosineAnnealingLR", T_max=n_epochs)
    #ensemble.set_scheduler("StepLR", gamma=0.01, step_size=100, verbose=True)
    ensemble.set_optimizer("Adam", lr=0.01, weight_decay=0.0)

    ensemble.fit(train_dataloader, epochs=n_epochs)


