import torch
from torch.utils.data import DataLoader, random_split
from torchensemble import VotingRegressor

from modeling import Imager, QuadScanTransport, ImageDataset, \
    QuadScanModel, NonparametricTransformLReLU, InitialBeam


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=5.0)


def det(A):
    return A[0, 0] * A[1, 1] - A[1, 0] ** 2


class CustomLoss(torch.nn.MSELoss):
    def forward(self, input, target):
        # image_loss = mse_loss(input[0], target, reduction="sum")
        # return image_loss + 1.0 * input[1]
        eps = 1e-8
        image_loss = torch.sum(target * ((target + eps).log() - (input[0] + eps).log()))
        return image_loss + 1.0e2 * input[1]


if __name__ == "__main__":
    all_k = torch.load("../../test_case/kappa.pt")
    all_images = torch.load("../../test_case/images.pt").unsqueeze(1)
    bins = torch.load("../../test_case/bins.pt")

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
        "initial_beam": InitialBeam(100000, transformer, **defaults),
        "transport": QuadScanTransport(torch.tensor(0.1), torch.tensor(1.0)),
        "imager": Imager(bins, bandwidth),
        "condition": True
    }

    ensemble = VotingRegressor(
        estimator=QuadScanModel, estimator_args=module_kwargs, n_estimators=1, n_jobs=1
    )

    # criterion = torch.nn.MSELoss(reduction="sum")
    criterion = CustomLoss()
    ensemble.set_criterion(criterion)

    n_epochs = 300
    #ensemble.set_scheduler("CosineAnnealingLR", T_max=n_epochs)
    ensemble.set_scheduler("StepLR", gamma=0.1, step_size=200, verbose=True)
    ensemble.set_optimizer("Adam", lr=0.01, weight_decay=0.001)

    ensemble.fit(train_dataloader, epochs=n_epochs)


