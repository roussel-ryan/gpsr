from copy import deepcopy

import torch

from phase_space_reconstruction.histogram import histogram2d
from torch import nn
from torch.utils.data import Dataset
from torch_track import Beam, TorchDrift, TorchQuad
from tqdm import trange


class NonparametricTransform(torch.nn.Module):
    def __init__(self, n_hidden, width, dropout=0.0, activation=torch.nn.Tanh()):
        """
        Nonparametric transformation - NN
        """
        super(NonparametricTransform, self).__init__()

        layer_sequence = [nn.Linear(6, width), activation]

        for i in range(n_hidden):
            layer_sequence.append(torch.nn.Linear(width, width))
            layer_sequence.append(torch.nn.Dropout(dropout))
            layer_sequence.append(activation)

        layer_sequence.append(torch.nn.Linear(width, 6))

        self.stack = torch.nn.Sequential(*layer_sequence)

    def forward(self, X):
        # scale inputs
        # X = X * 1e3
        X = self.stack(X)

        return X * 1e-2


class InitialBeam(torch.nn.Module):
    def __init__(self, n, transformer, base_dist, **kwargs):
        super(InitialBeam, self).__init__()
        self.transformer = transformer
        self.n = n
        self.kwargs = kwargs
        # dist = torch.distributions.Uniform(-torch.ones(6), torch.ones(6))
        # dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
        dist = base_dist
        base_distribution_samples = dist.sample([n])

        self.register_buffer("base_distribution_samples", base_distribution_samples)

    def forward(self, X=None):
        if X is None:
            X = self.base_distribution_samples

        tX = self.transformer(X)
        return Beam(tX, **self.kwargs)


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
    def __init__(
        self, initial_beam, transport, imager, condition=True, init_weights=None
    ):
        super(QuadScanModel, self).__init__()
        self.beam_generator = deepcopy(initial_beam)

        self.lattice = transport
        self.imager = imager

        if init_weights is not None:
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

        # scalar_metric = 0
        # calculate 6D emittance of input beam
        # emit =
        # cov = torch.cov(initial_beam.data.T)
        scalar_metric = torch.norm(initial_beam.data, dim=1).pow(2).mean()
        # scalar_metric = cov.trace()
        return output_images, scalar_metric


class MaxEntropyQuadScan(QuadScanModel):
    def forward(self, K):
        initial_beam = self.beam_generator()
        output_beam = self.lattice(initial_beam, K)
        output_coords = torch.cat(
            [output_beam.x.unsqueeze(0), output_beam.y.unsqueeze(0)]
        )
        output_images = self.imager(output_coords)

        scale = torch.tensor(1e3, device=K.device)
        cov = torch.cov(initial_beam.data.T * scale)
        exp_factor = torch.det(2 * 3.14 * 2.71 * cov)

        return output_images, 0.5 * torch.log(exp_factor), cov


def beam_loss(out_beam, target_beam):
    return torch.std(out_beam.data - target_beam * 10e-3)


def condition_initial_beam(initial_beam):
    optim = torch.optim.Adam(initial_beam.transformer.parameters(), lr=0.01)
    n_iter = 2500
    losses = []
    test_beam = initial_beam.base_distribution_samples[:1000]

    for i in trange(n_iter):
        optim.zero_grad(set_to_none=True)

        output_beam = initial_beam(test_beam)
        loss = beam_loss(output_beam, test_beam)

        losses += [loss.cpu().detach()]
        loss.backward()

        optim.step()

    # plt.plot(losses)
    # plt.show()

    # check to make sure that the initial beam is a good fit to the base distribution
    output_beam = initial_beam()
    loss = beam_loss(output_beam, initial_beam.base_distribution_samples)
    print(f"conditioning loss: {loss.cpu().detach()}")

    return initial_beam


class Imager(torch.nn.Module):
    def __init__(self, bins, bandwidth):
        super(Imager, self).__init__()
        self.register_buffer("bins", bins)
        self.register_buffer("bandwidth", bandwidth)

    def forward(self, X):
        return histogram2d(X[0], X[1], self.bins, self.bandwidth)


class InitialBeam2(torch.nn.Module):
    def __init__(self, n, n_hidden=2, width=100, **kwargs):
        super(InitialBeam2, self).__init__()
        self.n = n
        self.kwargs = kwargs
        dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
        base_distribution_samples = dist.sample([n])

        self.register_parameter(
            "base_distribution_samples", torch.nn.Parameter(base_distribution_samples)
        )

    def forward(self, X=None):
        X = self.base_distribution_samples * 1e-2
        return Beam(X, **self.kwargs)


class QuadScanTransport(torch.nn.Module):
    def __init__(self, quad_thick, drift, quad_steps=1):
        super(QuadScanTransport, self).__init__()
        # AWA
        # self.quad = TorchQuad(torch.tensor(0.12), K1=torch.tensor(0.0))
        # self.drift = TorchDrift(torch.tensor(2.84 + 0.54))

        self.quad = TorchQuad(quad_thick, K1=torch.tensor(0.0), NUM_STEPS=quad_steps)
        self.drift = TorchDrift(drift)

    def forward(self, X, K1):
        self.quad.K1 = K1.unsqueeze(-1)
        X = self.quad(X)
        X = self.drift(X)
        return X