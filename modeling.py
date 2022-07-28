import matplotlib.pyplot as plt
import torch

from histogram import histogram2d
from torch import nn
from tqdm import trange
from track import Particle
from torch_track import TorchQuad, TorchDrift, Beam


class NonparametricTransform(torch.nn.Module):
    def __init__(self, n_hidden, width):
        """
        Nonparametric transformation - NN
        """
        super(NonparametricTransform, self).__init__()

        layer_sequence = [torch.nn.Linear(6, width), torch.nn.Tanh()]

        for _ in range(n_hidden):
            layer_sequence.append(torch.nn.Linear(width, width))
            layer_sequence.append(torch.nn.Tanh())

        layer_sequence.append(torch.nn.Linear(width, 6))

        self.linear_tanh_stack = torch.nn.Sequential(
            *layer_sequence
        )

    def forward(self, X):
        # scale inputs
        X = X * 1e3
        X = self.linear_tanh_stack(X)

        return X * 1e-3


class Imager(torch.nn.Module):
    def __init__(self, bins, bandwidth):
        super(Imager, self).__init__()
        self.register_buffer("bins", bins)
        self.register_buffer("bandwidth", bandwidth)

    def forward(self, X):
        return histogram2d(X[0], X[1], self.bins, self.bandwidth)


class InitialBeam(torch.nn.Module):
    def __init__(self, n, n_hidden=2, width=100, **kwargs):
        super(InitialBeam, self).__init__()
        self.transformer = NonparametricTransform(n_hidden, width)
        self.n = n
        self.kwargs = kwargs
        dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
        base_distribution_samples = dist.sample([n])

        self.register_buffer("base_distribution_samples", base_distribution_samples)


    def forward(self, X=None):
        if X is None:
            X = self.base_distribution_samples

        tX = self.transformer(X)
        return Beam(tX, **self.kwargs)



class InitialBeam2(torch.nn.Module):
    def __init__(self, n, n_hidden=2, width=100, **kwargs):
        super(InitialBeam2, self).__init__()
        self.n = n
        self.kwargs = kwargs
        dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
        base_distribution_samples = dist.sample([n])

        self.register_parameter(
            "base_distribution_samples",
            torch.nn.Parameter(base_distribution_samples)
        )


    def forward(self, X=None):
        X = self.base_distribution_samples*1e-2
        return Beam(X, **self.kwargs)


class QuadScanTransport(torch.nn.Module):
    def __init__(self, quad_thick, drift):
        super(QuadScanTransport, self).__init__()
        # AWA
        #self.quad = TorchQuad(torch.tensor(0.12), K1=torch.tensor(0.0))
        #self.drift = TorchDrift(torch.tensor(2.84 + 0.54))

        self.quad = TorchQuad(quad_thick, K1=torch.tensor(0.0),NUM_STEPS=1)
        self.drift = TorchDrift(drift)

    def forward(self, X, K1):
        self.quad.K1 = K1.unsqueeze(-1)
        X = self.quad(X)
        X = self.drift(X)
        return X
