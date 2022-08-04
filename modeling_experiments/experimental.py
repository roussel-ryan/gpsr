import torch
from matplotlib import pyplot as plt
from tqdm import trange

from torch_track import Beam


class NonparametricTransformLReLU(torch.nn.Module):
    def __init__(self, n_hidden, width):
        """
        Nonparametric transformation - NN
        """
        super(NonparametricTransformLReLU, self).__init__()

        layer_sequence = [torch.nn.Linear(6, width), torch.nn.LeakyReLU()]

        for _ in range(n_hidden):
            layer_sequence.append(torch.nn.Linear(width, width))
            layer_sequence.append(torch.nn.Tanh())

        layer_sequence.append(torch.nn.Linear(width, 6))

        self.stack = torch.nn.Sequential(*layer_sequence)

    def forward(self, X):
        # scale inputs
        # X = X * 1e3
        X = self.stack(X)

        return X * 1e-3


class NonparametricTransformTanh(torch.nn.Module):
    def __init__(self, n_hidden, width):
        """
        Nonparametric transformation - NN
        """
        super(NonparametricTransformTanh, self).__init__()

        layer_sequence = [torch.nn.Linear(6, width), torch.nn.Tanh()]

        for _ in range(n_hidden):
            layer_sequence.append(torch.nn.Linear(width, width))
            layer_sequence.append(torch.nn.Tanh())

        layer_sequence.append(torch.nn.Linear(width, 6))

        self.stack = torch.nn.Sequential(*layer_sequence)

    def forward(self, X):
        # scale inputs
        # X = X * 1e3
        X = self.stack(X)

        return X * 1e-3


class ExperimentalInitialBeam(torch.nn.Module):
    def __init__(self, n, transformer, **kwargs):
        super(ExperimentalInitialBeam, self).__init__()
        self.transformer = transformer
        self.n = n
        self.kwargs = kwargs
        # dist = torch.distributions.Uniform(-torch.ones(6), torch.ones(6))
        dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
        base_distribution_samples = dist.sample([n])

        self.register_buffer("base_distribution_samples", base_distribution_samples)

    def forward(self, X=None):
        if X is None:
            X = self.base_distribution_samples

        tX = self.transformer(X)
        return Beam(tX, **self.kwargs)


def beam_loss(out_beam, target_beam):
    return torch.std(out_beam.data - target_beam * 1e-3)


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

    #plt.plot(losses)
    #plt.show()

    # check to make sure that the initial beam is a good fit to the base distribution
    output_beam = initial_beam()
    loss = beam_loss(output_beam, initial_beam.base_distribution_samples)
    print(f"conditioning loss: {loss.cpu().detach()}")

    return initial_beam
