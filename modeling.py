import torch

from histogram import histogram2d
from torch import nn
from torch.distributions import MultivariateNormal, Uniform
from tqdm import trange
from track import Particle


class NonparametricTransform(torch.nn.Module):
    def __init__(self):
        """
        Nonparametric transformation - NN
        """
        super(NonparametricTransform, self).__init__()
        width = 100

        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(6, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 6),
        )

    def forward(self, X):
        # scale inputs
        X = X * 1e3
        X = self.linear_tanh_stack(X)

        return X * 1e-3


class ImagingModel(nn.Module):
    def __init__(self, transformer, bins, bandwidth, n_particles, defaults):
        super(ImagingModel, self).__init__()
        self.register_buffer("bins", bins)
        self.register_buffer("bandwidth", bandwidth)
        self.register_buffer("n_particles", n_particles)

        for name, val in defaults.items():
            self.register_buffer(name, val)

        self.transformer = transformer

        self.register_buffer(
            "normal_samples", self.generate_normal_distribution(self.n_particles)
        )

    def forward(self, lattice):
        # calculate the initial distribution
        guess_dist = self.get_initial_beam()

        # propagate beam
        output_beams = lattice(guess_dist)[-1]
        images = histogram2d(output_beams.x, output_beams.y, self.bins, self.bandwidth)

        return images

    def get_initial_beam(self, n=-1):
        if n < 0:
            return Particle(*self.transformer(self.normal_samples).T, **self.defaults)
        else:
            return Particle(
                *self.transformer(self.generate_normal_distribution(n)).T,
                **self.defaults
            )

    def generate_normal_distribution(self, n):
        # create normalized distribution
        #normal_dist = Uniform(
        #    -torch.ones(6), torch.ones(6)
        #)
        normal_dist = MultivariateNormal(torch.zeros(6), torch.eye(6))
        return normal_dist.sample([n]).to(**self.tkwargs())

    def tkwargs(self):
        return {
            "device": next(self.parameters()).device,
            "dtype": next(self.parameters()).dtype,
        }

    @property
    def defaults(self):
        return {"s": self.s, "p0c": self.p0c, "mc2": self.mc2}


def train(model, train_lattice, train_images, n_iter, lr=0.001):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for _ in trange(n_iter):
        optim.zero_grad(set_to_none=True)
        loss_function = nn.MSELoss(reduction="sum")
        loss = loss_function(model(train_lattice), train_images)

        losses += [loss.cpu().detach()]
        loss.backward()

        optim.step()

    return losses
