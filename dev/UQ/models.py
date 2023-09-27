from copy import deepcopy

import torch
from torch import nn
from torch.nn.functional import mse_loss


class IndependentVariationalNN(torch.nn.Module):
    def __init__(self, n_outputs=1, n_samples=1, output_scale=1):
        super(IndependentVariationalNN, self).__init__()

        # define architecture
        n_common_layers = 2
        width = 20
        activation = nn.Tanh()
        self.n_samples = n_samples
        self.output_scale = output_scale

        # create input layer
        layers = [nn.Linear(n_outputs, width), activation]

        # create common layers
        for _ in range(n_common_layers):
            layers.append(nn.Linear(width, width))
            layers.append(activation)

        layers.append(nn.Linear(width, n_outputs))

        self.mean = nn.Sequential(*deepcopy(layers))
        self.rho = nn.Sequential(*deepcopy(layers))

    def std(self, X):
        return torch.log1p(torch.exp(self.rho(X))) * self.output_scale

    def forward(self, X):
        mean = self.mean(X)
        std = self.std(X)

        # sample
        eps = torch.randn((self.n_samples, *X.shape)).to(X)

        return mean * self.output_scale + std * eps


class CoupledVariationalNN(torch.nn.Module):
    def __init__(self):
        super(CoupledVariationalNN, self).__init__()

        # define architecture
        n_common_layers = 2
        n_local_layers = 2
        width = 10
        activation = nn.Tanh()

        # create input layer
        layers = [nn.Linear(1, width), activation]

        # create common layers
        for _ in range(n_common_layers):
            layers.append(nn.Linear(width, width))
            layers.append(activation)

        common = nn.Sequential(*layers)

        mean_layers = []
        rho_layers = []
        for _ in range(n_local_layers):
            mean_layers.append(nn.Linear(width, width))
            mean_layers.append(activation)

            rho_layers.append(nn.Linear(width, width))
            rho_layers.append(activation)

        # outputs
        mean_layers.append(nn.Linear(width, 1))
        rho_layers.append(nn.Linear(width, 1))

        self.mean = nn.Sequential(common, *mean_layers)
        self.rho = nn.Sequential(common, *rho_layers)

    def std(self, X):
        return torch.log1p(torch.exp(self.rho(X)))

    def forward(self, X):
        mean = self.mean(X)
        std = self.std(X)

        # sample unit normal
        eps = torch.randn_like(X)

        return mean + std * eps


def precondition(model, train_x, n_iter=500):
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(n_iter):
        optim.zero_grad()

        # calculate loss
        loss = mse_loss(model(train_x), train_x)

        loss.backward()

        if i % 100 == 0:
            print(f"precondition_loss: {loss}")

        optim.step()
    return model
