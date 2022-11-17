import torch
from torch import nn


class NNBeam(nn.Module):
    def __init__(
        self,
        n_hidden,
        width,
        dropout=0.0,
        activation=torch.nn.Tanh(),
        output_scale=1e-2,
    ):
        super(NNBeam, self).__init__()

        # create input layer
        layers = [nn.Linear(6, width), activation]

        # create hidden layers
        for _ in range(n_hidden):
            layers.append(nn.Linear(width, width))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(activation)

        layers.append(nn.Linear(width, 6))

        self.stack = nn.Sequential(*layers)
        self.register_buffer("output_scale", torch.tensor(output_scale))

    def forward(self, X):
        return self.stack(X) * self.output_scale
