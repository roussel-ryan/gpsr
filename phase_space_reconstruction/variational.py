import torch
from torch import nn


class VariationalNNTransform(torch.nn.Module):
    def __init__(
        self,
        n_hidden,
        width,
        dropout=0.0,
        activation=torch.nn.Tanh(),
        output_scale=1e-2,
    ):
        """
        Nonparametric transformation - NN
        """
        super(VariationalNNTransform, self).__init__()

        layer_sequence = [nn.Linear(6, width), activation]

        for i in range(n_hidden):
            layer_sequence.append(torch.nn.Linear(width, width))
            layer_sequence.append(torch.nn.Dropout(dropout))
            layer_sequence.append(activation)

        layer_sequence.append(torch.nn.Linear(width, 6))

        self.stack = torch.nn.Sequential(*layer_sequence)
        self.register_buffer("output_scale", torch.tensor(output_scale))

    def forward(self, X):
        return self.stack(X) * self.output_scale
