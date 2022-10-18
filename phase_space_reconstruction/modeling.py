import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from bmadx.bmad_torch.track_torch import Beam, TorchDrift, TorchQuadrupole
from bmadx.track import Particle

from .histogram import histogram2d
from torch import nn
from torch.utils.data import Dataset
from tqdm import trange


class NNTransform(torch.nn.Module):
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
        super(NNTransform, self).__init__()

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


class InitialBeam(torch.nn.Module):
    def __init__(self, transformer, base_dist):
        super(InitialBeam, self).__init__()
        self.transformer = transformer
        self.base_beam = base_dist

    def forward(self):
        transformed_beam = self.transformer(self.base_beam.data)
        return Beam(
            transformed_beam, self.base_beam.p0c, self.base_beam.s, self.base_beam.mc2
        )

    def get_entropy(self, beam):
        # note: multiply and divide by 1e3 to help underflow issues
        emit = (torch.det(torch.cov(beam.data.T*1e3)) * 10**-(3*6)) ** 0.5
        #emit = (torch.det(torch.cov(beam.data.T*1e3)) * 10**-(3*6)) ** 0.5
        return torch.log(
            (2 * 3.14 * 2.71) ** 3 * emit
        )


# create data loader
class ImageDataset(Dataset):
    def __init__(self, k, images):
        self.images = images
        self.k = k

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.k[idx], self.images[idx]


def predict_images(beam, lattice, screen):
    out_beam = lattice(beam)
    images = screen.calculate_images(out_beam.x, out_beam.y)
    return images
