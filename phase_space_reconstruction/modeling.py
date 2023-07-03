import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from bmadx.bmad_torch.track_torch import Beam, TorchDrift, TorchQuadrupole
from bmadx import Particle
from torch import nn
from torch.utils.data import Dataset
from tqdm import trange


class PhaseSpaceReconstructionModel(torch.nn.Module):
    def __init__(self, lattice, diagnostic, beam):
        super(PhaseSpaceReconstructionModel, self).__init__()

        self.base_lattice = lattice
        self.diagnostic = diagnostic
        self.beam = deepcopy(beam)

    def track_and_observe_beam(self, beam, K):
        # alter quadrupole strength
        lattice = deepcopy(self.base_lattice)
        lattice.elements[0].K1.data = K

        # track beam through lattice
        final_beam = lattice(beam)

        # analyze beam with diagnostic
        observations = self.diagnostic(final_beam)

        return observations, final_beam

    def forward(self, K):
        proposal_beam = self.beam()

        # track beam
        observations, final_beam = self.track_and_observe_beam(proposal_beam, K)

        # get entropy
        entropy = calculate_beam_entropy(proposal_beam)

        # get beam covariance
        cov = calculate_covariance(proposal_beam)

        return observations, entropy, cov


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
    def __init__(self, transformer, base_dist, n_particles, **kwargs):
        super(InitialBeam, self).__init__()
        self.transformer = transformer
        self.base_dist = base_dist
        self.base_beam = None

        self.set_base_beam(base_dist, n_particles, **kwargs)

    def set_base_beam(self, base_dist, n_particles, **kwargs):
        self.base_beam = Beam(base_dist.sample([n_particles]), **kwargs)

    def forward(self):
        transformed_beam = self.transformer(self.base_beam.data)
        return Beam(
            transformed_beam, self.base_beam.p0c, self.base_beam.s, self.base_beam.mc2
        )


def calculate_covariance(beam):
    # note: multiply and divide by 1e3 to help underflow issues
    return torch.cov(beam.data.T * 1e3) * 1e-6


def calculate_entropy(cov):
    emit = (torch.det(cov * 1e9)) ** 0.5 * 1e-27
    return torch.log((2 * 3.14 * 2.71) ** 3 * emit)


def calculate_beam_entropy(beam):
    return calculate_entropy(calculate_covariance(beam))


# create data loader
class ImageDataset(Dataset):
    def __init__(self, k, images):
        self.images = images
        self.k = k

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.k[idx], self.images[idx]


class NormalizedQuadScan(nn.Module):
    def __init__(self, A, drift, quad_length):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("d", drift)
        self.register_buffer("l", quad_length)

        # note params are normalized
        self.register_parameter("lambda_1", nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("lambda_2", nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("c", nn.Parameter(torch.tensor(0.0)))

    def forward(self, k):
        # input should be real k, output is real sigma_x^2
        norm_k = 1 + self.d * self.l * k
        norm_c = torch.tanh(self.c)
        norm_s11 = (
            norm_k ** 2 * self.lambda_1 ** 2
            + 2 * self.d * norm_k * self.lambda_1 * self.lambda_2 * norm_c
            + self.lambda_2 ** 2 * self.d ** 2
        )
        return norm_s11 * self.A ** 2

    def emittance(self):
        norm_c = torch.tanh(self.c)
        norm_emit = (self.lambda_1**2 * self.lambda_2**2 * (1 - norm_c**2)).sqrt()
        return norm_emit * self.A ** 2


def predict_images(beam, lattice, screen):
    out_beam = lattice(beam)
    images = screen.calculate_images(out_beam.x, out_beam.y)
    return images
