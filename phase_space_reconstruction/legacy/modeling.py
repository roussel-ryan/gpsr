from copy import deepcopy

import torch
from torch.utils.data import Dataset  # def set_dipole_G(self, G):

from bmadx.bmad_torch.track_torch import Beam


#    theta = torch.arcsin(self.l_bend * G)
    #    l_arc = theta / G
    #    self.lattice.elements[-2].G.data = G
    #    self.lattice.elements[-2].L.data = l_arc
    #    self.lattice.elements[-2].E2.data = theta
    #    self.lattice.elements[-1].L.data = self.l3 - self.l_bend / 2 / torch.cos(theta)


# class VariationalPhaseSpaceReconstructionModel(PhaseSpaceReconstructionModel):
#    def forward(self, K, scan_quad_id=0):
#        proposal_beam = self.beam()

# track beam
#        observations, _ = self.track_and_observe_beam(proposal_beam, K, scan_quad_id)

#        return observations


class FixedBeam(torch.nn.Module):
    def __init__(self, beam):
        super(FixedBeam, self).__init__()
        self.beam = beam

    def forward(self):
        return self.beam


class OffsetBeam(torch.nn.Module):
    """
    Define a beam distribution that has learnable parameters for centroid offset for
    each shot to represent jitter in the beam centroid. This implies several assumptions
    - the beam profile is independent of the beam centroid location, this ignores
    higher order changes in the beam distribution -- this could eventually be
    replaced by a more complex transformer

    This should provide more detail in the reconstruction

    """

    def __init__(self, offset, base_beam):
        super(OffsetBeam, self).__init__()
        self.offset = offset
        self.base_beam = base_beam

    def forward(self):
        transformed_beam = self.base_beam().data + self.offset
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


class NormalizedQuadScan(torch.nn.Module):
    def __init__(self, A, drift, quad_length):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("d", drift)
        self.register_buffer("l", quad_length)

        # note params are normalized
        self.register_parameter("lambda_1", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("lambda_2", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("c", torch.nn.Parameter(torch.tensor(0.0)))

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
        norm_emit = (self.lambda_1 ** 2 * self.lambda_2 ** 2 * (1 - norm_c ** 2)).sqrt()
        return norm_emit * self.A ** 2


def predict_images(beam, lattice, screen):
    out_beam = lattice(beam)
    images = screen.calculate_images(out_beam.x, out_beam.y)
    return images


class SextPhaseSpaceReconstructionModel(torch.nn.Module):
    def __init__(self, lattice, diagnostic, beam):
        super(SextPhaseSpaceReconstructionModel, self).__init__()

        self.base_lattice = lattice
        self.diagnostic = diagnostic
        self.beam = deepcopy(beam)

    def track_and_observe_beam(self, beam, K2, scan_quad_id=0):
        lattice = deepcopy(self.base_lattice)
        lattice.elements[0].K2.data = K2

        # track beam through lattice
        final_beam = lattice(beam)

        # analyze beam with diagnostic
        observations = self.diagnostic(final_beam)

        return observations, final_beam

    def forward(self, params, ids):
        proposal_beam = self.beam()

        # track beam
        observations, final_beam = self.track_and_observe_beam(
            proposal_beam, params, ids
        )

        # get entropy
        entropy = calculate_beam_entropy(proposal_beam)

        # get beam covariance
        cov = calculate_covariance(proposal_beam)

        return observations, entropy, cov


class PhaseSpaceReconstructionModel3D(torch.nn.Module):
    def __init__(self, lattice, diagnostic, beam):
        super(PhaseSpaceReconstructionModel3D, self).__init__()

        self.base_lattice = lattice
        self.diagnostic = diagnostic
        self.beam = deepcopy(beam)

    def track_and_observe_beam(self, beam, params, ids):
        lattice = deepcopy(self.base_lattice)
        lattice.elements[ids[0]].K1.data = params[:, 0].unsqueeze(-1)
        lattice.elements[ids[1]].VOLTAGE.data = params[:, 1].unsqueeze(-1)
        # change the dipole attributes + drift attribute
        G = params[:, 2].unsqueeze(-1)
        l_bend = 0.3018
        theta = torch.arcsin(l_bend * G)  # AWA parameters
        l_arc = theta / G
        lattice.elements[ids[2]].G.data = G
        lattice.elements[ids[2]].L.data = l_arc
        lattice.elements[ids[2]].E2.data = theta
        lattice.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

        # track beam through lattice
        final_beam = lattice(beam)

        # analyze beam with diagnostic
        observations = self.diagnostic(final_beam)

        return observations, final_beam

    def forward(self, params, ids):
        proposal_beam = self.beam()

        # track beam
        observations, final_beam = self.track_and_observe_beam(
            proposal_beam, params, ids
        )

        # get entropy
        entropy = calculate_beam_entropy(proposal_beam)

        # get beam covariance
        cov = calculate_covariance(proposal_beam)

        return observations, entropy, cov


class ImageDataset3D(Dataset):
    def __init__(self, params, images):
        self.params = params
        self.images = images

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.images[idx]


#### 2 screen adaptation: ####


class PhaseSpaceReconstructionModel3D_2screens(torch.nn.Module):
    def __init__(self, lattice0, lattice1, diagnostic0, diagnostic1, beam):
        super(PhaseSpaceReconstructionModel3D_2screens, self).__init__()

        self.lattice0 = lattice0
        self.lattice1 = lattice1
        self.diagnostic0 = diagnostic0
        self.diagnostic1 = diagnostic1
        self.beam = deepcopy(beam)

    def track_and_observe_beam(self, beam, params, n_imgs_per_param, ids):
        params_dipole_off = params[:, :, 0].unsqueeze(-1)
        diagnostics_lattice0 = self.lattice0.copy()
        diagnostics_lattice0.elements[ids[0]].K1.data = params_dipole_off[:, :, 0]
        diagnostics_lattice0.elements[ids[1]].VOLTAGE.data = params_dipole_off[:, :, 1]
        # change the dipole attributes + drift attribute
        G = params_dipole_off[:, :, 2]
        l_bend = 0.3018
        theta = torch.arcsin(l_bend * G)  # AWA parameters
        l_arc = theta / G
        diagnostics_lattice0.elements[ids[2]].G.data = G
        diagnostics_lattice0.elements[ids[2]].L.data = l_arc
        diagnostics_lattice0.elements[ids[2]].E2.data = theta
        diagnostics_lattice0.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

        params_dipole_on = params[:, :, 1].unsqueeze(-1)
        diagnostics_lattice1 = self.lattice1.copy()
        diagnostics_lattice1.elements[ids[0]].K1.data = params_dipole_on[:, :, 0]
        diagnostics_lattice1.elements[ids[1]].VOLTAGE.data = params_dipole_on[:, :, 1]
        # change the dipole attributes + drift attribute
        G = params_dipole_on[:, :, 2]
        l_bend = 0.3018
        theta = torch.arcsin(l_bend * G)  # AWA parameters
        l_arc = theta / G
        diagnostics_lattice1.elements[ids[2]].G.data = G
        diagnostics_lattice1.elements[ids[2]].L.data = l_arc
        diagnostics_lattice1.elements[ids[2]].E2.data = theta
        diagnostics_lattice1.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

        # track through lattice for dipole off(0) and dipole on (1)
        output_beam0 = diagnostics_lattice0(beam)
        output_beam1 = diagnostics_lattice1(beam)

        # histograms at screens for dipole off(0) and dipole on (1)
        images_dipole_off = self.diagnostic0(output_beam0).squeeze()
        images_dipole_on = self.diagnostic1(output_beam1).squeeze()

        # stack on dipole dimension:
        images_stack = torch.stack((images_dipole_off, images_dipole_on), dim=2)

        # create images copies simulating multi-shot per parameter config:
        copied_images = torch.stack([images_stack] * n_imgs_per_param, dim=-3)

        return copied_images

    def forward(self, params, n_imgs_per_param, ids):
        proposal_beam = self.beam()

        # track beam
        observations = self.track_and_observe_beam(
            proposal_beam, params, n_imgs_per_param, ids
        )

        # get entropy
        entropy = calculate_beam_entropy(proposal_beam)

        # get beam covariance
        cov = calculate_covariance(proposal_beam)

        return observations, entropy, cov