import torch
from bmadx.bmad_torch.track_torch import (
    TorchCrabCavity,
    TorchDrift,
    TorchLattice,
    TorchQuadrupole,
    Beam
)
from bmadx.bmad_torch.utils import get_transport_matrix
from diagnostics import ImageDiagnostic
from matplotlib import pyplot as plt

bins = torch.linspace(-25, 25, 100) * 1e-3
screen = ImageDiagnostic(bins)


def define_beamline(k, k_tdc):
    s1 = TorchQuadrupole(
        torch.tensor(0.1), torch.tensor(1.0), TILT=torch.tensor(3.14 / 4)
    )
    d0 = TorchDrift(torch.tensor(0.5))

    q1 = TorchQuadrupole(torch.tensor(0.1), k)
    d1 = TorchDrift(torch.tensor(0.5))

    tdc1 = TorchCrabCavity(torch.tensor(0.3), k_tdc, torch.tensor(1.3e9))
    d2 = TorchDrift(torch.tensor(1.0))

    lattice = TorchLattice([s1, d0, q1, d1, tdc1, d2])

    return lattice


def create_test_beam(N=1000):
    beam_cov = torch.eye(6) * 1e-6
    particles = torch.distributions.MultivariateNormal(torch.zeros(6), beam_cov).sample(
        [N]
    )
    p0c = 10.0e6
    beam = Beam(particles, p0c)

    return beam


def generate_test_images(lattice):
    beam = create_test_beam()
    out_beam = lattice(beam)

    images = screen.calculate_images(out_beam.x, out_beam.y)
    return images


if __name__ == "__main__":
    k = torch.linspace(-10, 20, 3).unsqueeze(-1)
    k_tdc = torch.tensor(1e6)
    lattice = define_beamline(k, k_tdc)

    test_images = generate_test_images(lattice)

    for ele in test_images:
        fig,ax = plt.subplots()
        ax.pcolor(*screen.mesh, ele)

    plt.show()
