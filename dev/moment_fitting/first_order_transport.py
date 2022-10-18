import torch
from bmadx.bmad_torch.track_torch import (
    TorchCrabCavity,
    TorchDrift,
    TorchLattice,
    TorchQuadrupole,
)
from bmadx.bmad_torch.utils import get_transport_matrix
from matplotlib import pyplot as plt


def define_beamline(k, k_tdc):
    s1 = TorchQuadrupole(
        torch.tensor(0.1), torch.tensor(1.0), TILT=torch.tensor(3.14 / 4)
    )
    d0 = TorchDrift(torch.tensor(0.5))

    q1 = TorchQuadrupole(torch.tensor(0.1), k)
    d1 = TorchDrift(torch.tensor(0.5))

    tdc1 = TorchCrabCavity(torch.tensor(0.23), k_tdc, torch.tensor(1.3e9))
    d2 = TorchDrift(torch.tensor(1.0))

    lattice = TorchLattice([s1, d0, q1, d1, tdc1, d2])

    return lattice


if __name__ == "__main__":
    kq = torch.linspace(-10, 20, 25).unsqueeze(-1)
    k_tdc = torch.linspace(-10, 10, 25).unsqueeze(0) * 1e6
    test_lattice = define_beamline(kq, k_tdc)
    print(test_lattice.batch_shape)
    M = get_transport_matrix(
        test_lattice, torch.tensor(0.0), torch.tensor(10.0e6), torch.tensor(0.511e6)
    )

    sigma_i = torch.eye(6) * 1e-6
    sigma_f = torch.transpose(M, -2, -1) @ sigma_i @ M

    print(sigma_f.shape)

    inputs = torch.broadcast_tensors(kq, k_tdc)

    for ele in [[0, 0], [0, 2], [2, 2]]:
        fig, ax = plt.subplots()
        c = ax.pcolor(
            inputs[0], inputs[1], sigma_f.select(-2, ele[0])[..., ele[1]]
        )
        fig.colorbar(c)
    # ax.plot(k, sigma_f[:, 0, 2])
    # ax.plot(k, sigma_f[:, 2, 2].sqrt())
    plt.show()
