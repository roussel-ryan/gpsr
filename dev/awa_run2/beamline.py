import torch
from bmadx.bmad_torch.track_torch import (
    TorchCrabCavity,
    TorchDrift,
    TorchLattice,
    TorchQuadrupole,
)


def create_quad_scan_beamline():
    q1 = TorchQuadrupole(torch.tensor(0.12), torch.tensor(0.0), 5)
    d1 = TorchDrift(torch.tensor(3.38))

    lattice = TorchLattice([q1, d1])
    return lattice
