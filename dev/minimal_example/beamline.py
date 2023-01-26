from bmadx.bmad_torch.track_torch import TorchDrift, TorchLattice, TorchQuadrupole
import torch


def create_quad_scan_beamline():
    q1 = TorchQuadrupole(torch.tensor(0.1), torch.tensor(0.0), 5)
    d1 = TorchDrift(torch.tensor(1.0))

    lattice = TorchLattice([q1, d1])
    return lattice
