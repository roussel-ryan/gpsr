import torch
from bmadx import track
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from utils import get_slopes


class Beam(torch.nn.Module):
    def __init__(self, data, p0c: float, s: float = 0.0, mc2: float = 0.511e6):
        super(Beam, self).__init__()
        self.keys = ["x", "px", "y", "py", "z", "pz"]
        self.data = data

        for i, key in enumerate(self.keys):
            self.register_parameter(key, Parameter(data[..., i], requires_grad=False))

        self.register_parameter(
            "p0c", Parameter(torch.tensor(p0c), requires_grad=False)
        )
        self.register_parameter("s", Parameter(torch.tensor(s), requires_grad=False))
        self.register_parameter(
            "mc2", Parameter(torch.tensor(mc2), requires_grad=False)
        )

    def to_list_of_beams(self):
        beams = []
        for i in range(len(getattr(self, self.keys[0]))):
            beams += [
                track.Particle(
                    *[getattr(self, key)[i] for key in self.keys], **self._defaults
                )
            ]

        return beams

    @property
    def xp(self):
        return get_slopes(self.px, self.py, self.pz)[0]

    @property
    def yp(self):
        return get_slopes(self.px, self.py, self.pz)[1]


class TorchQuadrupole(Module):
    def __init__(
        self,
        L: Tensor,
        K1: Tensor,
        NUM_STEPS: int = 1,
        X_OFFSET: Tensor = torch.tensor(0.0),
        Y_OFFSET: Tensor = torch.tensor(0.0),
        TILT: Tensor = torch.tensor(0.0),
    ):
        super(TorchQuadrupole, self).__init__()
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_parameter(
            "NUM_STEPS", Parameter(torch.tensor(NUM_STEPS), requires_grad=False)
        )
        self.register_parameter("TILT", Parameter(TILT, requires_grad=False))

        self.K1 = K1
        self.track = track.make_track_a_quadrupole(torch)

    def forward(self, X):
        return self.track(X, self)


class TorchDrift(Module):
    def __init__(
        self,
        L: Tensor,
    ):
        super(TorchDrift, self).__init__()
        self.register_parameter("L", Parameter(L, requires_grad=False))

        # create function
        self.track = track.make_track_a_drift(torch)

    def forward(self, X):
        return self.track(X, self)


class TorchLattice(Module):
    def __init__(self, elements, only_last=True):
        super(TorchLattice, self).__init__()
        self.elements = ModuleList(elements)
        self.only_last = only_last

    def forward(self, p_in):
        if self.only_last:
            p = p_in
            for i in range(self.n_elements):
                p = self.elements[i](p)

            return p
        else:
            all_p = [None] * (self.n_elements + 1)
            all_p[0] = p_in

            for i in range(self.n_elements):
                all_p[i + 1] = self.elements[i](all_p[i])

            return all_p

    @property
    def n_elements(self):
        return len(self.elements)
