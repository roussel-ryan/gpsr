import torch
from torch import Tensor
from torch.nn import Module, Parameter, ModuleList
from utils import get_slopes
from track import make_track_a_quadrupole, make_track_a_drift, Particle


class Beam(torch.nn.Module):
    def __init__(self, data, **kwargs):
        super(Beam, self).__init__()
        self.keys = ["x", "px", "y", "py", "z", "pz"]
        self.data = data

        for i, key in enumerate(self.keys):
            self.register_buffer(key, data[..., i])

        for name, val in kwargs.items():
            self.register_buffer(name, val)
        self._defaults = {"s": self.s, "p0c": self.p0c, "mc2": self.mc2}

    def to_list_of_beams(self):
        beams = []
        for i in range(len(getattr(self, self.keys[0]))):
            beams += [Particle(*[
                getattr(self, key)[i] for key in self.keys
            ], **self._defaults)]

        return beams

    @property
    def xp(self):
        return get_slopes(self.px, self.py, self.pz)[0]

    @property
    def yp(self):
        return get_slopes(self.px, self.py, self.pz)[1]


class TorchQuad(Module):

    def __init__(
            self,
            L: Tensor,
            K1: Tensor,
            NUM_STEPS: int = 1,
            X_OFFSET: Tensor = torch.tensor(0.0),
            Y_OFFSET: Tensor = torch.tensor(0.0),
    ):
        super(TorchQuad, self).__init__()
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_buffer("NUM_STEPS", torch.tensor(NUM_STEPS))

        self.K1 = K1
        self.track = make_track_a_quadrupole(torch)

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
        self.track = make_track_a_drift(torch)

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