import time

import numpy as np
import torch
from distgen import Generator

# create beam
from distgen.physical_constants import unit_registry as unit
from torch_track import Beam

# add a transformation to gen
from track import Particle


def generate_beam(
    n, n_samples=1, centroid_noise=0.0, centroid_offset=0.0, tkwargs=None
):
    tkwargs = tkwargs or {"device": "cpu", "dtype": torch.float32}

    gen = Generator("test_beams/gaussian.yaml")
    gen.input["n_particle"] = n

    beams = []
    for i in range(n_samples):
        gen.run()
        particles = gen.particles

        defaults = {
            "s": torch.tensor(0.0, **tkwargs),
            "p0c": torch.mean(torch.tensor(particles.pz, **tkwargs)),
            "mc2": torch.tensor(particles.mass, **tkwargs),
        }

        keys = ["x", "px", "y", "py", "z", "pz"]
        data = torch.cat(
            [
                torch.tensor(getattr(particles, key), **tkwargs).unsqueeze(0)
                for key in keys
            ]
        ).T

        # add in centroid noise
        data[:, 0] += torch.randn(1, **tkwargs) * centroid_noise + centroid_offset
        data[:, 2] += torch.randn(1, **tkwargs) * centroid_noise + centroid_offset

        data[:, 1] = data[:, 1] / defaults["p0c"]
        data[:, 3] = data[:, 3] / defaults["p0c"]
        data[:, 5] = (data[:, 5] - defaults["p0c"]) / defaults["p0c"]

        beams += [data.clone()]

    beam = Beam(torch.cat([b.unsqueeze(0) for b in beams]), **defaults)
    return beam
