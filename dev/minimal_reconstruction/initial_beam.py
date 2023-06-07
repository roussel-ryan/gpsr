import numpy as np
from distgen import Generator
from distgen.physical_constants import unit_registry as unit
from distgen.physical_constants import c
from math import pi
from matplotlib import pyplot as plt

import sys
sys.path.append("../../../")

from phase_space_reconstruction.histogram import histogram2d
from bmadx.bmad_torch.track_torch import Beam, TorchDrift, TorchLattice, TorchQuadrupole


def create_initial_beam(yaml_file):
    """
    Returns initial beam in openPMD-beamphysics ParticleGroup standard
    using the provided yaml_file and transformations.

        Parameters: 
            yaml_file (str): yaml file to generate non-transformed beam

        Returns:
            particle_group (ParticleGroup): initial beam.
            Note: openPMD-beamphysics ParticleGroup coords are defined
            different from bmad coords, specially the z coordinate.
    """

    gen = Generator(yaml_file)

    # Transforms: 
    k = 2 * pi / (30 * unit("mm"))
    pycos = {
        "type": "cosine y:py",
        "amplitude": {"value": 0.05, "units": "MeV/c"},
        "omega": {"value": k.magnitude, "units": str(k.units)},
        "phase": {"value": 90, "units": "deg"},
    }

    linear_position = {
        "type": "polynomial x:y",
        "coefficients": [
            {"value": -0.005, "units": "m"},
            {"value": 0.75, "units": ""},
            {"value": 50.0, "units":"1/m"}
        ]
    }

    twiss_x = {
        "type": "set_twiss x",
        "beta": {
            "value": 9,
            "units": "m",
        },
        "alpha": {"value": 5, "units": ""},
        "emittance": {"value": 2.0, "units": "um"},
    }

    twiss_y = {
        "type": "set_twiss y",
        "beta": {
            "value": 9,
            "units": "m",
        },
        "alpha": {"value": 0, "units": ""},
        "emittance": {"value": 2.0, "units": "um"},
    }

    gen.input["transforms"] = {
        "twissx": twiss_x,
        "twissy": twiss_y,
        "pycos": pycos,
        "linear_position": linear_position,
        "order": ["twissx", "twissy", "linear_position", "pycos"]
    }

    # generate beam

    particle_group = gen.run()
    particle_group.drift_to_z(z=0)

    return particle_group


def transform_to_bmad_coords(particle_group, p0c):
    """
    Transforms openPMD-beamphysics ParticleGroup to 
    bmad phase-space coordinates.

        Parameters:
            particle_group (ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV

        Returns:
            bmad_coords (ndarray): bmad phase space coordinates
    """

    x = particle_group.x
    px = particle_group.px / p0c
    y = particle_group.y
    py = particle_group.py / p0c
    z = particle_group.beta * 299792458 * particle_group.t
    pz = particle_group.p / p0c -1.0

    bmad_coords = np.column_stack((x, px, y, py, z, pz))

    return bmad_coords

