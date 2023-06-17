import numpy as np
import distgen
from distgen import Generator
from distgen.physical_constants import unit_registry as unit

from math import pi

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

    linear_energy = {
        "type": "polynomial z:pz",
        "coefficients": [
            {"value": 0.0, "units": "MeV/c"},
            {"value": -75.0, "units": "MeV/c/meter"},
            {"value": 7500.0, "units": "MeV/c/meter/meter"}
        ]
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


    if distgen.__version__ >= '1.0.0':
        gen["transforms"] = {
            "twissx": twiss_x,
            "twissy": twiss_y,
            "pycos": pycos,
            "linear_position": linear_position,
            "linear_energy": linear_energy,
            "order": ["twissx", "twissy", "linear_position", "pycos"]
        }
    else:
        gen.input["transforms"] = {
            "twissx": twiss_x,
            "twissy": twiss_y,
            "pycos": pycos,
            "linear_position": linear_position,
            "linear_energy": linear_energy,
            "order": ["twissx", "twissy", "linear_position", "pycos", "linear_energy"]
        }

    # generate beam

    particle_group = gen.run()
    particle_group.drift_to_z(z=0)

    return particle_group
