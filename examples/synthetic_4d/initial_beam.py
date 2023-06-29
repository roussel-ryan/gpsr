import os

import torch
import numpy as np

import distgen
from distgen import Generator
from distgen.physical_constants import unit_registry as unit
from math import pi

from bmadx import M_ELECTRON
from bmadx.bmad_torch.track_torch import Beam
from bmadx.bmad_torch.track_torch import Beam
from bmadx.coordinates import openPMD_to_bmadx


def create_ground_truth_beam(folder='data', yaml_file='gaussian.yaml', gt_file='gt_dist.pt'):
    tkwargs = {"dtype": torch.float32}

    # create openPMD-beamphysics particle group
    par = create_initial_beam(os.path.join(folder, yaml_file))

    # transform to Bmad phase space coordinates
    p0c = 10.0e6 # reference momentum in eV
    coords = np.array(openPMD_to_bmadx(par, p0c)).T
    coords = torch.tensor(coords, **tkwargs)

    # create Bmad-x pytorch beam:
    gt_beam = Beam(
        coords,
        s=torch.tensor(0.0),
        p0c=torch.tensor(10.0e6),
        mc2=torch.tensor(M_ELECTRON)
    )

    # save ground truth beam
    torch.save(gt_beam, os.path.join(folder,gt_file))
    print(f'ground truth distribution saved at {os.path.join(folder,gt_file)}')

    return gt_beam

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
