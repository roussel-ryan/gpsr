import matplotlib.pyplot as plt
import numpy as np
import torch
from bmadx.bmad_torch.track_torch import Beam
from bmadx.structures import Particle
from pmd_beamphysics.particles import ParticleGroup

# --------------------------------------------------------------------------


def screen_stats(image, bins_x, bins_y):
    """
    Returns screen stats

    Parameters
    ----------
    image: 2D array-like
        screen image of size [n_x, n_y].

    bins_x: 1D array-like
        x axis bins physical locations of size [n_x]

    bins_y: 2D array-like
        x axis bins physical locations of size [n_y]

    Returns
    -------
    dictionary with 'avg_x', 'avg_y', 'std_x' and 'std_y'.
    """
    proj_x = image.sum(axis=1)
    proj_y = image.sum(axis=0)

    # stats
    avg_x = (bins_x * proj_x).sum() / proj_x.sum()
    avg_y = (bins_y * proj_y).sum() / proj_y.sum()

    std_x = (((bins_x * proj_x - avg_x) ** 2).sum() / proj_x.sum()) ** (1 / 2)
    std_y = (((bins_y * proj_y - avg_y) ** 2).sum() / proj_y.sum()) ** (1 / 2)

    return {"avg_x": avg_x, "avg_y": avg_y, "std_x": std_x, "std_y": std_y}


# --------------------------------------------------------------------------


def calculate_beam_matrix(beam_distribution: ParticleGroup, beam_fraction: float = 1.0):
    fractional_beam = get_beam_fraction_openpmd_par(beam_distribution, beam_fraction)
    return fractional_beam.cov("x", "py", "y", "py", "t", "pz")


def get_beam_fraction_openpmd_par(beam_distribution: ParticleGroup, beam_fraction):
    """get core of the beam according to 6D normalized beam coordinates"""
    vnames = ["x", "px", "y", "py", "t", "pz"]
    data = np.copy(np.stack([beam_distribution[name] for name in vnames]).T)
    data[:, -2] = data[:, -2] - np.mean(data[:, -2])
    data[:, -1] = data[:, -1] - np.mean(data[:, -1])
    cov = np.cov(data.T)

    # get inverse cholesky decomp
    t_data = (np.linalg.inv(np.linalg.cholesky(cov)) @ data.T).T

    J = np.linalg.norm(t_data, axis=1)
    sort_idx = np.argsort(J)
    frac_beam = beam_distribution[sort_idx][
        : int(len(beam_distribution) * beam_fraction)
    ]

    return frac_beam


def get_beam_fraction_bmadx_beam(beam_distribution: Beam, beam_fraction):
    """get core of the beam according to 6D normalized beam coordinates"""
    data = beam_distribution.data.detach().clone().numpy()
    # data[:, -2] = data[:, -2] - np.mean(data[:, -2])
    # data[:, -1] = data[:, -1] - np.mean(data[:, -1])
    cov = np.cov(data.T)

    # get inverse cholesky decomp
    t_data = (np.linalg.inv(np.linalg.cholesky(cov)) @ data.T).T

    J = np.linalg.norm(t_data, axis=1)
    sort_idx = np.argsort(J)
    frac_coords = data[sort_idx][: int(len(data) * beam_fraction)]
    frac_beam = Beam(
        torch.tensor(frac_coords),
        p0c=beam_distribution.p0c,
        s=beam_distribution.s,
        mc2=beam_distribution.mc2,
    )

    return frac_beam


def get_beam_fraction_bmadx_particle(beam_distribution: Particle, beam_fraction):
    """get core of the beam according to 6D normalized beam coordinates"""
    data = np.stack(beam_distribution[:6]).T
    # data[:, -2] = data[:, -2] - np.mean(data[:, -2])
    # data[:, -1] = data[:, -1] - np.mean(data[:, -1])
    cov = np.cov(data.T)

    # get inverse cholesky decomp
    t_data = (np.linalg.inv(np.linalg.cholesky(cov)) @ data.T).T

    J = np.linalg.norm(t_data, axis=1)
    sort_idx = np.argsort(J)
    frac_coords = data[sort_idx][: int(len(data) * beam_fraction)]
    frac_particle = Particle(
        *frac_coords.T,
        p0c=beam_distribution.p0c,
        s=beam_distribution.s,
        mc2=beam_distribution.mc2
    )

    return frac_particle


def get_beam_fraction_numpy_coords(beam_distribution, beam_fraction):
    """get core of the beam according to 6D normalized beam coordinates"""
    data = np.stack(beam_distribution[:6]).T
    cov = np.cov(data.T)

    # get inverse cholesky decomp
    t_data = (np.linalg.inv(np.linalg.cholesky(cov)) @ data.T).T

    J = np.linalg.norm(t_data, axis=1)
    sort_idx = np.argsort(J)
    frac_coords = data[sort_idx][: int(len(data) * beam_fraction)].T

    return frac_coords
