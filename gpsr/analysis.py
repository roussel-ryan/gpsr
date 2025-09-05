import torch
from cheetah.particles import ParticleBeam


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


def get_beam_fraction(beam_distribution, beam_fraction, particle_slices=None):
    """
    Get the core of the beam according to 6D normalized beam coordinates.
    Particles from the beam distribution are scaled to normalized coordinates
    via the covariance matrix. Then they are sorted by distance from the origin.

    Parameters
    ----------
    beam_distribution: ParticleBeam
        Cheetah ParticleBeam object representing the beam distribution.
    beam_fraction: float
        The fraction of the beam to keep from 0 - 1.
    particle_slices: list[slice] or slice, optional
        List of slices to apply along the particle coordinates

    Returns
    -------
    ParticleBeam
        The core of the beam represented as a Cheetah ParticleBeam object.

    """
    # extract macroparticles
    macroparticles = beam_distribution.particles.detach()[..., :6]

    if isinstance(particle_slices, list):
        data = macroparticles[..., particle_slices]
    elif isinstance(particle_slices, slice):
        data = macroparticles[..., particle_slices]
    else:
        data = macroparticles

    # calculate covariance matrix
    cov = torch.cov(data.T)

    # get inverse cholesky decomp -- if it fails, add a small value to the
    # diagonal and try again
    try:
        cov_cholesky = torch.linalg.cholesky(cov)
    except torch._C._LinAlgError as e:
        cov_cholesky = torch.linalg.cholesky(cov + 1e-8 * torch.eye(6))

    # transform particles to normalized coordinates
    t_data = (torch.linalg.inv(cov_cholesky) @ data.T).T

    # sort particles by their distance from the origin and hold onto a fraction of them
    J = torch.linalg.norm(t_data, axis=1)
    sort_idx = torch.argsort(J)
    frac_coords = macroparticles[sort_idx][: int(len(data) * beam_fraction)]

    # create a beam distribution to return
    frac_coords = torch.hstack((frac_coords, torch.ones((len(frac_coords), 1))))
    frac_particle = ParticleBeam(
        frac_coords.to(beam_distribution.energy),
        energy=beam_distribution.energy,
    )

    return frac_particle

def compute_fractional_emittance_curve(
    fractions,
    distribution,
    _slice
):
    """
    Compute the fractional emittance curve for a given distribution.

    Parameters
    ----------
    fractions: list[float]
        List of fractions to compute the emittance for.
    distribution: ParticleBeam
        The particle beam distribution to analyze.
    _slice: slice
        The slice of particle coordinates to consider.

    Returns
    -------
    torch.Tensor
        A tensor containing the fractional emittance values for each fraction.
    """

    result = []
    for f in fractions:
        frac_dist = get_beam_fraction(distribution, f, particle_slices=_slice)

        particles = frac_dist.particles[..., _slice]
        n_dims = particles.shape[-1]
        
        result += [torch.det(torch.cov(particles.T)).pow(1/n_dims)]

    return torch.tensor(result)