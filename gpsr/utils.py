from gpsr.beams import NNParticleBeamGenerator, ResNNTransform
from gpsr.beams import ParticleBeam
import torch

LINEAR_BEAM_PHASE_SPACE_DIM = 6
LINEAR_BEAM_NOISE_SCALE = 1e-7


def calculate_centroid(images, x, y):
    x_projection = images.sum(dim=-2)
    y_projection = images.sum(dim=-1)

    # calculate weighted avg
    x_centroid = (x_projection * x).sum(-1) / (x_projection.sum(-1) + 1e-8)
    y_centroid = (y_projection * y).sum(-1) / (y_projection.sum(-1) + 1e-8)

    return torch.stack((x_centroid, y_centroid))


def calculate_ellipse(images, x, y):
    x_projection = images.sum(dim=-2)
    y_projection = images.sum(dim=-1)
    xx, yy = torch.meshgrid(x, y)
    xx = xx.unsqueeze(0).repeat(*images.shape[:-2], 1, 1)
    yy = yy.unsqueeze(0).repeat(*images.shape[:-2], 1, 1)

    # calculate weighted avg
    x_centroid = (x_projection * x).sum(-1) / (x_projection.sum(-1) + 1e-8)
    y_centroid = (y_projection * y).sum(-1) / (y_projection.sum(-1) + 1e-8)

    x_centroid = x_centroid.reshape(*images.shape[:-2], 1, 1)
    y_centroid = y_centroid.reshape(*images.shape[:-2], 1, 1)
    # calculate rms
    x_var = (images.transpose(-2, -1) * (xx - x_centroid) ** 2).sum((-1, -2)) / (
        images.sum((-1, -2)) + 1e-8
    )
    y_var = (images.transpose(-2, -1) * (yy - y_centroid) ** 2).sum((-1, -2)) / (
        images.sum((-1, -2)) + 1e-8
    )
    c_var = (images.transpose(-2, -1) * (xx - x_centroid) * (yy - y_centroid)).sum(
        (-1, -2)
    ) / (images.sum((-1, -2)) + 1e-8)

    cov = torch.empty(*images.shape[:-2], 2, 2)
    cov[..., 0, 0] = x_var
    cov[..., 1, 0] = c_var
    cov[..., 0, 1] = c_var
    cov[..., 1, 1] = y_var
    return torch.cat((x_centroid, y_centroid), dim=-1), cov


def get_norm_coords(beam_coords):
    # beam_coords is N x 6

    # center beam
    beam_coords = beam_coords - beam_coords.mean(dim=0)

    # get eignvectors/vals for cov matrix to normalize
    gt_cov = torch.cov(beam_coords.T)
    eig_val, eig_vec = torch.linalg.eigh(gt_cov)
    norm_coords = torch.inverse(eig_vec) @ beam_coords.T
    norm_coords = norm_coords.T

    # get norm cov
    norm_cov = torch.cov(norm_coords.T)
    norm_coords = norm_coords / torch.diagonal(norm_cov).sqrt()

    return norm_coords


def get_core_fraction(beam_coords, frac=0.9, dims=slice(0, 4), normalized_output=False):
    normalized_coords = get_norm_coords(beam_coords)[:, dims]
    origin_dist = torch.norm(normalized_coords, dim=1)

    if normalized_output:
        sorted_norm_coords = normalized_coords[torch.argsort(origin_dist)]
        return sorted_norm_coords[: int(beam_coords.shape[0] * frac)]
    else:
        sorted_coords = beam_coords[torch.argsort(origin_dist)]
        return sorted_coords[: int(beam_coords.shape[0] * frac)]


def to_linear_beam(beam_generator: NNParticleBeamGenerator) -> ParticleBeam:
    """Return a particle beam computed by the linear part of a ResNNTransform.

    Parameters
    ----------
    beam_generator : NNParticleBeamGenerator
        The beam generator containing a ResNNTransform as its transformer.

    Returns
    -------
    ParticleBeam
        The resulting ParticleBeam with linear output embedded in 6d phase space.
    """
    transformer = beam_generator.transformer

    if not isinstance(transformer, ResNNTransform):
        raise TypeError(
            f"Expected beam generator transformer to be an instance of ResNNTransform, "
            f"but got {type(transformer)}"
        )

    with torch.no_grad():
        linear_output = (
            transformer.linear_forward(beam_generator.base_particles)
            * transformer.output_scale
        )

    coords = (
        torch.randn(
            linear_output.shape[0],
            LINEAR_BEAM_PHASE_SPACE_DIM,
            device=linear_output.device,
            dtype=linear_output.dtype,
        )
        * LINEAR_BEAM_NOISE_SCALE
    )
    coords[:, : linear_output.shape[-1]] = linear_output
    coords = torch.cat((coords, torch.ones_like(coords[:, :1])), dim=-1)

    return ParticleBeam(
        particles=coords,
        energy=beam_generator.beam_energy,
        particle_charges=beam_generator.particle_charges,
        survival_probabilities=beam_generator.survival_probabilities,
    )
