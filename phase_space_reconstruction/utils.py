import matplotlib.pyplot as plt
import numpy as np
import torch

from phase_space_reconstruction.modeling import ImageDataset3D


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


def split_2screen_dset(dset):
    n = dset.__len__()
    train_dset_params, train_dset_imgs = dset.__getitem__(np.arange(0, n, 2))
    train_dset = ImageDataset3D(train_dset_params, train_dset_imgs)
    test_dset_params, train_dset_imgs = dset.__getitem__(np.arange(1, n, 2))
    test_dset = ImageDataset3D(test_dset_params, train_dset_imgs)
    return train_dset, test_dset
