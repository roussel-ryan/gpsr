import time

import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss

from histogram import histogram2d
from track import Quadrupole, Drift, track_a_lattice


def track_in_quad(beam, lattice):
    result = track_a_lattice(beam, lattice)[-1]
    return result


def final_beam_size(beam, k):
    p_out = track_in_quad(beam, k)
    return torch.std(p_out.x), torch.std(p_out.y)


def final_beam_histogram(beam, k, bins, bandwidth):
    p_out = track_in_quad(beam, k)
    return histogram2d(p_out.x.unsqueeze(0), p_out.y.unsqueeze(0), bins, bandwidth)[0]


def get_images(output_beams, bins, bandwidth):
    # get beam images
    x = output_beams.x
    y = output_beams.y

    images = histogram2d(x, y, bins, bandwidth)

    return images


def get_mean_position(beams):
    x = beams.x
    y = beams.y

    mean_x = torch.mean(x, dim=-1)
    mean_y = torch.mean(y, dim=-1)

    return torch.sum(torch.sqrt(mean_x ** 2 + mean_y ** 2))


def zero_centroid_loss(beam):
    # calculate the beam centroids for each dim and add them in quadrature
    keys = ["x", "px", "y", "py", "z", "pz"]
    coords = torch.cat([getattr(beam, key).unsqueeze(0) for key in keys], dim=0)
    return torch.sum(coords.mean().pow(2))


def image_difference_loss(test_beam, true_beam_images, lattice, bins, bandwidth,
                          plot_images=False, n_images=5):
    test_output_beams = lattice(test_beam)[-1]
    beam1_images = get_images(test_output_beams, bins, bandwidth)

    loss = MSELoss(reduction="sum")

    if plot_images:
        # plot the first 5 images
        fig, ax = plt.subplots(n_images, 2, sharex="all", sharey="all")
        fig.set_size_inches(8, 8)
        xx = torch.meshgrid(bins.cpu(), bins.cpu())

        vmax = torch.max(true_beam_images)
        vmin = 0

        for i in range(n_images):
            ax[i, 0].pcolor(*xx, beam1_images[i].cpu().detach(), vmin=vmin, vmax=vmax)
            ax[i, 1].pcolor(*xx, true_beam_images[i].cpu().detach(), vmin=vmin, vmax=vmax)

            print(torch.max(beam1_images[i].cpu().detach()))
            print(torch.max(true_beam_images[i].cpu().detach()))

        # add titles
        ax[0, 0].set_title("Model prediction")
        ax[0, 1].set_title("Ground truth")
        ax[-1, 0].set_xlabel("x (m)")
        ax[-1, 1].set_xlabel("x (m)")

        for i in range(n_images):
            ax[i, 0].set_ylabel("y (m)")

    return loss(beam1_images, true_beam_images)


def beam_position_loss(beam):
    return torch.mean(beam.x) ** 2 + torch.mean(beam.y) ** 2 + torch.mean(beam.z) ** 2


