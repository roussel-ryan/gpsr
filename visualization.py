import matplotlib.pyplot as plt
import numpy as np
import torch


def compare_images(model, lattice, train_images, n_images=5):
    fig, ax = plt.subplots(n_images, 2, sharey="all", sharex="all")
    xx = np.meshgrid(model.bins.cpu().numpy(), model.bins.cpu().numpy())

    # calculate images
    predicted_images = model(lattice)

    vmin = 0
    vmax = max(predicted_images.max(), train_images.max())

    for i in range(n_images):
        ax[i, 0].pcolor(*xx, train_images[i].cpu().detach(), vmin=vmin, vmax=vmax)
        ax[i, 1].pcolor(*xx, predicted_images[i].cpu().detach(), vmin=vmin, vmax=vmax)

    # add titles
    ax[0, 0].set_title("Ground truth")
    ax[0, 1].set_title("Model prediction")
    ax[-1, 0].set_xlabel("x (m)")
    ax[-1, 1].set_xlabel("x (m)")

    for i in range(n_images):
        ax[i, 0].set_ylabel("y (m)")

    return fig


def plot_reconstructed_phase_space(x, y, ms):
    #define mesh
    bins = ms[0].bins.cpu().numpy()
    xx = np.meshgrid(bins, bins)

    # calculate histograms
    histograms = []
    for ele in ms:
        initial_beam = ele.get_initial_beam(100000)
        histograms += [np.histogram2d(
            getattr(initial_beam, x).cpu().detach().numpy(),
            getattr(initial_beam, y).cpu().detach().numpy(),
            bins=bins
        )[0]]

        del initial_beam
        torch.cuda.empty_cache()

    if len(ms) != 1:
        # calculate mean and std of histograms
        histograms = np.asfarray(histograms)

        means = np.mean(histograms, axis=0)
        stds = np.std(histograms, axis=0)

        fig, ax = plt.subplots(3, 1, sharex="all", sharey="all")
        fig.set_size_inches(8, 16)
        for a in ax:
            a.set_aspect("equal")

        c = ax[0].pcolor(*xx, means.T)
        fig.colorbar(c, ax=ax[0])
        c = ax[1].pcolor(*xx, stds.T)
        fig.colorbar(c, ax=ax[1])

        # fractional error
        c = ax[2].pcolor(*xx, stds.T / means.T)
        fig.colorbar(c, ax=ax[2])
    else:
        fig, ax = plt.subplots()
        c = ax.pcolor(*xx, histograms[0].T)
        fig.colorbar(c, ax=ax)
