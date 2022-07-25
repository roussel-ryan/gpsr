import matplotlib.pyplot as plt
import numpy as np
import torch


def add_image_projection(ax, image, xx, axis):
    if axis == "x":
        axb = ax.twinx()
        proj = image.sum(dim=0)
        proj = proj / proj.sum()

        axb.plot(xx[0][0], proj, "r")
        axb.set_ylim(0.0, 0.5)
        axb.set_yticks([])
        axb.set_xticks([])

    elif axis == "y":
        axb = ax.twiny()
        proj = image.sum(dim=1)
        proj = proj / proj.sum()

        axb.plot(proj, xx[1].T[0], "r")
        axb.set_xlim(0.0, 0.5)
        axb.set_yticks([])
        axb.set_xticks([])
    else:
        raise RuntimeError()


def compare_images(xx, predicted_images, train_images):
    n_images = len(predicted_images)
    fig, ax = plt.subplots(n_images, 2, sharey="all", sharex="all")

    vmin = 0
    vmax = max(predicted_images.max(), train_images.max())

    for i in range(n_images):
        ax[i, 0].pcolor(*xx, train_images[i].cpu().detach(), vmin=vmin, vmax=vmax)
        ax[i, 1].pcolor(*xx, predicted_images[i].cpu().detach(), vmin=vmin, vmax=vmax)

        #add_image_projection(ax[i, 0], train_images[i].cpu().detach(), xx, "x")
        #add_image_projection(ax[i, 0], train_images[i].cpu().detach(), xx, "y")

        #add_image_projection(ax[i, 1], predicted_images[i].cpu().detach(), xx, "x")
        #add_image_projection(ax[i, 1], predicted_images[i].cpu().detach(), xx, "y")

    # add titles
    ax[0, 0].set_title("Ground truth")
    ax[0, 1].set_title("Model prediction")
    ax[-1, 0].set_xlabel("x (m)")
    ax[-1, 1].set_xlabel("x (m)")

    for i in range(n_images):
        ax[i, 0].set_ylabel("y (m)")

    return fig


def compare_image_projections(x, train_images, predicted_images):
    fig, ax = plt.subplots(len(predicted_images), 2, sharex="all")

    for images in [train_images, predicted_images]:
        for jj in range(2):
            if jj == 0:
                # get projections along x axis
                projections = images.sum(-1)
            elif jj == 1:
                projections = images.sum(-2)

            # calc stats
            mean_proj = projections.mean(-2)
            l_proj = torch.quantile(projections, 0.05, dim=-2)
            u_proj = torch.quantile(projections, 0.95, dim=-2)

            for ii in range(len(predicted_images)):
                ax[ii][jj].plot(x, mean_proj[ii])
                ax[ii][jj].fill_between(x, l_proj[ii], u_proj[ii], alpha=0.25)

    return fig


def get_predictive_distribution(model_image_mean, model_image_variance):
    # get pixelized probability distribution based on nn predictions
    # clip variance to not be zero
    model_image_variance = torch.clip(model_image_variance, min=1e-6)
    model_image_mean = torch.clip(model_image_mean, min=1e-6)

    concentration = model_image_mean ** 2 / model_image_variance
    rate = model_image_mean / model_image_variance

    # form distribution
    return torch.distributions.Gamma(concentration, rate)


def calculate_pixel_log_likelihood(model_image_mean, model_image_variance, true_image):
    # use a gamma distribution to calculate the likelihood at each pixel
    dist = get_predictive_distribution(model_image_mean, model_image_variance)

    # replace zeros with nans
    true = true_image.clone()
    true[true_image == 0] = 1e-6
    return dist.log_prob(true)


def beam_to_tensor(beam):
    keys = ["x", "px", "y", "py", "z", "pz"]
    data = []
    for key in keys:
        data += [getattr(beam, key).cpu()]

    return torch.cat([ele.unsqueeze(1) for ele in data], dim=1)


def calculate_covariances(true_beam, model_beams):
    beams = [true_beam] + model_beams
    covars = torch.empty(len(beams), 6, 6)
    for i, beam in enumerate(beams):
        data = beam_to_tensor(beam).cpu()
        covars[i] = torch.cov(data.T)

    stats = torch.cat(
        [
            covars[0].flatten().unsqueeze(1),
            torch.mean(covars[1:], dim=0).flatten().unsqueeze(1),
            torch.std(covars[1:], dim=0).flatten().unsqueeze(1),
        ],
        dim=1,
    )
    print(stats)


def plot_log_likelihood(x, y, true_beam, model_beams, bins):
    # plot the log likelihood of a collection of test_beams predicted by the model

    xx = torch.meshgrid(*bins)

    # calculate histograms
    all_beams = [true_beam] + model_beams

    histograms = []
    for beam in all_beams:
        # convert beam to tensor
        data = torch.cat(
            [getattr(beam, ele).cpu().detach().unsqueeze(0) for ele in [x, y]]
        ).T

        histograms += [
            torch.histogramdd(data, bins=bins, density=True).hist.unsqueeze(0)
        ]

    histograms = torch.cat(histograms, dim=0)
    # for h in histograms:
    #    fig, ax = plt.subplots()
    #    c = ax.pcolor(*xx, h)
    #    fig.colorbar(c)

    # plot mean / var / log-likelihood
    meas_mean = torch.mean(histograms[1:], dim=0)
    meas_var = torch.var(histograms[1:], dim=0)
    log_lk = calculate_pixel_log_likelihood(meas_mean, meas_var, histograms[0])

    # remove locations where the true val is zero
    log_lk[histograms[0] == 0] = torch.nan

    fig, ax = plt.subplots(4, 1, sharex="all", sharey="all")
    plot_data = [histograms[0], meas_mean, meas_var.sqrt()]
    for i, d in enumerate(plot_data):
        c = ax[i].pcolor(*xx, d)
        fig.colorbar(c, ax=ax[i])

    c = ax[-1].pcolor(*xx, log_lk, vmin=-10, vmax=0)
    fig.colorbar(c, ax=ax[-1])


def add_projection(ax, key, beams, bins, scale_axis=1):
    histograms = []
    for ele in beams:
        histograms += [
            np.histogram(
                getattr(ele, key).cpu().detach().numpy(), bins=bins.cpu(), density=True
            )[0]
        ]

    histograms = np.asfarray(histograms)

    means = np.mean(histograms, axis=0)
    l = np.quantile(histograms, 0.05, axis=0)
    u = np.quantile(histograms, 0.95, axis=0)

    ax.plot(bins[:-1].cpu()*scale_axis, means, label=key)
    ax.fill_between(bins[:-1].cpu()*scale_axis, l, u, alpha=0.5)

    return ax


def add_image(ax, key1, key2, beams, bins, scale_axis=1):
    histograms = []
    xx = np.meshgrid(bins.cpu(), bins.cpu())

    for ele in beams:
        histograms += [
            np.histogram2d(
                getattr(ele, key1).cpu().detach().numpy(),
                getattr(ele, key2).cpu().detach().numpy(),
                bins=bins.cpu(),
                density=True,
            )[0].T
        ]

    histograms = np.asfarray(histograms)

    means = np.mean(histograms, axis=0)
    l = np.quantile(histograms, 0.05, axis=0)
    u = np.quantile(histograms, 0.95, axis=0)

    ax.pcolor(xx[0]*scale_axis,xx[1]*scale_axis, means)

    return ax, means


def plot_reconstructed_phase_space_projections(x, true_beam, model_beams, bins):
    # define mesh
    beams = [true_beam] + model_beams

    # calculate histograms
    histograms = []
    for ele in beams:
        histograms += [
            np.histogram(
                getattr(ele, x).cpu().detach().numpy(), bins=bins, density=True
            )[0]
        ]

    # calculate mean and std of histograms
    histograms = np.asfarray(histograms)

    means = np.mean(histograms, axis=0)
    stds = np.std(histograms, axis=0)

    fig, ax = plt.subplots()
    ax.plot(bins[:-1], means)
    ax.fill_between(bins[:-1], means - stds, means + stds, alpha=0.5)

    ax.plot(bins[:-1], histograms[0])
    ax.set_title(x)


def plot_reconstructed_phase_space(x, y, ms):
    # define mesh
    bins = ms[0].bins.cpu().numpy()
    xx = np.meshgrid(bins, bins)

    # calculate histograms
    histograms = []
    for ele in ms:
        initial_beam = ele.get_initial_beam(100000)
        histograms += [
            np.histogram2d(
                getattr(initial_beam, x).cpu().detach().numpy(),
                getattr(initial_beam, y).cpu().detach().numpy(),
                bins=bins,
            )[0]
        ]

        del initial_beam
        torch.cuda.empty_cache()

    if len(ms) != 1:
        # calculate mean and std of histograms
        histograms = np.asfarray(histograms)

        means = np.mean(histograms, axis=0)
        stds = np.std(histograms, axis=0)

        fig, ax = plt.subplots(3, 1, sharex="all", sharey="all")
        fig.set_size_inches(4, 9)
        for a in ax:
            a.set_aspect("equal")

        c = ax[0].pcolor(*xx, means.T)
        fig.colorbar(c, ax=ax[0])
        c = ax[1].pcolor(*xx, stds.T)
        fig.colorbar(c, ax=ax[1])

        # fractional error
        c = ax[2].pcolor(*xx, stds.T / means.T)
        fig.colorbar(c, ax=ax[2])

        ax[2].set_xlabel(x)
        for a in ax:
            a.set_ylabel(y)
    else:
        fig, ax = plt.subplots()
        c = ax.pcolor(*xx, histograms[0].T)
        fig.colorbar(c, ax=ax)
