import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import transforms
from matplotlib.patches import Ellipse

from phase_space_reconstruction.utils import split_2screen_dset

# --------------------------------------------------------------------------


def plot_scan_data(train_dset, test_dset, bins_x, bins_y):
    """
    Plots train and test images from data sets sorted by quad strength.

    Parameters
    ----------
    train_dset: ImageDataset
        training dataset. train_dset.k is of shape
        [n_scan x n_imgs x 1]
        train_dset.images is of shape
        [n_scan x n_imgs x pixels_x x pixels_y]

    test_dset: ImageDataset
        test dataset.

    bins_x: array-like
        Array of pixel centers that correspond to the physical diagnostic.
        Should have length pixels_x

    bins_y: array-like
        Array of pixel centers that correspond to the physical diagnostic.
        Should have length pixels_y

    """

    # id is zero if sample is train and 1 if test
    train_ids = torch.zeros(len(train_dset.k), dtype=torch.bool)
    test_ids = torch.ones(len(test_dset.k), dtype=torch.bool)
    is_test = torch.hstack((train_ids, test_ids))

    # stack training and tests data
    all_k = torch.vstack((train_dset.k, test_dset.k))
    all_im = torch.vstack((train_dset.images, test_dset.images))

    # sort by k value
    _, indices = torch.sort(all_k[:, 0, 0], dim=0, stable=True)
    sorted_k = all_k[indices]
    sorted_im = all_im[indices]
    sorted_is_test = is_test[indices]

    # plot
    n_k = len(sorted_k)
    imgs_per_k = sorted_im.shape[1]
    fig, ax = plt.subplots(imgs_per_k, n_k + 1, figsize=(n_k + 1, imgs_per_k))
    extent = np.array([bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]])

    if imgs_per_k == 1:
        for i in range(n_k):
            ax[i + 1].imshow(
                sorted_im[i, 0].T, origin="lower", extent=extent, interpolation="none"
            )
            ax[i + 1].tick_params(
                bottom=False, left=False, labelbottom=False, labelleft=False
            )
            if sorted_is_test[i]:
                for spine in ax[i + 1].spines.values():
                    spine.set_edgecolor("orange")
                    spine.set_linewidth(2)

            ax[i + 1].set_title(f"{sorted_k[i, 0, 0]:.1f}")
            ax[0].set_axis_off()
            ax[0].text(0.5, 0.5, f"img 1", va="center", ha="center")

        ax[0].set_title("$k$ (1/m):")

    else:
        for i in range(n_k):
            for j in range(imgs_per_k):
                ax[j, i + 1].imshow(
                    sorted_im[i, j].T,
                    origin="lower",
                    extent=extent,
                    interpolation="none",
                )
                ax[j, i + 1].tick_params(
                    bottom=False, left=False, labelbottom=False, labelleft=False
                )
                if sorted_is_test[i]:
                    for spine in ax[j, i + 1].spines.values():
                        spine.set_edgecolor("orange")
                        spine.set_linewidth(2)

            ax[0, i + 1].set_title(f"{sorted_k[i, 0, 0]:.1f}")

        for j in range(imgs_per_k):
            ax[j, 0].set_axis_off()
            ax[j, 0].text(0.5, 0.5, f"img {j + 1}", va="center", ha="center")

        ax[0, 0].set_title("$k$ (1/m$^2$):")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    print(
        f"image size = {(bins_x[-1] - bins_x[0]) * 1e3:.0f} x {(bins_y[-1] - bins_y[0]) * 1e3:.0f} mm"
    )
    print("test samples boxed in orange")

    return fig, ax


def plot_predicted_screens(prediction_dset, train_dset, test_dset, bins_x, bins_y):
    """
    Plots predictions (and measurements for reference)

    Parameters
    ----------
    prediction_dset: ImageDataset
        predicted screens dataset. prediction_dset.k is of shape
        [n_scan x n_imgs x 1]
        prediction_dset.images is of shape
        [n_scan x n_imgs x pixels_x x pixels_y]

    train_dset: ImageDataset
        training dataset.

    test_dset: ImageDataset
        test dataset

    bins_x: array-like
        Array of pixel centers that correspond to the physical diagnostic.
        Should have length pixels_x

    bins_y: array-like
        Array of pixel centers that correspond to the physical diagnostic.
        Should have length pixels_y

    """

    # sort prediction dset
    _, indices = torch.sort(prediction_dset.k[:, 0, 0], dim=0, stable=True)
    sorted_pred = prediction_dset.images[indices]

    # id is zero if sample is train and 1 if test
    train_ids = torch.zeros(len(train_dset.k), dtype=torch.bool)
    test_ids = torch.ones(len(test_dset.k), dtype=torch.bool)
    is_test = torch.hstack((train_ids, test_ids))

    # stack training and tests data
    all_k = torch.vstack((train_dset.k, test_dset.k))
    all_im = torch.vstack((train_dset.images, test_dset.images))

    # sort by k value
    _, indices = torch.sort(all_k[:, 0, 0], dim=0, stable=True)
    sorted_k = all_k[indices]
    sorted_im = all_im[indices]
    sorted_is_test = is_test[indices]

    # plot
    n_k = len(sorted_k)
    imgs_per_k = sorted_im.shape[1]
    fig, ax = plt.subplots(imgs_per_k + 1, n_k + 1, figsize=(n_k + 1, imgs_per_k + 1))
    extent = np.array([bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]])

    for i in range(n_k):
        for j in range(imgs_per_k):
            ax[j, i + 1].imshow(
                sorted_im[i, j].T, origin="lower", extent=extent, interpolation="none"
            )

            ax[j, i + 1].tick_params(
                bottom=False, left=False, labelbottom=False, labelleft=False
            )

            if sorted_is_test[i]:
                for spine in ax[j, i + 1].spines.values():
                    spine.set_edgecolor("orange")
                    spine.set_linewidth(2)

        ax[0, i + 1].set_title(f"{sorted_k[i, 0, 0]:.1f}")

        ax[-1, i + 1].imshow(
            sorted_pred[i, j].T, origin="lower", extent=extent, interpolation="none"
        )

        ax[-1, i + 1].tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False
        )

        if sorted_is_test[i]:
            for spine in ax[-1, i + 1].spines.values():
                spine.set_edgecolor("orange")
                spine.set_linewidth(2)

        ax[-1, 0].set_axis_off()
        ax[-1, 0].text(0.5, 0.5, "pred", va="center", ha="center")

    for j in range(imgs_per_k):
        ax[j, 0].set_axis_off()
        ax[j, 0].text(0.5, 0.5, f"img {j + 1}", va="center", ha="center")

    ax[0, 0].set_title("$k$ (1/m$^2$):")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    print(
        f"image size = {(bins_x[-1] - bins_x[0]) * 1e3:.0f} x {(bins_y[-1] - bins_y[0]) * 1e3:.0f} mm"
    )
    print("test samples boxed in orange")

    return fig, ax


def plot_3d_scan_data(train_dset, bins, publication_size=False):
    # reshape data into parameter 3D mesh:
    n_k = len(
        torch.unique(train_dset.params.squeeze(-1)[:, 0])
    )  # number of quad strengths
    n_v = len(torch.unique(train_dset.params.squeeze(-1)[:, 1]))  # number of TDC states
    n_g = len(
        torch.unique(train_dset.params.squeeze(-1)[:, 2])
    )  # number of dipole states
    image_shape = train_dset.images.shape
    params = train_dset.params.reshape((n_k, n_v, n_g, 3))
    images = train_dset.images.reshape(
        (n_k, n_v, n_g, image_shape[-2], image_shape[-1])
    )
    
    xx = torch.meshgrid(bins*1e3,bins*1e3)
    print(xx[0].shape)

    vmax = torch.max(images)
    # plot
    if publication_size:
        figsize = (7.5, (n_v + n_g) * 1.4)
        kwargs = {
            "top": 0.925,
            "bottom": 0.025,
            "right": 0.975,
            "hspace": 0.1,
            "wspace": 0.1,
        }
    else:
        figsize = ((n_k + 1) * 2, (n_v + n_g + 1) * 2)
        kwargs = {"right": 0.9}
    fig, ax = plt.subplots(n_v + n_g, n_k, figsize=figsize, gridspec_kw=kwargs,sharex="all",sharey="all")

    #ax[0, 0].set_axis_off()
    ax[0, 0].text(
        -0.1,
        1.1,
        "$k_1$ (1/m$^2$)",
        va="bottom",
        ha="right",
        transform=ax[0, 0].transAxes,
    )
    for i in range(n_k):
        #ax[0, i].set_axis_off()
        ax[0, i].text(
            0.5,
            1.1,
            f"{params[i, 0, 0, 0]:.1f}",
            va="bottom",
            ha="center",
            transform=ax[0, i].transAxes,
        )
        for k in range(n_v):
            for j in range(n_g):
                row_number = 2 * k + j
                ax[row_number, i].pcolormesh(
                    xx[0].numpy(), xx[1].numpy(),
                    images[i, j, k] / images[i, j, k].max(),
                    rasterized=True,
                    vmax=1.0,
                    vmin=0,
                )

                if j == 0:
                    v_lbl = "off"
                else:
                    v_lbl = "on"
                if k == 0:
                    g_lbl = "off"
                else:
                    g_lbl = "on"

                if i == 0:
                    ax[row_number, 0].text(
                        -0.6,
                        0.5,
                        f"T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}",
                        va="center",
                        ha="right",
                        transform=ax[row_number, 0].transAxes,
                    )
    # fig.tight_layout()
    for ele in ax[-1]:
        ele.set_xlabel("x (mm)")
    for ele in ax[:,0]:
        ele.set_ylabel("y (mm)")
    return fig, ax


def plot_3d_scan_data_2screens(dset, select_img=0, splitted=True):
    """
    Plots 3D scan dataset for 6D phase space reconstruction
    with 2 screens.

    Parameters
    ----------
    dset: ImageDataset
        scan data.
        dset.images should be a 6D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of images per parameter configuration,
        screen width in pixels,
        screen height in pixels]
        dset.params should be a 4D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of scanning elements (3: quad, tdc, dipole) ]

    select_img: int
        index of image to plot for each parameter configuration

    splitted: bool
        if True, data is assumed to be splitted into train and test data.
        dset.images
    Returns
    -------
    fig: matplotlib figure
        figure object
    """
    params = dset.params
    imgs = dset.images[:, :, :, select_img, :, :]
    n_k = params.shape[0]
    n_v = params.shape[1]
    n_g = params.shape[2]
    fig, ax = plt.subplots(
        n_v * n_g + 1, n_k + 1, figsize=((n_k + 1) * 2, (n_v * n_g + 1) * 2)
    )
    ax[0, 0].set_axis_off()
    ax[0, 0].text(1, 0, "$k_1$ (1/m$^2$)", va="bottom", ha="right")
    for i in range(n_k):
        ax[0, i + 1].set_axis_off()
        ax[0, i + 1].text(0.5, 0, f"{params[i, 0, 0, 0]:.1f}", va="bottom", ha="center")
        for j in range(n_g):
            for k in range(n_v):
                ax[2 * j + k + 1, i + 1].imshow(
                    imgs[i, k, j].T, origin="lower", interpolation="none"
                )
                ax[2 * j + k + 1, i + 1].tick_params(
                    bottom=False, left=False, labelbottom=False, labelleft=False
                )

                if k == 0:
                    v_lbl = "off"
                else:
                    v_lbl = "on"
                if j == 0:
                    g_lbl = "off"
                else:
                    g_lbl = "on"

                ax[2 * j + k + 1, 0].set_axis_off()
                ax[2 * j + k + 1, 0].text(
                    1,
                    0.5,
                    f"T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}",
                    va="center",
                    ha="right",
                )

    return fig, ax


def get_beam_fraction_hist2d(hist2d, fraction: float):
    levels = np.linspace(hist2d.max(), 0.0, 100)
    total = hist2d.sum()
    final_beam = np.copy(hist2d)
    for level in levels:
        test_beam = np.where(hist2d >= level, hist2d, 0.0)
        test_frac = test_beam.sum() / total
        if test_frac > fraction:
            final_beam = test_beam
            break

    return final_beam


def plot_test_vs_pred_2screens(
    test_dset, pred_dset, select_img=0, contour_percentiles=[50, 95]
):
    """
    sadf

    Parameters
    ----------
    test_dset: ImageDataset
        test data.
        dset.images should be a 6D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of images per parameter configuration,
        screen width in pixels,
        screen height in pixels]
        dset.params should be a 4D tensor of shape
        [number of quad strengths,
        number of tdc voltages (2, off/on),
        number of dipole angles (2, off/on),
        number of scanning elements (3: quad, tdc, dipole) ]

    select_img: int
        index of image to plot for each parameter configuration

    Returns
    -------
    fig: matplotlib figure
        figure object
    """
    params = pred_dset.params
    imgs = pred_dset.images[:, :, :, select_img, :, :]
    test_imgs = test_dset.images[:, :, :, select_img, :, :]
    n_k = params.shape[0]
    n_v = params.shape[1]
    n_g = params.shape[2]
    fig, ax = plt.subplots(
        n_v * n_g + 1, n_k + 1, figsize=((n_k + 1) * 2, (n_v * n_g + 1) * 2)
    )
    ax[0, 0].set_axis_off()
    ax[0, 0].text(1, 0, "$k_1$ (1/m$^2$)", va="bottom", ha="right")
    for i in range(n_k):
        ax[0, i + 1].set_axis_off()
        ax[0, i + 1].text(0.5, 0, f"{params[i, 0, 0, 0]:.1f}", va="bottom", ha="center")
        for j in range(n_g):
            for k in range(n_v):
                ax[2 * j + k + 1, i + 1].imshow(
                    imgs[i, k, j].T, origin="lower", interpolation="none"
                )

                ax[2 * j + k + 1, i + 1].tick_params(
                    bottom=False, left=False, labelbottom=False, labelleft=False
                )

                if k == 0:
                    v_lbl = "off"
                else:
                    v_lbl = "on"
                if j == 0:
                    g_lbl = "off"
                else:
                    g_lbl = "on"

                ax[2 * j + k + 1, 0].set_axis_off()
                ax[2 * j + k + 1, 0].text(
                    1,
                    0.5,
                    f"T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}",
                    va="center",
                    ha="right",
                )

    return fig, ax


# --------------------------------------------------------------------------


def add_image_projection(ax, image, bins, axis, scale_x=1, c="r"):
    if axis == "x":
        proj = image.sum(dim=-1)
        proj = proj

        # get proj stats
        mean_proj = torch.mean(proj, dim=0)
        l = torch.quantile(proj, 0.05, dim=0)
        u = torch.quantile(proj, 0.95, dim=0)

        ax.plot(bins * scale_x, mean_proj, c)
        ax.fill_between(bins * scale_x, l, u, alpha=0.75, fc=c)
        # axb.set_yticks([])
        # axb.set_xticks([])

    elif axis == "y":
        proj = image.sum(dim=-2)
        proj = proj

        mean_proj = torch.mean(proj, dim=0)
        l = torch.quantile(proj, 0.05, dim=0)
        u = torch.quantile(proj, 0.95, dim=0)

        ax.plot(mean_proj, bins * scale_x, c)
        ax.fill_betweenx(bins * scale_x, l, u, alpha=0.75, fc=c)

        # axb.set_yticks([])
        # axb.set_xticks([])
    else:
        raise RuntimeError()

    return ax


def compare_images(xx, predicted_images, train_images):
    n_images = len(predicted_images)
    fig, ax = plt.subplots(n_images, 2, sharey="all", sharex="all")

    vmin = 0
    vmax = max(predicted_images.max(), train_images.max())

    for i in range(n_images):
        ax[i, 0].pcolor(*xx, train_images[i].cpu().detach(), vmin=vmin, vmax=vmax)
        ax[i, 1].pcolor(*xx, predicted_images[i].cpu().detach(), vmin=vmin, vmax=vmax)

        # add_image_projection(ax[i, 0], train_images[i].cpu().detach(), xx, "x")
        # add_image_projection(ax[i, 0], train_images[i].cpu().detach(), xx, "y")

        # add_image_projection(ax[i, 1], predicted_images[i].cpu().detach(), xx, "x")
        # add_image_projection(ax[i, 1], predicted_images[i].cpu().detach(), xx, "y")

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

    concentration = model_image_mean**2 / model_image_variance
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


def add_projection(ax, key, beams, bins, axis="x", x_scale=1, y_scale=1):
    histograms = []
    for ele in beams:
        histograms += [
            np.histogram(
                getattr(ele, key).cpu().detach().numpy(), bins=bins.cpu(), density=True
            )[0]
        ]

    histograms = np.asfarray(histograms)

    means = np.mean(histograms, axis=0) * y_scale
    l = np.quantile(histograms, 0.05, axis=0) * y_scale
    u = np.quantile(histograms, 0.95, axis=0) * y_scale

    if axis == "x":
        ax.plot(bins[:-1].cpu() * x_scale, means, label=key)
        ax.fill_between(bins[:-1].cpu() * x_scale, l, u, alpha=0.5)
    elif axis == "y":
        ax.plot(means, bins[:-1].cpu() * x_scale, label=key)
        ax.fill_betweenx(bins[:-1].cpu() * x_scale, l, u, alpha=0.5)
    else:
        raise RuntimeError()

    return ax


def add_image(ax, key1, key2, beams, bins, scale_axis=1, vmax=None):
    histograms = []
    xx = np.meshgrid(bins[0], bins[1])

    for ele in beams:
        histograms += [
            np.histogram2d(
                getattr(ele, key1).cpu().detach().numpy(),
                getattr(ele, key2).cpu().detach().numpy(),
                bins=bins,
                density=True,
            )[0].T
        ]

    histograms = np.asfarray(histograms)

    means = np.mean(histograms, axis=0)
    l = np.quantile(histograms, 0.05, axis=0)
    u = np.quantile(histograms, 0.95, axis=0)

    ax.pcolor(xx[0] * scale_axis, xx[1] * scale_axis, means, vmin=0, vmax=vmax)

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


def add_ellipse(ax, mean, cov, n_std=1, edgecolor="r"):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor="none",
        edgecolor=edgecolor,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
