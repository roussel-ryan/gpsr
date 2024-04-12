import os
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch
from phase_space_reconstruction.analysis import get_beam_fraction_numpy_coords
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt


def read_reconstructed_particle(dr, i):
    fname = f"r_{i}.pt"
    particle = torch.load(os.path.join(dr, fname)).numpy_particles()

    return particle


def read_all_particles(dr, n_beams, n_par):
    all_particles = np.zeros([n_beams, 6, n_par])

    for i in range(0, n_beams):
        particle = read_reconstructed_particle(dr, i + 1)
        all_particles[i] = particle[:6]

    return all_particles


def get_cov(particle, beam_fraction=1.0):
    par_frac = get_beam_fraction_numpy_coords(particle, beam_fraction)
    return np.cov(par_frac[:6])


def get_all_covs(all_particles, beam_fraction=1.0):
    all_cov = np.zeros([len(all_particles), 6, 6])

    for i in range(len(all_particles)):
        all_cov[i] = get_cov(all_particles[i], beam_fraction)

    cov_avg = all_cov.mean(axis=0)
    cov_std = all_cov.std(axis=0)

    return all_cov, cov_avg, cov_std


def get_cov_discrepancy(rec_avg, rec_std, gt):
    cov_sigmas = (rec_avg - gt) / rec_std
    return cov_sigmas


def plot_cov_sigmas(cov_sigmas):
    coords = ("x", "px", "y", "py", "z", "pz")
    fig, ax = plt.subplots()
    c = ax.imshow(cov_sigmas, cmap="seismic", vmin=-3, vmax=3, alpha=0.5)
    for (j, i), label in np.ndenumerate(cov_sigmas):
        ax.text(i, j, f"{label:.3f}", ha="center", va="center")
    fig.colorbar(c)
    ax.set_xticks(np.arange(len(coords)), labels=coords)
    ax.set_yticks(np.arange(len(coords)), labels=coords)


def show_cov_stats(pars, gt, beam_fraction):
    cov_gt_frac = get_cov(gt, beam_fraction=beam_fraction)
    covs_frac, cov_avg_frac, cov_std_frac = get_all_covs(
        pars, beam_fraction=beam_fraction
    )

    print(f"ground truth: \n{cov_gt_frac*1e6}\n")
    print(f"reconstruction avg: \n{cov_avg_frac*1e6}\n")
    print(f"reconstruction std: \n{cov_std_frac*1e6}\n")
    print(f"reconstruction relative uncertainty: \n{cov_std_frac/cov_avg_frac}")
    cov_sigmas_frac = get_cov_discrepancy(cov_avg_frac, cov_std_frac, cov_gt_frac)
    plot_cov_sigmas(cov_sigmas_frac)
    plt.show()


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


def scale_beam_coords(particles, scale_dict):
    """return a copy of `particles` scaled by scale_dict"""
    particles_copy = deepcopy(particles)
    particles_copy.data = particles.data * torch.tensor(
        [scale_dict[ele] for ele in particles.keys]
    )

    return particles_copy


def get_coord_unit_and_label(name, scale, use_pz_percentage_units=True):
    SPACE_COORDS = ("x", "y", "z")
    MOMENTUM_COORDS = ("px", "py", "pz")
    
    if name in SPACE_COORDS and scale == 1e3:
        unit = "mm"
    elif name in SPACE_COORDS and scale == 1:
        unit = "m"
    elif name in MOMENTUM_COORDS and scale == 1e3:
        unit = "mrad"
    elif name in MOMENTUM_COORDS and scale == 1:
        unit = "rad"
    else:
        raise ValueError(
            """scales should be 1 or 1e3,
        coords should be a subset of ('x', 'px', 'y', 'py', 'z', 'pz')
        """
        )

    if name == "pz" and use_pz_percentage_units:
        unit = "%"

    l = name
    if "p" in name:
        l = f"$p_{name[-1]}$"
        
    return unit, l
    

def plot_single_projection_with_contours(
    reconstruction, x_coord:str, y_coord:str, 
    ground_truth=None, 
    hist_range=None, 
    n_bins=20,
    contour_percentiles=[50,90],
    ax=None,
    background=0,
    contour_smoothing=0.0,
    median_filter_size=5,
    use_pz_percentage_units=True,

):
    n_contours = len(contour_percentiles)

    COLORS = ["white", "gray", "black"]
    COLORS = COLORS * (n_contours // int(len(COLORS) + 0.1) + 1)
    if ax is None:
        fig,ax = plt.subplots()
        
    
    xunit, xlabel = get_coord_unit_and_label(x_coord, 1e3,use_pz_percentage_units)
    yunit, ylabel = get_coord_unit_and_label(y_coord, 1e3,use_pz_percentage_units)
    
    # scale beam distribution to correct units
    scale_dict = {ele: 1e3 for ele in ("x", "px", "y", "py", "z", "pz")}
    if use_pz_percentage_units:
        scale_dict["pz"] = 1e2

    reconstruction = scale_beam_coords(reconstruction, scale_dict)
    if ground_truth is not None:
        ground_truth = scale_beam_coords(ground_truth, scale_dict)

    ax.set_xlabel(f"{xlabel} ({xunit})")
    ax.set_ylabel(f"{ylabel} ({yunit})")

    x_array = getattr(reconstruction, x_coord)
    y_array = getattr(reconstruction, y_coord)

    hist, x_edges, y_edges, _ = ax.hist2d(
                x_array,
                y_array,
                bins=int(n_bins),
                range=hist_range,
                vmin=background,
                rasterized=True,
    )

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    for k, percentile in enumerate(contour_percentiles):
        h_r_fractions = get_beam_fraction_hist2d(hist, percentile / 100)
        ax.contour(
            x_centers,
            y_centers,
            medfilt(
                gaussian_filter(h_r_fractions, contour_smoothing),
                median_filter_size,
            ).T,
            # h_r_fractions.T,
            levels=[1],
            linestyles="--",
            colors=COLORS[k],
            linewidths=1,
            zorder=10,
            )

        if ground_truth is not None:
            h_gt, _, _ = np.histogram2d(
                getattr(ground_truth, x_coord),
                getattr(ground_truth, y_coord),
                bins=int(n_bins),
                range=hist_range,
            )
            h_gt_fractions = get_beam_fraction_hist2d(h_gt, percentile / 100)

            ax.contour(
                x_centers,
                y_centers,
                medfilt(
                    gaussian_filter(h_gt_fractions, contour_smoothing),
                    median_filter_size,
                ).T,
                # h_gt_fractions.T,
                levels=[1],
                linestyles="-",
                colors=COLORS[k],
                linewidths=1,
                )

    ax.set_xlim(*hist_range[0])
    ax.set_ylim(*hist_range[1])
    
    return ax


def plot_projections_with_contours(
    reconstruction,
    ground_truth=None,
    contour_percentiles=[50, 95],
    contour_smoothing=0.0,
    coords=("x", "px", "y", "py", "z", "pz"),
    n_bins=200,
    background=0,
    same_lims=False,
    custom_lims=None,
    use_pz_percentage_units=True,
    median_filter_size=5,
):

    # set up plot objects
    n_coords = len(coords)

    fig_size = (n_coords * 2,) * 2

    fig, ax = plt.subplots(
        n_coords, n_coords, figsize=fig_size, dpi=300, sharex="col",
        gridspec_kw={"left":0.1,"bottom":0.075,"top":0.975}
    )

    all_coords = []

    for coord in coords:
        all_coords.append(getattr(reconstruction, coord))

    all_coords = np.array(all_coords)

    if same_lims:
        if custom_lims is None:
            coord_min = np.ones(n_coords) * all_coords.min()
            coord_max = np.ones(n_coords) * all_coords.max()
        elif len(custom_lims) == 2:
            coord_min = np.ones(n_coords) * custom_lims[0]
            coord_max = np.ones(n_coords) * custom_lims[1]
        else:
            raise ValueError("custom lims should have shape 2 when same_lims=True")
    else:
        if custom_lims is None:
            coord_min = all_coords.min(axis=1)
            coord_max = all_coords.max(axis=1)
        elif custom_lims.shape == (n_coords, 2):
            coord_min = custom_lims[:, 0]
            coord_max = custom_lims[:, 1]
        else:
            raise ValueError(
                "custom lims should have shape (n_coords x 2) when same_lims=False"
            )

    for i in range(n_coords):
        x_coord = coords[i]
        min_x = coord_min[i]
        max_x = coord_max[i]
        x_array = getattr(reconstruction, x_coord)

        scale = 1e2 if x_coord=="pz" else 1e3
        h, bins = np.histogram(
            x_array*scale,
            range=(float(min_x), float(max_x)),
            bins=int(n_bins),
            density=True,
        )
        binc = (bins[:-1] + bins[1:]) / 2

        ax[i, i].plot(binc, h, "C1--", alpha=1, lw=2, zorder=5)
        try:
            ax[i, i].set_ylim(0, 1.1 * np.max(h))
        except ValueError:
            pass

        if ground_truth is not None:
            h, bins = np.histogram(
                getattr(ground_truth, x_coord)*scale,
                range=(float(min_x), float(max_x)),
                bins=int(n_bins),
                density=True,
            )

            binc = (bins[:-1] + bins[1:]) / 2
            ax[i, i].plot(binc, h, "C0-", alpha=1, lw=2)

        ax[i, i].yaxis.set_tick_params(left=False, labelleft=False)

        if i != n_coords - 1:
            ax[i, i].xaxis.set_tick_params(labelbottom=False)

        for j in range(i + 1, n_coords):
            min_y = coord_min[j]
            max_y = coord_max[j]
            hist_range = [[min_x, max_x], [min_y, max_y]]
            plot_single_projection_with_contours(
                reconstruction,
                coords[i],
                coords[j],
                ground_truth=ground_truth, 
                hist_range=hist_range, 
                n_bins=n_bins,
                contour_percentiles=contour_percentiles,
                ax=ax[j,i],
                background=background,
                contour_smoothing=contour_smoothing,
                median_filter_size=median_filter_size,
                use_pz_percentage_units=use_pz_percentage_units,
            )
            
            
            ax[i, j].set_visible(False)

            if i != 0:
                ax[j, i].yaxis.set_tick_params(labelleft=False)
                ax[j, i].set_ylabel(None)

            if j != n_coords - 1:
                ax[j, i].xaxis.set_tick_params(labelbottom=False)
                ax[j, i].set_xlabel(None)
        # ax[i,i].set_xlim(min_x, max_x)

    #fig.tight_layout()

    return fig, ax


def plot_cov_sigmas(cov_gt, cov_reconstruction, fig=None):
    # calculate fractional errors
    frac_error = np.abs((cov_gt - cov_reconstruction)/cov_gt)

    # mask data that should not be plotted (lower triangular portion of the matrix)
    mask = np.tri(frac_error.shape[0],k=-1)
    frac_error = np.flipud(np.ma.array(frac_error, mask=mask))

    # flip orientation for plotting
    manipulated_cov_gt = np.flipud(cov_gt).T
    manipulated_cov_recon = np.flipud(cov_reconstruction).T
    
    coords = ('x', '$p_x$', 'y', '$p_y$', 'z', '$p_z$')
    
    if fig is None:
        fig, ax = plt.subplots()
        cax = fig.add_axes([0, 0, 0.1, 0.1],zorder=0)
    else:
        centerx = 0.325
        centery = 0.425
        cax_width = 0.025
        buffer = 0.05
        ax_width = 1-centerx-cax_width - buffer*3
        ax = fig.add_axes([centerx, centery, ax_width, ax_width],zorder=0)
        cax = fig.add_axes([centerx + ax_width + buffer, centery, cax_width, ax_width],zorder=0)
    
    test = np.ma.filled(frac_error,1e-3)
    c = ax.pcolormesh(frac_error, norm=colors.LogNorm(frac_error.min(),frac_error.max()),cmap="Blues",alpha=0.25)
    for (j,i), label in np.ndenumerate(frac_error):
        if not frac_error.mask[i,j]:
            ax.text(i + 0.5, j + 0.6, f'{manipulated_cov_gt[i,j]*1e6:.2f}', ha='center', va='center',zorder=10)
            ax.text(i + 0.5, j + 0.4, f'{manipulated_cov_recon[i,j]*1e6:.2f}', ha='center', va='center',zorder=10,font={"weight":"bold"})

    fig.colorbar(c,cax=cax,label="Covariance fractional error")
    ax.set_xticks(np.arange(len(coords))+0.5, labels=coords)
    ax.set_yticks(np.arange(len(coords))+0.5, labels=coords[::-1])
    
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #fig.tight_layout()
    
    ax.set_title("Covariance prediction comparison")

    return fig, ax 

def plot_prab_figure(
    reconstruction, gt_beam, *args, **kwargs
):
    fig,ax = plot_projections_with_contours(
        reconstruction, ground_truth=gt_beam, **kwargs
    )
    
    # add covariance comparison
    frac = 0.9
    cov_gt_frac = get_cov(gt_beam.numpy_particles(), beam_fraction=frac)
    cov_reconstruction_frac = get_cov(reconstruction.numpy_particles(), beam_fraction=frac)
    
    fig, ax = plot_cov_sigmas(cov_gt_frac,cov_reconstruction_frac, fig)
    a = fig.get_axes()[:-10]
    for ele in a:
        ele.zorder=10

    fig.set_size_inches(7,7)
    
    return fig