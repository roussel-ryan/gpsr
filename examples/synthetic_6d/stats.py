import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from phase_space_reconstruction.analysis import get_beam_fraction_numpy_coords


def read_reconstructed_particle(dr, i):
    
    fname = f'r_{i}.pt'
    particle = torch.load(os.path.join(dr, fname)).numpy_particles()
    
    return particle


def read_all_particles(dr, n_beams, n_par):
    
    all_particles = np.zeros([n_beams, 6, n_par])
    
    for i in range(0, n_beams):
        particle = read_reconstructed_particle(dr, i+1)
        all_particles[i] = particle[:6]
    
    return all_particles

def get_cov(particle, beam_fraction=1.0):
    par_frac = get_beam_fraction_numpy_coords(
            particle, 
            beam_fraction
        )
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
    coords = ('x', 'px', 'y', 'py', 'z', 'pz')
    fig, ax = plt.subplots()
    c = ax.imshow(cov_sigmas, cmap='seismic', vmin=-3, vmax=3, alpha=0.5)
    for (j,i), label in np.ndenumerate(cov_sigmas):
        ax.text(i, j, f'{label:.3f}', ha='center', va='center')
    fig.colorbar(c)
    ax.set_xticks(np.arange(len(coords)), labels=coords)
    ax.set_yticks(np.arange(len(coords)), labels=coords)

def show_cov_stats(pars, gt, beam_fraction):
    cov_gt_frac = get_cov(gt, beam_fraction=beam_fraction)
    covs_frac, cov_avg_frac, cov_std_frac = get_all_covs(
        pars, beam_fraction=beam_fraction)

    print(f'ground truth: \n{cov_gt_frac*1e6}\n')
    print(f'reconstruction avg: \n{cov_avg_frac*1e6}\n')
    print(f'reconstruction std: \n{cov_std_frac*1e6}\n')
    print(f'reconstruction relative uncertainty: \n{cov_std_frac/cov_avg_frac}')
    cov_sigmas_frac = get_cov_discrepancy(
        cov_avg_frac, 
        cov_std_frac, 
        cov_gt_frac
    )
    plot_cov_sigmas(cov_sigmas_frac)
    plt.show()

def get_beam_fraction_hist2d(hist2d, fraction: float):
    levels = np.linspace(hist2d.max(), 0.0, 100)
    total = hist2d.sum()
    final_beam = np.copy(hist2d)
    for level in levels:
        test_beam = np.where(hist2d>=level, hist2d, 0.0)
        test_frac = test_beam.sum() / total
        if test_frac > fraction:
            final_beam = test_beam
            break

    return final_beam
    
def plot_projections_with_contours(
        reconstruction,
        ground_truth = None,
        contour_percentiles = [50, 95],
        contour_smoothing_r = 1,
        contour_smoothing_gt = 1,
        coords = ('x', 'px', 'y', 'py', 'z', 'pz'),
        bins = 200,
        scale = 1e3,
        background = 0,
        same_lims = False,
        custom_lims = None
        ):
    
    SPACE_COORDS = ('x', 'y', 'z')
    MOMENTUM_COORDS = ('px', 'py', 'pz')

    n_coords = len(coords)
    n_contours = len(contour_percentiles)
    COLORS = ["white", "gray", "black"]
    COLORS = COLORS * (n_contours // int(len(COLORS)+0.1) + 1)
    fig_size = (n_coords*2,) * 2

    fig, ax = plt.subplots(n_coords, n_coords, figsize=fig_size, dpi=300)
    mycmap = plt.get_cmap('viridis')
    mycmap.set_under(color='white') # map 0 to this color

    all_coords = []
    
    for coord in coords:
        all_coords.append(getattr(reconstruction, coord))
    
    all_coords = np.array(all_coords)
    
    if same_lims:
        if custom_lims is None:
            coord_min = np.ones(n_coords)*all_coords.min()
            coord_max = np.ones(n_coords)*all_coords.max()
        elif len(custom_lims) == 2:
            coord_min = np.ones(n_coords)*custom_lims[0]
            coord_max = np.ones(n_coords)*custom_lims[1]
        else:
            raise ValueError("custom lims should have shape 2 when same_lims=True")
    else:
        if custom_lims is None:
            coord_min = all_coords.min(axis=1)
            coord_max = all_coords.max(axis=1)
        elif custom_lims.shape == (n_coords, 2):
            coord_min = custom_lims[:,0]
            coord_max = custom_lims[:,1]
        else:
            raise ValueError("custom lims should have shape (n_coords x 2) when same_lims=False")

    for i in range(n_coords):
        x_coord = coords[i]

        if x_coord in SPACE_COORDS and scale==1e3:
            x_coord_unit = 'mm'
        elif x_coord in SPACE_COORDS and scale==1:
            x_coord_unit = 'm'
        elif x_coord in MOMENTUM_COORDS and scale==1e3:
            x_coord_unit = 'mrad'
        elif x_coord in MOMENTUM_COORDS and scale==1:
            x_coord_unit = 'rad'
        else:
            raise ValueError("""scales should be 1 or 1e3,
            coords should be a subset of ('x', 'px', 'y', 'py', 'z', 'pz')
            """)

        x_array = getattr(reconstruction, x_coord)*scale
        ax[n_coords-1,i].set_xlabel(f'{x_coord} ({x_coord_unit})')
        min_x = coord_min[i]*scale
        max_x = coord_max[i]*scale

        if i>0:
            ax[i,0].set_ylabel(f'{x_coord} ({x_coord_unit})')

        ax[i,i].hist(
            x_array,
            bins=bins,
            range=([min_x, max_x]),
            density=True,
            histtype = 'step'
        )
        if ground_truth is not None:
            ax[i,i].hist(
                getattr(ground_truth, x_coord)*scale,
                bins = bins,
                range = ([min_x, max_x]),
                density = True,
                histtype = 'step'
            )
        
        ax[i,i].yaxis.set_tick_params(left=False, labelleft=False)

        if i!= n_coords-1:
            ax[i,i].xaxis.set_tick_params(labelbottom=False)

        for j in range(i+1, n_coords):
            y_coord = coords[j]
            y_array = getattr(reconstruction, y_coord)*scale
            min_y = coord_min[j]*scale
            max_y = coord_max[j]*scale
            rng=[[min_x, max_x],[min_y, max_y]]
            
            hist, x_edges, y_edges, _ = ax[j,i].hist2d(
                x_array,
                y_array,
                bins = bins,
                range = rng,
                cmap = mycmap,
                vmin = background
            )
            
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2

            for k, percentile in enumerate(contour_percentiles):
                h_r_fractions = get_beam_fraction_hist2d(hist, percentile/100)
                ax[j,i].contour(
                    x_centers, 
                    y_centers, 
                    gaussian_filter(h_r_fractions, contour_smoothing_r).T,
                    #h_r_fractions.T,
                    levels=[1],
                    linestyles="-",
                    colors=COLORS[k],
                    linewidths=1
                )

                if ground_truth is not None:
                    h_gt, _, _ = np.histogram2d(
                        getattr(ground_truth, x_coord)*scale,
                        getattr(ground_truth, y_coord)*scale,
                        bins = bins,
                        range = rng
                    )
                    h_gt_fractions = get_beam_fraction_hist2d(h_gt, percentile/100)

                    ax[j,i].contour(
                        x_centers, 
                        y_centers, 
                        gaussian_filter(h_gt_fractions, contour_smoothing_gt).T,
                        #h_gt_fractions.T,
                        levels=[1],
                        linestyles="--",
                        colors=COLORS[k],
                        linewidths=1
                    )  

            #ax[j,i].get_shared_x_axes().join(ax[j,i], ax[i,i])
            ax[i,j].set_visible(False)

            if i != 0:
                ax[j, i].yaxis.set_tick_params(labelleft=False)
            
            if j != n_coords-1:
                ax[j,i].xaxis.set_tick_params(labelbottom=False)

    fig.tight_layout()

    return fig, ax