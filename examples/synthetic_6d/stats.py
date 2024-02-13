import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from phase_space_reconstruction.analysis import get_beam_fraction_numpy_coords 
from copy import deepcopy

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
    
def scale_beam_coords(particles, scale_dict):
    """ return a copy of `particles` scaled by scale_dict"""
    particles_copy = deepcopy(particles)
    particles_copy.data = particles.data * torch.tensor(
        [scale_dict[ele] for ele in particles.keys]
    )
        
    return particles_copy
    
from stats import scale_beam_coords, get_beam_fraction_hist2d
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from phase_space_reconstruction.analysis import get_beam_fraction_numpy_coords 
from copy import deepcopy

def plot_projections_with_contours(
        reconstruction,
        ground_truth = None,
        contour_percentiles = [50, 95],
        contour_smoothing = 0.0,
        coords = ('x', 'px', 'y', 'py', 'z', 'pz'),
        n_bins = 200,
        background = 0,
        same_lims = False,
        custom_lims = None,
        use_pz_percentage_units = True,
        scale=1e3,
        ):    
    
    SPACE_COORDS = ('x', 'y', 'z')
    MOMENTUM_COORDS = ('px', 'py', 'pz')

    # set up plot objects
    n_coords = len(coords)
    n_contours = len(contour_percentiles)
    COLORS = ["white", "gray", "black"]
    COLORS = COLORS * (n_contours // int(len(COLORS)+0.1) + 1)
    fig_size = (n_coords*2,) * 2

    fig, ax = plt.subplots(n_coords, n_coords, figsize=fig_size, dpi=300, sharex="col")
    mycmap = plt.get_cmap('viridis')
    mycmap.set_under(color='white') # map 0 to this color

    # scale beam distribution to correct units
    scale_dict = {ele:scale for ele in coords}
    if use_pz_percentage_units:
        scale_dict["pz"] = 1e2
        
    reconstruction = scale_beam_coords(reconstruction, scale_dict)
    if ground_truth is not None:
        ground_truth = scale_beam_coords(ground_truth, scale_dict)
    
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
            
        if x_coord == "pz" and use_pz_percentage_units:
            x_coord_unit = "%"

        x_array = getattr(reconstruction, x_coord).numpy()
        ax[n_coords-1,i].set_xlabel(f'{x_coord} ({x_coord_unit})')
        min_x = coord_min[i]
        max_x = coord_max[i]

        if i>0:
            l = x_coord
            if "p" in x_coord:
                l = f"$p_{x_coord[-1]}$"
            ax[i,0].set_ylabel(f'{l} ({x_coord_unit})')

        print(min_x, max_x)
        h, bins = np.histogram(
            x_array,
            range=(float(min_x), float(max_x)),
            bins=int(n_bins),
            density=True,
        )
        binc = (bins[:-1] + bins[1:])/2

        ax[i,i].plot(
            binc,h,"C1--",alpha=1, lw=2,zorder=5
        )
        #ax[i,i].set_ylim(0,1.1*np.max(h))
        
        if ground_truth is not None:
            h,bins=np.histogram(
                getattr(ground_truth, x_coord),
                range=(float(min_x), float(max_x)),
                bins=int(n_bins),
                density=True,
            )
            
            binc = (bins[:-1] + bins[1:])/2
            ax[i,i].plot(
                binc,h,"C0-",alpha=1,lw=2
            )
        
        ax[i,i].yaxis.set_tick_params(left=False, labelleft=False)

        if i!= n_coords-1:
            ax[i,i].xaxis.set_tick_params(labelbottom=False)

        for j in range(i+1, n_coords):
            y_coord = coords[j]
            y_array = getattr(reconstruction, y_coord)
            min_y = coord_min[j]
            max_y = coord_max[j]
            rng=[[min_x, max_x],[min_y, max_y]]
            print(x_coord, y_coord)
            print(rng)
            
            hist, x_edges, y_edges, _ = ax[j,i].hist2d(
                x_array,
                y_array,
                bins=int(n_bins),
                range = rng,
                cmap = mycmap,
                vmin = background,
                rasterized=True
            )
            
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2

            for k, percentile in enumerate(contour_percentiles):
                h_r_fractions = get_beam_fraction_hist2d(hist, percentile/100)
                ax[j,i].contour(
                    x_centers, 
                    y_centers, 
                    medfilt(
                        gaussian_filter(h_r_fractions, contour_smoothing), 5
                    ).T,                    
                    #h_r_fractions.T,
                    levels=[1],
                    linestyles="--",
                    colors=COLORS[k],
                    linewidths=1,
                    zorder=10
                )

                if ground_truth is not None:
                    h_gt, _, _ = np.histogram2d(
                        getattr(ground_truth, x_coord),
                        getattr(ground_truth, y_coord),
                        bins = int(n_bins),
                        range = rng
                    )
                    h_gt_fractions = get_beam_fraction_hist2d(h_gt, percentile/100)

                    ax[j,i].contour(
                        x_centers, 
                        y_centers, 
                        medfilt(
                            gaussian_filter(h_gt_fractions, contour_smoothing), 5
                        ).T,
                        #h_gt_fractions.T,
                        levels=[1],
                        linestyles="-",
                        colors=COLORS[k],
                        linewidths=1,
                    )  

            ax[j,i].set_xlim(min_x, max_x)
            ax[j,i].set_ylim(min_y, max_y)
            #ax[i,i].set_xlim(min_x, max_x)

            ax[i,j].set_visible(False)

            if i != 0:
                ax[j, i].yaxis.set_tick_params(labelleft=False)
            
            if j != n_coords-1:
                ax[j,i].xaxis.set_tick_params(labelbottom=False)
        #ax[i,i].set_xlim(min_x, max_x)
        
    fig.tight_layout()

    return fig, ax