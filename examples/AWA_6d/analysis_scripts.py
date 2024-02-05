import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from phase_space_reconstruction.modeling import ImageDataset3D


def plot_3d_scan_data_2screens(dset, select_img = 'avg', vmax1=None, vmax2=None):
    if select_img == 'avg':
        imgs = dset.images.sum(dim=-3)
        imgs = imgs / dset.images.shape[-3]
    else:
        imgs = dset.images[:,:,:,select_img,:,:]
    params = dset.params
    n_k = params.shape[0]
    n_v = params.shape[1]
    n_g = params.shape[2]
    fig, ax = plt.subplots(
        n_v * n_g + 1,
        n_k + 1,
        figsize=( (n_k+1)*2, (n_v*n_g+1)*2 )
    )
    ax[0, 0].set_axis_off()
    ax[0, 0].text(1, 0, '$k_1$ (1/m$^2$)', va='bottom', ha='right')
    for i in range(n_k):
        ax[0, i + 1].set_axis_off()
        ax[0, i + 1].text(
            0.5, 0, f'{params[i, 0, 0, 0]:.1f}', va='bottom', ha='center'
        )
        for j in range(n_g):
            for k in range(n_v):
                if k == 0:
                    v_lbl = "off"
                else:
                    v_lbl = "on"
                if j == 0:
                    g_lbl = "off"
                    vmax = vmax1
                else:
                    g_lbl = "on"
                    vmax = vmax2
                    
                ax[2 * j + k + 1, i + 1].imshow(
                    imgs[i, k, j].T, 
                    origin='lower', 
                    interpolation='none',
                    vmin=0,
                    vmax=vmax
                )
                
                ax[2 * j + k + 1, i + 1].tick_params(
                    bottom=False, left=False,
                    labelbottom=False, labelleft=False
                )

                

                ax[2 * j + k + 1, 0].set_axis_off()
                ax[2 * j + k + 1, 0].text(
                    1, 0.5, f'T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}',
                    va='center', ha='right'
                )

    return fig, ax

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

def plot_3d_scan_data_2screens_contour(
    pred_dset, 
    test_dset, 
    select_img = 'avg', 
    contour_percentiles = [50, 95],
    contour_smoothing_r = 1,
    contour_smoothing_gt = 1,
    screen_0_len = None,
    screen_1_len = None,
    vmax1=None,
    vmax2=None
):

    n_contours = len(contour_percentiles)
    COLORS = ["white", "gray", "black"]
    COLORS = COLORS * (n_contours // int(len(COLORS)+0.1) + 1)
    pred_imgs = pred_dset.images[:,:,:,0,:,:]
    test_imgs = test_dset.images
    if select_img == 'avg':
        test_imgs_tmp = test_dset.images.sum(dim=-3)
        test_imgs = test_imgs_tmp / test_imgs.shape[-3]
    else:
        test_imgs = test_dset.images[:,:,:,select_img,:,:]
        
    params = pred_dset.params
    n_k = params.shape[0]
    n_v = params.shape[1]
    n_g = params.shape[2]
    fig, ax = plt.subplots(
        n_v * n_g + 1,
        n_k + 1,
        figsize=( (n_k+1)*2, (n_v*n_g+1)*2 )
    )
    ax[0, 0].set_axis_off()
    ax[0, 0].text(1, 0, '$k_1$ (1/m$^2$)', va='bottom', ha='right')
    corners=None
    centers=None
    if screen_0_len is not None:
        corners_0 = torch.linspace(-screen_0_len/2, screen_0_len/2, test_imgs.shape[-1]+1)*1e3
        corners_1 = torch.linspace(-screen_1_len/2, screen_1_len/2, test_imgs.shape[-1]+1)*1e3
    
    for i in range(n_k):
        ax[0, i + 1].set_axis_off()
        ax[0, i + 1].text(
            0.5, 0, f'{params[i, 0, 0, 0]:.1f}', va='bottom', ha='center'
        )
        for j in range(n_g):
            for k in range(n_v):
                if k == 0:
                    v_lbl = "off"
                else:
                    v_lbl = "on"
                if j == 0:
                    g_lbl = "off"
                    vmax=vmax1
                    if screen_0_len is not None:
                        corners = corners_0
                        centers = corners[:-1] + (corners[1]-corners[0])/2
                else:
                    g_lbl = "on"
                    vmax=vmax2
                    if screen_0_len is not None:
                        corners = corners_1
                        centers = corners[:-1] + (corners[1]-corners[0])/2
                '''
                ax[2 * j + k + 1, i + 1].imshow(
                    pred_imgs[i, k, j].T,
                    origin='lower', 
                    interpolation='none', 
                    vmin=0, 
                    vmax=vmax
                )
                '''
                if screen_0_len is not None:
                    ax[2 * j + k + 1, i + 1].pcolormesh(
                        corners,
                        corners,
                        pred_imgs[i, k, j].T, 
                        vmin=0, 
                        vmax=vmax
                    )
                else:
                    ax[2 * j + k + 1, i + 1].pcolormesh(
                        pred_imgs[i, k, j].T, 
                        vmin=0, 
                        vmax=vmax
                    )
                
                proj_y = pred_imgs[i, k, j].sum(axis=0)
                proj_y_gt = test_imgs[i, k, j].sum(axis=0)
                hist_y ,_ = np.histogram(proj_y)
                ax_y = ax[2 * j + k + 1, i + 1].twiny()
                if screen_0_len is not None:
                    bin_y = centers
                else:
                    bin_y = np.linspace(0, len(proj_y)-1, len(proj_y), dtype=int)
                
                ax_y.plot(proj_y, bin_y)
                ax_y.plot(proj_y_gt, bin_y)
                ax_y.set_xlim(0.0, proj_y.max()*5)
                ax_y.set_axis_off()
                
                #print(pred_imgs[i, k, j].sum())
                #print(test_imgs[i, k, j].sum())
                #print('----------------------')
                
                
                proj_x = pred_imgs[i, k, j].sum(axis=1)
                proj_x_gt = test_imgs[i, k, j].sum(axis=1)
                hist_x ,_ = np.histogram(proj_x)
                ax_x = ax[2 * j + k + 1, i + 1].twinx()
                if screen_0_len is not None:
                    bin_x = centers
                else:
                    bin_x = np.linspace(0, len(proj_x)-1, len(proj_x), dtype=int)
                
                ax_x.plot(bin_x, proj_x)
                ax_x.plot(bin_x, proj_x_gt)
                ax_x.set_ylim(0.0, proj_x.max()*5)
                ax_x.set_axis_off()
                
                
                
                for l, percentile in enumerate(contour_percentiles):
                    h_r_fractions = get_beam_fraction_hist2d(pred_imgs[i, k, j], percentile/100)
                    h_gt_fractions = get_beam_fraction_hist2d(test_imgs[i,k,j], percentile/100)
                    if screen_0_len is not None:
                        ax[2 * j + k + 1, i + 1].contour(
                            #h_r_fractions.T,
                            centers,
                            centers,
                            gaussian_filter(h_r_fractions, contour_smoothing_r).T,
                            levels=[0],
                            linestyles="-",
                            colors=COLORS[l],
                            linewidths=1
                        )  
                        ax[2 * j + k + 1, i + 1].contour(
                            #h_gt_fractions.T,
                            centers,
                            centers,
                            gaussian_filter(h_gt_fractions, contour_smoothing_gt).T,
                            levels=[0],
                            linestyles="--",
                            colors=COLORS[l],
                            linewidths=1
                        ) 
                    else:
                        ax[2 * j + k + 1, i + 1].contour(
                            #h_r_fractions.T,
                            gaussian_filter(h_r_fractions, contour_smoothing_r).T,
                            levels=[0],
                            linestyles="-",
                            colors=COLORS[l],
                            linewidths=1
                        )  
                        ax[2 * j + k + 1, i + 1].contour(
                            #h_gt_fractions.T,
                            gaussian_filter(h_gt_fractions, contour_smoothing_gt).T,
                            levels=[0],
                            linestyles="--",
                            colors=COLORS[l],
                            linewidths=1
                        ) 
                ax[2 * j + k + 1, i + 1].tick_params(
                    bottom=False, left=False,
                    labelbottom=False, labelleft=False
                )

                ax[2 * j + k + 1, 0].set_axis_off()
                ax[2 * j + k + 1, 0].text(
                    1, 0.5, f'T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}',
                    va='center', ha='right'
                )

    custom_lines = [Line2D([0], [0], color=COLORS[0], lw=1),
                    Line2D([0], [0], color=COLORS[0], lw=1, linestyle='--'),
                    Line2D([0], [0], color=COLORS[1], lw=1),
                    Line2D([0], [0], color=COLORS[1], lw=1, linestyle='--')]
    ax[1,1].legend(
        custom_lines, 
        ['prediction 50 %ile', 'measured 50 %ile', 'prediction 95 %ile', 'measured 95 %ile'],
        fontsize=5
    )
    for i in range(4):
        ax[-1, i+1].tick_params(
                    bottom=True,
                    labelbottom=True
        )
        ax[-3, i+1].tick_params(
                    bottom=True,
                    labelbottom=True
        )
        ax[i + 1, -1].tick_params(
                    right=True,
                    labelright=True
        )
    

    return fig, ax

def clip_imgs(imgs, center, width):
    half_width = width // 2
    return imgs[Ellipsis, center-half_width:center+half_width, center-half_width:center+half_width]

def create_clipped_dset(dset, width):
    imgs = dset.images
    params = dset.params
    center = imgs.shape[-1] // 2
    clipped_imgs = clip_imgs(imgs, center, width)
    return ImageDataset3D(params, clipped_imgs)

def run_3d_scan_2screens(
    beam,
    lattice0,
    lattice1,
    screen0,
    screen1,
    params,
    n_imgs_per_param = 1,
    ids = [0, 2, 4],
    save_as = None
):

    # base lattices 
    #params = torch.meshgrid(ks, vs, gs, indexing='ij')
    #params = torch.stack(params, dim=-1)
    print(params.shape)
    params_dipole_off = params[:,:,0].unsqueeze(-1)
    print(params_dipole_off.shape)
    diagnostics_lattice0 = lattice0.copy()
    diagnostics_lattice0.elements[ids[0]].K1.data = params_dipole_off[:,:,0]
    diagnostics_lattice0.elements[ids[1]].VOLTAGE.data = params_dipole_off[:,:,1]
    diagnostics_lattice0.elements[ids[2]].G.data = params_dipole_off[:,:,2]

    params_dipole_on = params[:,:,1].unsqueeze(-1)
    diagnostics_lattice1 = lattice1.copy()
    diagnostics_lattice1.elements[ids[0]].K1.data = params_dipole_on[:,:,0]
    diagnostics_lattice1.elements[ids[1]].VOLTAGE.data = params_dipole_on[:,:,1]
    diagnostics_lattice1.elements[ids[2]].G.data = params_dipole_on[:,:,2]

    # track through lattice for dipole off(0) and dipole on (1)
    output_beam0 = diagnostics_lattice0(beam)
    output_beam1 = diagnostics_lattice1(beam)

    # histograms at screens for dipole off(0) and dipole on (1)
    images_dipole_off = screen0(output_beam0).squeeze()
    images_dipole_on = screen1(output_beam1).squeeze()

    # stack on dipole dimension:
    images_stack = torch.stack((images_dipole_off, images_dipole_on), dim=2)
    
    # create images copies simulating multi-shot per parameter config:
    copied_images = torch.stack([images_stack]*n_imgs_per_param, dim=-3)

    # create image dataset
    dset = ImageDataset3D(params, copied_images)
    
    # save scan data if wanted
    if save_as is not None:
        torch.save(dset, save_as)
        print(f"dataset0 saved as '{save_as}'")

    return dset