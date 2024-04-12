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
        figsize=( (n_k+1)*2, (n_v*n_g+1)*2 ),
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
    vmax2=None,
    rasterized = True
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
        n_v * n_g,
        n_k,
        figsize=( (n_k)*2, (n_v*n_g)*2 ),
        sharex="row",
        sharey="row",
    )
    ax[0, 0].text(-0.1, 1.1, '$k_1$ (1/m$^2$)', va='bottom', ha='right',
                  transform=ax[0, 0].transAxes,)
    corners=None
    centers=None
    if screen_0_len is not None:
        corners_0 = torch.linspace(-screen_0_len/2, screen_0_len/2, test_imgs.shape[-1]+1)*1e3
        corners_1 = torch.linspace(-screen_1_len/2, screen_1_len/2, test_imgs.shape[-1]+1)*1e3
    
    for i in range(n_k):
        ax[0, i].text(
            0.5, 1.1, f'{params[i, 0, 0, 0]:.1f}', va='bottom', ha='center',
            transform=ax[0, i].transAxes,

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
                    ax[2 * j + k, i].pcolormesh(
                        corners,
                        corners,
                        pred_imgs[i, k, j].T, 
                        vmin=0, 
                        vmax=vmax,
                        rasterized=rasterized
                    )
                else:
                    ax[2 * j + k, i].pcolormesh(
                        pred_imgs[i, k, j].T, 
                        vmin=0, 
                        vmax=vmax,
                        rasterized=rasterized
                    )
                                    
                proj_y = pred_imgs[i, k, j].sum(axis=0)
                proj_y_gt = test_imgs[i, k, j].sum(axis=0)
                hist_y ,_ = np.histogram(proj_y)
                ax_y = ax[2 * j + k, i].twiny()
                if screen_0_len is not None:
                    bin_y = centers
                else:
                    bin_y = np.linspace(0, len(proj_y)-1, len(proj_y), dtype=int)
                
                ax_y.plot(proj_y_gt, bin_y,"C0")
                ax_y.plot(proj_y, bin_y,"C1--")

                ax_y.set_xlim(0.0, proj_y.max()*6)
                ax_y.set_axis_off()
                
                
                
                proj_x = pred_imgs[i, k, j].sum(axis=1)
                proj_x_gt = test_imgs[i, k, j].sum(axis=1)
                hist_x ,_ = np.histogram(proj_x)
                ax_x = ax[2 * j + k, i].twinx()
                if screen_0_len is not None:
                    bin_x = centers
                else:
                    bin_x = np.linspace(0, len(proj_x)-1, len(proj_x), dtype=int)
                
                ax_x.plot(bin_x, proj_x_gt,"C0")
                ax_x.plot(bin_x, proj_x,"C1--")
                ax_x.set_ylim(0.0, proj_x.max()*6)
                ax_x.set_axis_off()
                
                
                
                for l, percentile in enumerate(contour_percentiles):
                    h_r_fractions = get_beam_fraction_hist2d(pred_imgs[i, k, j], percentile/100)
                    h_gt_fractions = get_beam_fraction_hist2d(test_imgs[i,k,j], percentile/100)
                    if screen_0_len is not None:
                        ax[2 * j + k, i].contour(
                            #h_r_fractions.T,
                            centers,
                            centers,
                            gaussian_filter(h_r_fractions, contour_smoothing_r).T,
                            levels=[0],
                            linestyles="--",
                            colors=COLORS[l],
                            linewidths=1
                        )  
                        ax[2 * j + k, i].contour(
                            #h_gt_fractions.T,
                            centers,
                            centers,
                            gaussian_filter(h_gt_fractions, contour_smoothing_gt).T,
                            levels=[0],
                            linestyles="-",
                            colors=COLORS[l],
                            linewidths=1
                        ) 
                    else:
                        ax[2 * j + k, i].contour(
                            #h_r_fractions.T,
                            gaussian_filter(h_r_fractions, contour_smoothing_r).T,
                            levels=[0],
                            linestyles="--",
                            colors=COLORS[l],
                            linewidths=1
                        )  
                        ax[2 * j + k, i].contour(
                            #h_gt_fractions.T,
                            gaussian_filter(h_gt_fractions, contour_smoothing_gt).T,
                            levels=[0],
                            linestyles="-",
                            colors=COLORS[l],
                            linewidths=1
                        ) 
                #ax[2 * j + k + 1, i + 1].tick_params(
                #    bottom=False, left=False,
                #    labelbottom=False, labelleft=False
                #)

                #ax[2 * j + k + 1, 0].set_axis_off()
                if i == 0:
                    ax[2 * j + k, 0].text(
                        -0.6, 0.5, 
                        f'T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}',
                        va='center', ha='right', 
                        transform=ax[2 * j + k, 0].transAxes,
                    )
    
    for a in ax[:,0]:
        a.set_ylabel("$y$ (mm)")
        
    for a in ax[-1,:]:
        a.set_xlabel("$x$ (mm)")
        
    for a in ax[::2,:].flatten():
        a.set_xticklabels([])
    
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
