import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_scan_data(
        train_dset,
        test_dset,
        bins_x,
        bins_y
        ):
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
    _, indices = torch.sort(all_k[:,0,0], dim=0, stable=True)
    sorted_k = all_k[indices]
    sorted_im = all_im[indices]
    sorted_is_test = is_test[indices]

    # plot
    n_k = len(sorted_k)
    imgs_per_k = sorted_im.shape[1]
    fig, ax = plt.subplots(imgs_per_k, n_k+1, figsize=(n_k+1, imgs_per_k))
    extent = np.array([bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]])

    if imgs_per_k==1:
        for i in range(n_k):
            ax[i+1].imshow(sorted_im[i,0].T,
                           origin = 'lower',
                           extent = extent,
                           interpolation = 'none')
            ax[i+1].tick_params(bottom=False, left=False,
                                labelbottom=False, labelleft=False)
            if sorted_is_test[i]:
                for spine in ax[i+1].spines.values():
                    spine.set_edgecolor('orange')
                    spine.set_linewidth(2)

            ax[i+1].set_title(f'{sorted_k[i,0,0]:.1f}')
            ax[0].set_axis_off()
            ax[0].text(0.5, 0.5, f'img 1', va='center', ha='center')
        
        ax[0].set_title('$k$ (1/m):')

    else:
        for i in range(n_k):
            for j in range(imgs_per_k):
                ax[j, i+1].imshow(sorted_im[i,j].T,
                                  origin = 'lower',
                                  extent = extent,
                                  interpolation = 'none')
                ax[j, i+1].tick_params(bottom=False, left=False,
                                       labelbottom=False, labelleft=False)
                if sorted_is_test[i]:
                    for spine in ax[j, i+1].spines.values():
                        spine.set_edgecolor('orange')
                        spine.set_linewidth(2)

            ax[0,i+1].set_title(f'{sorted_k[i,0,0]:.1f}')
        
        for j in range(imgs_per_k):
            ax[j,0].set_axis_off()
            ax[j,0].text(0.5, 0.5, f'img {j+1}', va='center', ha='center')
        
        ax[0,0].set_title('$k$ (1/m$^2$):')

        
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    print(f'image size = {(bins_x[-1]-bins_x[0])*1e3:.0f} x {(bins_y[-1]-bins_y[0])*1e3:.0f} mm')
    print('test samples boxed in orange')
    
    return fig, ax


def plot_predicted_screens(
        prediction_dset,
        train_dset,
        test_dset,
        bins_x,
        bins_y
        ):
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
    _, indices = torch.sort(prediction_dset.k[:,0,0], dim=0, stable=True)
    sorted_pred = prediction_dset.images[indices]

    # id is zero if sample is train and 1 if test
    train_ids = torch.zeros(len(train_dset.k), dtype=torch.bool)
    test_ids = torch.ones(len(test_dset.k), dtype=torch.bool)
    is_test = torch.hstack((train_ids, test_ids))

    # stack training and tests data
    all_k = torch.vstack((train_dset.k, test_dset.k))
    all_im = torch.vstack((train_dset.images, test_dset.images))

    # sort by k value
    _, indices = torch.sort(all_k[:,0,0], dim=0, stable=True)
    sorted_k = all_k[indices]
    sorted_im = all_im[indices]
    sorted_is_test = is_test[indices]

    # plot
    n_k = len(sorted_k)
    imgs_per_k = sorted_im.shape[1]
    fig, ax = plt.subplots(imgs_per_k+1, n_k+1, figsize=(n_k+1, imgs_per_k+1))
    extent = np.array([bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]])

    for i in range(n_k):
        for j in range(imgs_per_k):
            ax[j, i+1].imshow(sorted_im[i,j].T,
                              origin = 'lower',
                              extent = extent,
                              interpolation = 'none')
            
            ax[j, i+1].tick_params(bottom=False, left=False,
                                   labelbottom=False, labelleft=False)
            
            if sorted_is_test[i]:
                for spine in ax[j, i+1].spines.values():
                    spine.set_edgecolor('orange')
                    spine.set_linewidth(2)

        ax[0,i+1].set_title(f'{sorted_k[i,0,0]:.1f}')

        ax[-1, i+1].imshow(sorted_pred[i,j].T,
                         origin = 'lower',
                         extent = extent,
                         interpolation = 'none')

        ax[-1, i+1].tick_params(bottom=False, left=False,
                                labelbottom=False, labelleft=False)
        
        if sorted_is_test[i]:
            for spine in ax[-1, i+1].spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(2)

        ax[-1,0].set_axis_off()
        ax[-1,0].text(0.5, 0.5, 'pred', va='center', ha='center')
    
    for j in range(imgs_per_k):
        ax[j,0].set_axis_off()
        ax[j,0].text(0.5, 0.5, f'img {j+1}', va='center', ha='center')
    
    ax[0,0].set_title('$k$ (1/m$^2$):')

        
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    print(f'image size = {(bins_x[-1]-bins_x[0])*1e3:.0f} x {(bins_y[-1]-bins_y[0])*1e3:.0f} mm')
    print('test samples boxed in orange')
    
    return fig, ax


def screen_stats(image, bins_x, bins_y):
    """
    Returns screen stats

    Parameters
    ----------
    image: 2D array-like
        screen image of size [n_x, n_y].

    bins_x: 1D array-like
        x axis bins physical locations of size [n_x]

    bins_y: 2D array-like
        x axis bins physical locations of size [n_y]

    Returns
    -------
    dictionary with 'avg_x', 'avg_y', 'std_x' and 'std_y'.
    """
    proj_x = image.sum(axis=1)
    proj_y = image.sum(axis=0)

    # stats
    avg_x = (bins_x*proj_x).sum()/proj_x.sum()
    avg_y = (bins_y*proj_y).sum()/proj_y.sum()

    std_x = (((bins_x*proj_x - avg_x)**2).sum()/proj_x.sum())**(1/2)
    std_y = (((bins_y*proj_y - avg_y)**2).sum()/proj_y.sum())**(1/2)


    return {'avg_x': avg_x,
            'avg_y': avg_y,
            'std_x': std_x,
            'std_y': std_y}


def plot_3d_scan_data(
        train_dset
        ):

    # reshape data into parameter 3D mesh:
    n_k = len(torch.unique(train_dset.params.squeeze(-1)[:,0]))
    n_v = len(torch.unique(train_dset.params.squeeze(-1)[:,1]))
    n_g = len(torch.unique(train_dset.params.squeeze(-1)[:,2]))
    image_shape = train_dset.images.shape
    params = train_dset.params.reshape((n_k, n_v, n_g, 3))
    images = train_dset.images.reshape((n_k, n_v, n_g, image_shape[-2], image_shape[-1]))

    # plot
    fig, ax = plt.subplots(n_v+n_g+1, n_k+1, figsize=((n_k+1)*2, (n_v+n_g+1)*2))

    ax[0, 0].set_axis_off()
    ax[0, 0].text(1, 0, '$k_1$ (1/m$^2$)', va='bottom', ha='right')
    for i in range(n_k):
        ax[0, i+1].set_axis_off()
        ax[0, i+1].text(0.5, 0, f'{params[i,0,0,0]:.1f}', va='bottom', ha='center')
        for j in range(n_v):
            for k in range(n_g):
                ax[2*j+k+1, i+1].imshow(images[i,j,k].T,
                                    origin = 'lower',
                                    #extent = extent,
                                    interpolation = 'none')
                ax[2*j+k+1, i+1].tick_params(bottom=False, left=False,
                                       labelbottom=False, labelleft=False)

                if j == 0:
                    v_lbl = "off"
                else: 
                    v_lbl = "on"
                if k == 0:
                    g_lbl = "off"
                else: 
                    g_lbl = "on"

                ax[2*j+k+1, 0].set_axis_off()
                ax[2*j+k+1, 0].text(1,0.5, f'T.D.C.: {v_lbl}\n DIPOLE: {g_lbl}', va='center', ha='right')
    
    return fig, ax