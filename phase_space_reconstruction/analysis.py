import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_scan_data(train_dset, test_dset, bins):
    """
    Plots train and test images from data sets sorted by quad strength.

    Parameters
    ----------
    train_dset: ImageDataset
        training dataset. train_dset.k is of size
        number_of_quad_strengts x number_of_images_per_quad_strength x 1
    
    test_dset: ImageDataset
        test dataset
    
    bins: array-like
        image binning. size should be the same as dset.images.shape[-1]

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
    extent = np.array([bins[0], bins[-1], bins[0], bins[-1]])*1e3
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
        
        ax[0,0].set_title('$k$ (1/m):')

        
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    print(f'image size = {(bins[-1]-bins[0])*1e3:.0f} x {(bins[-1]-bins[0])*1e3:.0f} mm')
    print('test samples boxed in orange')
    
    return fig, ax

def plot_predicted_screens(prediction_dset, train_dset, test_dset, bins):
    """
    Plots predictions (and measurements for reference)

    Parameters
    ----------
    prediction_dset: ImageDataset
        predicted screens dataset. prediction_dset.k is of size
        number_of_quad_strengts x number_of_images_per_quad_strength x 1

    train_dset: ImageDataset
        training dataset. 
    
    test_dset: ImageDataset
        test dataset
    
    bins: array-like
        image binning. size should be the same as dset.images.shape[-1]

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
    extent = np.array([bins[0], bins[-1], bins[0], bins[-1]])*1e3

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
    
    ax[0,0].set_title('$k$ (1/m):')

        
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    print(f'image size = {(bins[-1]-bins[0])*1e3:.0f} x {(bins[-1]-bins[0])*1e3:.0f} mm')
    print('test samples boxed in orange')
    
    return fig, ax