import h5py
import numpy as np

# process_folder(data_folder, True, overwrite=True)
import torch
from scipy import ndimage
from skimage import filters
from skimage.transform import downscale_local_mean


def process_images(base_fname, n_samples=1, downsample=5, threshold_multiplier=1.0):
    # generate list of filenames for processing
    vals = np.arange(-84, 84, 8)
    vals[0] = -83
    vals[-1] = 83

    fnames = [f"{base_fname}{ele:+}.h5".replace("+", "p").replace("-", "n") for ele in
              vals]

    all_images = []
    all_charges = []
    for fname in fnames:
        print(fname)
        w = 450
        images = []
        charges = []
        with h5py.File(fname, "r") as f:
            screen_center = f.attrs["screen_center"].reshape(1, -1)

            for i in range(f.attrs["nframes"] - 1):
                bbox = np.vstack([screen_center - w, screen_center + w]).T
                slices = [slice(*ele.astype(int)) for ele in bbox]
                im = f[f"{i}"]["raw"][slices[1], slices[0]]
                images.append(im)
                charges.append(f[f"{i}"].attrs["charge"])

        all_images.append(images)
        all_charges.append(charges)

    all_images = np.array(all_images)
    all_charges = np.array(all_charges) * 1e9

    # select 10 best images based on how close they are to the target charge of 0.9 nC
    target_charge = 0.9
    charge_diff = np.abs(all_charges - target_charge)

    # argsort based on differences
    sorted_diff_args = np.argsort(charge_diff, axis=1).squeeze()

    sorted_charges = np.empty_like(all_charges)
    sorted_images = np.empty_like(all_images)
    for i in range(len(all_charges)):
        sorted_charges[i] = all_charges[i][sorted_diff_args[i]]
        sorted_images[i] = all_images[i][sorted_diff_args[i]]

    # get training data
    train_charges = sorted_charges[:, :n_samples]  # nC
    train_images = sorted_images[:, :n_samples]

    # apply filters and thresholding to training images
    thresh = filters.threshold_triangle(train_images[0, 0]) * threshold_multiplier
    t_images = np.clip(train_images - thresh, 0, None)

    def apply_filter(X):
        return ndimage.minimum_filter(X, size=3)

    filtered_images = np.empty_like(t_images)
    for ii in range(t_images.shape[0]):
        for jj in range(t_images.shape[1]):
            filtered_images[ii, jj] = apply_filter(t_images[ii, jj])

    # downsample data by a factor of 5
    scale_factor = downsample
    filtered_images = downscale_local_mean(filtered_images, (1, 1, scale_factor, scale_factor))
    train_images = filtered_images

    # normalize images such that the sum is 1
    train_images = train_images[:, :] / np.expand_dims(
        train_images.sum(axis=-1).sum(axis=-1), axis=[-2, -1]
    )

    # convert quad counts to geometric strengths
    count_to_gradient = 0.00893  # T/m
    gradients = vals * count_to_gradient
    beam_momentum = 65e-3  # GeV/c
    magnetic_rigity = 33.3564 * beam_momentum / 10  # T-m

    kappa = gradients / magnetic_rigity  # 1/m^2

    # reshape and copy kappa for 1-1 shapes
    kappa = np.repeat(np.expand_dims(kappa, axis=0), n_samples, 0).T

    # return pixel coordinates
    px_coords = torch.arange(train_images.shape[-1]) - train_images.shape[-1] / 2
    real_coords = px_coords * 3.787493924766505e-05 * scale_factor
    xx = torch.meshgrid(real_coords, real_coords)
    

    torch.save(torch.tensor(kappa), "kappa.pt")
    torch.save(torch.tensor(train_images), "train_images.pt")
    torch.save(torch.tensor(train_charges), "train_charges.pt")
    torch.save(xx, "xx.pt")


def import_images():
    return torch.load("kappa.pt"), torch.load("train_images.pt"), torch.load(
        "train_charges.pt"), torch.load("xx.pt")