import logging
import os

import numpy as np
import torch

from bmadx.bmad_torch.track_torch import Beam, TorchQuadrupole, TorchDrift, TorchLattice

from phase_space_reconstruction import modeling
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.losses import MENTLoss
from phase_space_reconstruction.modeling import ImageDataset
from torch.utils.data import DataLoader
from torchensemble import SnapshotEnsembleRegressor

logging.basicConfig(level=logging.INFO)


def create_quad_scan_beamline():
    q1 = TorchQuadrupole(torch.tensor(0.12), torch.tensor(0.0), 5)
    d1 = TorchDrift(torch.tensor(3.38 - 0.12/2))

    lattice = TorchLattice([q1, d1])
    return lattice


def create_ensemble(bins, bandwidth):
    lattice = create_quad_scan_beamline()
    diagnostic = ImageDiagnostic(bins, bandwidth=bandwidth)
    # create NN beam
    n_particles = 100000
    nn_transformer = modeling.NNTransform(2, 20, output_scale=1e-3)
    nn_beam = modeling.InitialBeam(
        nn_transformer,
        torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
        n_particles,
        p0c=torch.tensor(63.0e6),
    )
    module_kwags = {"lattice": lattice, "diagnostic": diagnostic, "beam": nn_beam}

    ensemble = SnapshotEnsembleRegressor(
        estimator=modeling.PhaseSpaceReconstructionModel,
        estimator_args=module_kwags,
        n_estimators=4,
    )

    return ensemble


def load_data(folder, tkwargs):
    image_data = np.load(f"{folder}/image_data.npy")
    quad_strengths = np.load(f"{folder}/quad_strength.npy")

    # get mesh for images, assuming that the image center is the origin
    px_scale = 2.081e-5 * 2  ## m / px
    image_size = image_data.shape[-1]
    x = (np.arange(image_size) - image_size / 2) * px_scale
    xx = np.stack(np.meshgrid(x, x))
    
    # normalize each image
    sums = image_data.sum(keepdims=True, axis=-1).sum(keepdims=True, axis=-2)
    image_data = image_data / sums

    # convert to torch objects
    quad_strengths = torch.tensor(quad_strengths, **tkwargs)
    image_data = torch.tensor(image_data, **tkwargs)
    x = torch.tensor(x, **tkwargs)
    xx = torch.tensor(xx, **tkwargs)

    n_samples = image_data.shape[1]
    quad_strengths = quad_strengths.unsqueeze(1).repeat(
        1, n_samples
    ).unsqueeze(-1) * 1.0

    return quad_strengths, image_data, x, xx


def create_datasets(all_k, all_images, save_dir):
    # select data points with small # of outliers
    all_k = all_k[:,[0,4,3],:]
    all_images = all_images[:,[0,4,3],:]
    
    train_dset = ImageDataset(all_k[:20:2], all_images[:20:2])
    test_dset = ImageDataset(all_k[1:20:2], all_images[1:20:2])
    torch.save(train_dset, save_dir + "/train.dset")
    torch.save(test_dset, save_dir + "/test.dset")

    return train_dset, test_dset


if __name__ == "__main__":
    folder = ""

    base_dir = "/global/cfs/cdirs/m669/rroussel/phase_space_reconstruction"
    save_dir = base_dir + "/mse_scale_1_l_1e10"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    tkwargs = {"dtype": torch.float}
    all_k, all_images, bins, xx = load_data(base_dir,tkwargs)
    train_dset, test_dset = create_datasets(all_k, all_images, save_dir)
    print(len(train_dset))

    train_dataloader = DataLoader(train_dset, batch_size=int(len(train_dset)/2), shuffle=True)
    test_dataloader = DataLoader(test_dset, shuffle=True)

    bin_width = bins[1] - bins[0]
    bandwidth = bin_width / 2
    ensemble = create_ensemble(bins, bandwidth)

    criterion = MENTLoss(
        torch.tensor(1e10), 
        gamma_=torch.tensor(1.0), 
        beta_=torch.tensor(0.0),
        alpha_=torch.tensor(0.0)
    )

    ensemble.set_criterion(criterion)

    n_epochs = 1000
    ensemble.set_optimizer("Adam", lr=1e-2)
    # with torch.autograd.detect_anomaly():
    ensemble.fit(
        train_dataloader, epochs=n_epochs, save_dir=save_dir, lr_clip=[1e-5, 10]
    )
    torch.save(criterion.loss_record, save_dir + "/loss_log.pt")
